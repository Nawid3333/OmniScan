"""Chat clients for hosted LLM APIs: OpenAI's chat completions (and every server that speaks it: OpenRouter,
Groq, DeepSeek, Mistral, LM Studio, vLLM, ...) and Anthropic's messages API (Claude).

`ApiChatClient.chat` takes the same arguments as `OllamaClient.chat` and returns the same `ChatResponse`, so the
translator and the judge use a profile with `endpoint = "openai"` or `"anthropic"` without knowing it: their
Ollama-style messages (`images`: base64 JPEG/PNG next to the text) are converted, `format` asks for a JSON reply
where the API has that (OpenAI's json_object; Claude follows the prompt), and `think` / `keep_alive` / Ollama
options other than the temperature do not apply. Retries mirror the Ollama client: transport errors, 429 and 5xx
back off and retry; an exhausted 429 (or a spent quota) raises `ApiRateLimitError`, which the fallback machinery
treats like Ollama's rate limit. A model that refuses a temperature other than its default (OpenAI's reasoning
models) is asked again without one. Every request counts against the run's cloud budget (llm/budget.py).

`provider_client` picks the client for a profile's endpoint: the given Ollama client for "local" and "cloud",
else an API client with the key from the environment or secrets.env (`core.config.api_key`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

import httpx

from omniscan.core.config import api_key
from omniscan.llm.budget import spend_cloud_request
from omniscan.llm.errors import OllamaError, OllamaRateLimitError
from omniscan.llm.ollama import ChatResponse, backoff_delay, retry_after_s

ApiKind = Literal["openai", "anthropic"]
API_KINDS: tuple[ApiKind, ...] = ("openai", "anthropic")
DEFAULT_BASE_URLS: dict[ApiKind, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}
DEFAULT_KEY_ENVS: dict[ApiKind, str] = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MAX_TOKENS = 8192  # the messages API needs a reply cap; a 30-region chunk's JSON stays far below it
_TIMEOUT_S = 600.0
_BODY_TRUNC_CHARS = 2000
_QUOTA_MARKERS = ("insufficient_quota", "credit balance is too low", "billing")


class ApiError(OllamaError):
    """An API request failed (bad key, a rejected request, retries exhausted)."""


class ApiRateLimitError(OllamaRateLimitError):
    """The API answered 429 until the retries ran out, or the account's quota or credit is spent."""


_SHARED_HTTP: httpx.Client | None = None


def _shared_http() -> httpx.Client:
    """One connection pool for every API client of the process (they are made per profile and run)."""
    global _SHARED_HTTP
    if _SHARED_HTTP is None:
        _SHARED_HTTP = httpx.Client(timeout=_TIMEOUT_S)
    return _SHARED_HTTP


class ApiChatClient:
    """`chat` against the OpenAI or Anthropic API with one key (see the module docstring)."""

    def __init__(
        self,
        kind: ApiKind,
        *,
        key_env: str | None = None,
        base_url: str | None = None,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        key_lookup: Callable[[str], Any] = api_key,
    ) -> None:
        self.kind: ApiKind = kind
        self.key_env = key_env or DEFAULT_KEY_ENVS[kind]
        self.base_url = (base_url or DEFAULT_BASE_URLS[kind]).rstrip("/")
        self._http = http if http is not None else _shared_http()
        self._sleep = sleep
        self._key_lookup = key_lookup
        self._no_temperature: set[str] = set()  # models that refused a temperature other than their default

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        cloud: bool = False,
        format: dict[str, Any] | Literal["json"] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
        think: bool | None = None,
        max_retries: int = 5,
    ) -> ChatResponse:
        """Send one chat request and lift the reply into a ChatResponse (`cloud`, `keep_alive` and `think` are
        Ollama's and do not apply)."""
        del cloud, keep_alive, think
        key = self._key_lookup(self.key_env)
        if key is None:
            raise ApiError(
                f"{self.key_env} is not set: add your {self.kind} API key in Settings → Profiles, in "
                "~/.config/omniscan/secrets.env, or as an environment variable"
            )
        spend_cloud_request(model)
        temperature = (options or {}).get("temperature")
        while True:
            send_temperature = temperature if model not in self._no_temperature else None
            body, url, headers = self._request(
                model, messages, format, send_temperature, key.get_secret_value()
            )
            try:
                data = self._post(url, headers, body, max_retries)
            except ApiError as error:
                if (
                    send_temperature is not None
                    and error.status_code == 400
                    and "temperature" in (error.body or "")
                ):
                    self._no_temperature.add(model)
                    continue
                raise
            return self._response(model, data)

    def _request(
        self,
        model: str,
        messages: list[dict[str, Any]],
        format: dict[str, Any] | Literal["json"] | None,
        temperature: float | None,
        key: str,
    ) -> tuple[dict[str, Any], str, dict[str, str]]:
        """The request body, URL and headers for this API."""
        if self.kind == "anthropic":
            system = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
            body: dict[str, Any] = {
                "model": model,
                "max_tokens": ANTHROPIC_MAX_TOKENS,
                "messages": [_anthropic_message(m) for m in messages if m.get("role") != "system"],
            }
            if system:
                body["system"] = system
            if temperature is not None:
                body["temperature"] = temperature
            headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
            return body, f"{self.base_url}/messages", headers
        body = {"model": model, "messages": [_openai_message(m) for m in messages]}
        if temperature is not None:
            body["temperature"] = temperature
        # json_object needs the word "json" in the messages; OmniScan's JSON prompts all say it
        if format is not None and any("json" in str(m.get("content", "")).lower() for m in messages):
            body["response_format"] = {"type": "json_object"}
        return body, f"{self.base_url}/chat/completions", {"Authorization": f"Bearer {key}"}

    def _post(
        self, url: str, headers: dict[str, str], body: dict[str, Any], max_retries: int
    ) -> dict[str, Any]:
        """POST with retries on transport errors, 429 and 5xx; ApiError / ApiRateLimitError when it fails."""
        status: int | None = None
        text: str | None = None
        for attempt in range(max_retries):
            try:
                response = self._http.post(url, headers=headers, json=body)
            except httpx.TransportError as exc:
                if attempt == max_retries - 1:
                    raise ApiError(
                        f"{self.kind} API unreachable after {max_retries} attempts: {exc}"
                    ) from exc
                self._sleep(backoff_delay(attempt))
                continue
            if response.is_success:
                return dict(response.json())
            status, text = response.status_code, response.text[:_BODY_TRUNC_CHARS]
            if status == 429 and any(marker in text.lower() for marker in _QUOTA_MARKERS):
                raise ApiRateLimitError(
                    f"{self.kind} API: the account's quota or credit is spent", status_code=429, body=text
                )
            if status == 429 or status >= 500:
                if attempt == max_retries - 1:
                    break
                self._sleep(backoff_delay(attempt, retry_after_s(response)))
                continue
            reason = "the API key was rejected" if status in (401, 403) else f"HTTP {status}"
            raise ApiError(f"{self.kind} API: {reason} on {url}", status_code=status, body=text)
        if status == 429:
            raise ApiRateLimitError(
                f"{self.kind} API rate limit (HTTP 429): retries exhausted", status_code=429, body=text
            )
        raise ApiError(
            f"{self.kind} API: HTTP {status} on {url} after {max_retries} attempts",
            status_code=status,
            body=text,
        )

    def _response(self, model: str, data: dict[str, Any]) -> ChatResponse:
        """The reply's text and token counts as a ChatResponse."""
        usage = data.get("usage") or {}
        if self.kind == "anthropic":
            content = "".join(
                str(part.get("text", "")) for part in data.get("content", []) if part.get("type") == "text"
            )
            return ChatResponse(
                content=content,
                model=str(data.get("model", model)),
                done=True,
                total_duration_ns=None,
                prompt_eval_count=usage.get("input_tokens"),
                eval_count=usage.get("output_tokens"),
                raw=data,
            )
        choices = data.get("choices") or [{}]
        message = choices[0].get("message") or {}
        return ChatResponse(
            content=str(message.get("content") or ""),
            model=str(data.get("model", model)),
            done=True,
            total_duration_ns=None,
            prompt_eval_count=usage.get("prompt_tokens"),
            eval_count=usage.get("completion_tokens"),
            raw=data,
        )


def _media_type(data: str) -> str:
    """The media type of a base64 image (OmniScan sends JPEG; PNG is recognised too)."""
    return "image/png" if data.startswith("iVBOR") else "image/jpeg"


def _openai_message(message: dict[str, Any]) -> dict[str, Any]:
    """An Ollama-style message as an OpenAI one: images become image_url parts (data URLs) after the text."""
    images = message.get("images") or []
    if not images:
        return {"role": message["role"], "content": message.get("content", "")}
    parts: list[dict[str, Any]] = [{"type": "text", "text": message.get("content", "")}]
    parts += [
        {"type": "image_url", "image_url": {"url": f"data:{_media_type(image)};base64,{image}"}}
        for image in images
    ]
    return {"role": message["role"], "content": parts}


def _anthropic_message(message: dict[str, Any]) -> dict[str, Any]:
    """An Ollama-style message as an Anthropic one: images become base64 image blocks before the text."""
    images = message.get("images") or []
    if not images:
        return {"role": message["role"], "content": message.get("content", "")}
    parts: list[dict[str, Any]] = [
        {"type": "image", "source": {"type": "base64", "media_type": _media_type(image), "data": image}}
        for image in images
    ]
    parts.append({"type": "text", "text": message.get("content", "")})
    return {"role": message["role"], "content": parts}


def key_env_for(endpoint: str, key_env: str | None = None) -> str | None:
    """The variable holding the key an endpoint needs: `key_env`, else the vendor default; None for Ollama."""
    if endpoint == "openai" or endpoint == "anthropic":
        return key_env or DEFAULT_KEY_ENVS[endpoint]
    return None


def provider_client[C](
    client: C, endpoint: str, *, base_url: str | None = None, key_env: str | None = None
) -> C | ApiChatClient:
    """The client a profile's requests go to: `client` itself (Ollama) for "local" and "cloud", else an API client
    for "openai" / "anthropic" with the profile's base URL and key variable."""
    if endpoint == "openai":
        return ApiChatClient("openai", key_env=key_env, base_url=base_url)
    if endpoint == "anthropic":
        return ApiChatClient("anthropic", key_env=key_env, base_url=base_url)
    return client

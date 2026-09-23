"""httpx client for Ollama's native /api/chat and /api/ps endpoints (local or cloud).

This is a transport client only — prompts, images and translation logic live elsewhere.
Retries cover transport errors, HTTP 429 and HTTP 5xx with exponential backoff, and any
other HTTP status fails fast after exactly one attempt. Ollama Cloud's account-wide
session-usage cap answers with HTTP 429 for the rest of the session, so when retries on
a 429 are exhausted this client raises `OllamaRateLimitError` — a distinct, loud failure
that callers must surface, never swallow (a past run lost 71 turns to exactly that).
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from omniscan.core.config import OllamaConfig, Secrets

_MAX_BACKOFF_S = 30.0
_BACKOFF_BASE_S = 1.0
_JITTER_FRACTION = 0.25
_BODY_TRUNC_CHARS = 2000
_PS_RETRIES = 5
_REPLY_TOKENS = 4096  # room left for the answer when sizing num_ctx to a prompt


@dataclass(slots=True)
class ChatResponse:
    content: str
    model: str
    done: bool
    total_duration_ns: int | None
    prompt_eval_count: int | None
    eval_count: int | None
    raw: dict[str, Any]  # the full decoded JSON body, for anything callers need that isn't lifted above


@dataclass(slots=True)
class RunningModel:
    name: str
    size_vram: int
    expires_at: str | None


class OllamaError(RuntimeError):
    """Ollama request failed (non-retryable status, or retryable failures exhausted)."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class OllamaRateLimitError(OllamaError):
    """Raised only when retries are exhausted on an HTTP 429. status_code is always 429."""


class OllamaClient:
    def __init__(
        self,
        cfg: OllamaConfig,
        secrets: Secrets,
        *,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._cfg = cfg
        self._secrets = secrets
        self._sleep = sleep
        self._owns_http = http is None
        self._http = http if http is not None else httpx.Client(timeout=cfg.request_timeout_s)

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
        """Send one non-streaming /api/chat request and lift the fields callers use."""
        base, headers = self._endpoint(cloud)
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if format is not None:
            body["format"] = format
        options = self._with_num_ctx(model, messages, options, cloud=cloud)
        if options is not None:
            body["options"] = options
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        if think is not None:
            body["think"] = think
        data = self._request(
            "POST", f"{base}/api/chat", headers=headers, json_body=body, max_retries=max_retries
        )
        message = data.get("message") or {}
        return ChatResponse(
            content=message.get("content", ""),
            model=data["model"],
            done=data["done"],
            total_duration_ns=data.get("total_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
            raw=data,
        )

    def ps(self, *, cloud: bool = False) -> list[RunningModel]:
        """List models currently loaded on the endpoint (local or cloud)."""
        base, headers = self._endpoint(cloud)
        data = self._request("GET", f"{base}/api/ps", headers=headers, max_retries=_PS_RETRIES)
        return [
            RunningModel(name=m["name"], size_vram=m.get("size_vram", 0), expires_at=m.get("expires_at"))
            for m in data.get("models", [])
        ]

    def close(self) -> None:
        """Close the internal httpx.Client (an injected one stays open; the caller owns it)."""
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _with_num_ctx(
        self, model: str, messages: list[dict[str, Any]], options: dict[str, Any] | None, *, cloud: bool
    ) -> dict[str, Any] | None:
        """`options` plus a `num_ctx` for a local model: the configured floor, or more for a long prompt.

        Two tokens per prompt character (an upper bound for CJK and English alike) plus room for the reply,
        rounded up to 4096, so a whole-chapter prompt is never truncated. Every other request gets the same
        floor, which keeps the loaded model (Ollama reloads it when num_ctx changes)."""
        floor = self._cfg.num_ctx
        if floor == 0 or cloud or model.endswith((":cloud", "-cloud")):
            return options
        if options is not None and "num_ctx" in options:
            return options
        chars = sum(len(str(message.get("content", ""))) for message in messages)
        needed = -(-(2 * chars + _REPLY_TOKENS) // 4096) * 4096
        return {**(options or {}), "num_ctx": max(floor, needed)}

    def _endpoint(self, cloud: bool) -> tuple[str, dict[str, str]]:
        """Resolve base URL and auth headers; cloud requires the API key before any I/O."""
        if not cloud:
            return self._cfg.local_url, {}
        key = self._secrets.ollama_api_key
        if key is None:
            raise ValueError("OLLAMA_API_KEY is not set (needed for cloud=True)")
        return self._cfg.cloud_url, {"Authorization": f"Bearer {key.get_secret_value()}"}

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        max_retries: int,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue up to `max_retries` attempts; retry transport errors, 429 and 5xx."""
        status: int | None = None
        body: str | None = None
        for attempt in range(max_retries):
            try:
                response = (
                    self._http.request(method, url, headers=headers)
                    if json_body is None
                    else self._http.request(method, url, headers=headers, json=json_body)
                )
            except httpx.TransportError as exc:
                if attempt == max_retries - 1:
                    raise OllamaError(f"Ollama unreachable after {max_retries} attempts: {exc}") from exc
                self._sleep(self._backoff_delay(attempt))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                status = response.status_code
                body = response.text[:_BODY_TRUNC_CHARS]
                if attempt == max_retries - 1:
                    break
                retry_after = self._retry_after_s(response)
                self._sleep(self._backoff_delay(attempt, retry_after))
                continue
            if response.is_success:
                return dict(response.json())
            raise OllamaError(
                f"Ollama HTTP {response.status_code} on {method} {url}",
                status_code=response.status_code,
                body=response.text[:_BODY_TRUNC_CHARS],
            )
        if status == 429:
            raise OllamaRateLimitError(
                "Ollama Cloud rate limit (HTTP 429): retries exhausted — the account's session usage cap is "
                "reached, so every call will keep failing until the session resets; stop the run",
                status_code=429,
                body=body,
            )
        raise OllamaError(
            f"Ollama HTTP {status} on {method} {url} after {max_retries} attempts",
            status_code=status,
            body=body,
        )

    @staticmethod
    def _retry_after_s(response: httpx.Response) -> float | None:
        """Parse a 429's Retry-After header as whole seconds; anything else means no hint."""
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return float(int(value))
        except ValueError:
            return None

    @staticmethod
    def _backoff_delay(attempt: int, retry_after_s: float | None = None) -> float:
        """Exponential backoff with jitter; a 429's Retry-After takes precedence over the curve."""
        delay = min(_BACKOFF_BASE_S * (2**attempt), _MAX_BACKOFF_S)
        if retry_after_s is not None:
            delay = max(delay, retry_after_s)
        return delay + random.uniform(0, delay * _JITTER_FRACTION)

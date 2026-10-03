"""Hosted LLM APIs (llm/api.py), the cloud request budget (llm/budget.py) and how profiles reach them (#41)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from omniscan.core.config import OllamaConfig, Secrets, api_key
from omniscan.core.schemas import BBox, Region
from omniscan.llm.api import ApiChatClient, ApiError, ApiRateLimitError, key_env_for, provider_client
from omniscan.llm.budget import CloudBudgetError, cloud_budget, cloud_requests_spent
from omniscan.llm.errors import OllamaRateLimitError
from omniscan.llm.ollama import OllamaClient
from omniscan.translate.profiles import TranslationProfile
from omniscan.translate.run import run_profile

JPEG = "/9j/4AAQSkZJRgABAQ"  # the start of a base64 JPEG
PNG = "iVBORw0KGgoAAAANSUhEUg"


def _key(value: str | None = "sk-test") -> Any:
    """A key lookup that never touches the environment or secrets.env."""
    return lambda _name: None if value is None else SecretStr(value)


class Recorder:
    """An httpx transport answering with `replies` in turn (the last one repeats) and keeping every request."""

    def __init__(self, *replies: httpx.Response) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def _client(kind: str, recorder: Recorder, **kwargs: Any) -> ApiChatClient:
    return ApiChatClient(
        kind,  # type: ignore[arg-type]
        http=httpx.Client(transport=httpx.MockTransport(recorder)),
        sleep=lambda _s: None,
        key_lookup=kwargs.pop("key_lookup", _key()),
        **kwargs,
    )


def _openai_reply(text: str = '{"translations": []}') -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "gpt-5.4-mini-2026",
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        },
    )


def _anthropic_reply(text: str = "Hello") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": text}, {"type": "tool_use", "id": "x"}],
            "usage": {"input_tokens": 80, "output_tokens": 12},
        },
    )


MESSAGES = [
    {"role": "system", "content": "Translate. Answer with JSON only."},
    {"role": "user", "content": "page 1", "images": [JPEG, PNG]},
]


def test_openai_request_carries_the_key_images_and_json_mode() -> None:
    recorder = Recorder(_openai_reply("{}"))
    response = _client("openai", recorder).chat(
        "gpt-5.4-mini",
        MESSAGES,
        format={"type": "object"},
        options={"temperature": 0.3, "num_ctx": 8192},
        think=False,
    )
    request = recorder.requests[0]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer sk-test"
    body = recorder.body()
    assert body["model"] == "gpt-5.4-mini" and body["temperature"] == 0.3
    assert body["response_format"] == {"type": "json_object"} and "num_ctx" not in json.dumps(body)
    assert body["messages"][0] == {"role": "system", "content": "Translate. Answer with JSON only."}
    parts = body["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "page 1"}
    assert parts[1]["image_url"]["url"] == f"data:image/jpeg;base64,{JPEG}"
    assert parts[2]["image_url"]["url"] == f"data:image/png;base64,{PNG}"
    assert (response.content, response.model, response.prompt_eval_count, response.eval_count) == (
        "{}",
        "gpt-5.4-mini-2026",
        120,
        30,
    )


def test_openai_compatible_server_and_plain_text_requests() -> None:
    recorder = Recorder(_openai_reply("Hi"))
    client = _client(
        "openai", recorder, base_url="https://openrouter.ai/api/v1/", key_env="OPENROUTER_API_KEY"
    )
    client.chat(
        "deepseek/deepseek-chat", [{"role": "user", "content": "translate this"}], format={"type": "object"}
    )
    assert str(recorder.requests[0].url) == "https://openrouter.ai/api/v1/chat/completions"
    assert "response_format" not in recorder.body()  # json_object needs the word JSON in the prompt
    assert client.key_env == "OPENROUTER_API_KEY"


def test_anthropic_request_moves_the_system_prompt_and_puts_images_first() -> None:
    recorder = Recorder(_anthropic_reply())
    response = _client("anthropic", recorder).chat(
        "claude-sonnet-5-5", MESSAGES, options={"temperature": 0.2}
    )
    request = recorder.requests[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "sk-test" and request.headers["anthropic-version"] == "2023-06-01"
    body = recorder.body()
    assert body["system"] == "Translate. Answer with JSON only." and body["max_tokens"] == 8192
    assert body["temperature"] == 0.2 and len(body["messages"]) == 1
    blocks = body["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["image", "image", "text"]
    assert blocks[0]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": JPEG}
    assert blocks[1]["source"]["media_type"] == "image/png"
    assert (response.content, response.prompt_eval_count, response.eval_count) == ("Hello", 80, 12)


def test_a_missing_key_fails_before_any_request_and_names_the_variable() -> None:
    recorder = Recorder(_openai_reply())
    with pytest.raises(ApiError, match="OPENAI_API_KEY is not set"):
        _client("openai", recorder, key_lookup=_key(None)).chat("gpt", [{"role": "user", "content": "x"}])
    assert recorder.requests == []


def test_rate_limits_retry_then_count_as_the_rate_limit() -> None:
    limited = Recorder(httpx.Response(429, json={"error": {"message": "slow down"}}))
    with pytest.raises(ApiRateLimitError, match="rate limit") as caught:
        _client("openai", limited).chat("gpt", [{"role": "user", "content": "x"}], max_retries=3)
    assert len(limited.requests) == 3 and isinstance(caught.value, OllamaRateLimitError)
    spent = Recorder(httpx.Response(429, json={"error": {"code": "insufficient_quota"}}))
    with pytest.raises(ApiRateLimitError, match="quota or credit"):
        _client("openai", spent).chat("gpt", [{"role": "user", "content": "x"}])
    assert len(spent.requests) == 1  # a spent quota does not come back by waiting
    flaky = Recorder(httpx.Response(529, text="overloaded"), _anthropic_reply("ok"))
    assert _client("anthropic", flaky).chat("claude", [{"role": "user", "content": "x"}]).content == "ok"


def test_a_rejected_key_fails_at_once() -> None:
    rejected = Recorder(httpx.Response(401, json={"error": "invalid x-api-key"}))
    with pytest.raises(ApiError, match="the API key was rejected") as caught:
        _client("anthropic", rejected).chat("claude", [{"role": "user", "content": "x"}])
    assert caught.value.status_code == 401 and len(rejected.requests) == 1


def test_a_model_that_refuses_a_temperature_is_asked_again_without_one() -> None:
    refused = httpx.Response(
        400,
        json={"error": {"message": "Unsupported value: 'temperature' does not support 0.3 with this model."}},
    )
    recorder = Recorder(refused, _openai_reply("ok"))
    client = _client("openai", recorder)
    assert (
        client.chat("gpt-5.4", [{"role": "user", "content": "x"}], options={"temperature": 0.3}).content
        == "ok"
    )
    assert "temperature" in recorder.body(0) and "temperature" not in recorder.body(1)
    client.chat("gpt-5.4", [{"role": "user", "content": "y"}], options={"temperature": 0.3})
    assert "temperature" not in recorder.body(2)  # remembered for the model
    other = Recorder(httpx.Response(400, json={"error": {"message": "bad request"}}))
    with pytest.raises(ApiError, match="HTTP 400"):
        _client("openai", other).chat("gpt", [{"role": "user", "content": "x"}], options={"temperature": 0.3})


def test_provider_client_routes_by_endpoint() -> None:
    ollama = object()
    assert provider_client(ollama, "local") is ollama and provider_client(ollama, "cloud") is ollama
    routed = provider_client(ollama, "openai", base_url="http://localhost:1234/v1", key_env="LMSTUDIO_KEY")
    assert isinstance(routed, ApiChatClient)
    assert (routed.kind, routed.base_url, routed.key_env) == (
        "openai",
        "http://localhost:1234/v1",
        "LMSTUDIO_KEY",
    )
    claude = provider_client(ollama, "anthropic")
    assert isinstance(claude, ApiChatClient) and claude.key_env == "ANTHROPIC_API_KEY"
    assert key_env_for("local") is None and key_env_for("openai") == "OPENAI_API_KEY"
    assert key_env_for("anthropic", "MY_KEY") == "MY_KEY"


def test_api_key_reads_the_environment_then_the_secrets_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "secrets.env"
    env_file.write_text("OPENAI_API_KEY=from-file\nEMPTY_KEY=\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    key = api_key("OPENAI_API_KEY", env_file=env_file)
    assert key is not None and key.get_secret_value() == "from-file"
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    key = api_key("OPENAI_API_KEY", env_file=env_file)
    assert key is not None and key.get_secret_value() == "from-env"
    assert api_key("EMPTY_KEY", env_file=env_file) is None
    assert api_key("NOT_THERE", env_file=tmp_path / "missing.env") is None


def test_the_cloud_budget_counts_cloud_requests_only() -> None:
    recorder = Recorder(httpx.Response(200, json={"model": "m", "message": {"content": "ok"}, "done": True}))
    ollama = OllamaClient(
        OllamaConfig(),
        Secrets(),
        http=httpx.Client(transport=httpx.MockTransport(recorder)),
        sleep=lambda _s: None,
    )
    api = _client("openai", Recorder(_openai_reply("ok")))
    assert cloud_requests_spent() is None
    with cloud_budget(2):
        ollama.chat("translategemma:27b", [{"role": "user", "content": "x"}])  # local: free
        ollama.chat("gemma4:31b-cloud", [{"role": "user", "content": "x"}])
        api.chat("gpt", [{"role": "user", "content": "x"}])
        assert cloud_requests_spent() == 2
        with pytest.raises(CloudBudgetError, match="cloud request budget spent: 2") as caught:
            ollama.chat("kimi-k3:cloud", [{"role": "user", "content": "x"}])
        assert isinstance(caught.value, OllamaRateLimitError) and caught.value.status_code == 429
        ollama.chat("gemma4:12b", [{"role": "user", "content": "x"}])  # local models still run
    assert len(recorder.requests) == 3 and cloud_requests_spent() is None
    with cloud_budget(0):  # no limit
        for _ in range(3):
            api.chat("gpt", [{"role": "user", "content": "x"}])


def test_a_profile_with_an_api_endpoint_translates_through_that_api(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(_openai_reply('{"translations": [{"id": "r0001", "text": "Hello"}]}'))

    class NoOllama:
        def chat(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("an API profile must not ask Ollama")

    monkeypatch.setattr(
        "omniscan.llm.api._shared_http", lambda: httpx.Client(transport=httpx.MockTransport(recorder))
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    profile = TranslationProfile(name="gpt", endpoint="openai", model="gpt-5.4-mini", style="chat_json")
    region = Region(
        id="r0001", slice_index=0, kind="bubble_text", bbox=BBox(x0=0, y0=0, x1=10, y1=10), text="안녕"
    )
    run = run_profile(NoOllama(), profile, [region], [])
    assert [(c.region_id, c.text) for c in run.candidates] == [("r0001", "Hello")]
    assert recorder.body()["model"] == "gpt-5.4-mini"


def test_profiles_check_their_provider_fields() -> None:
    TranslationProfile(
        name="ok",
        endpoint="openai",
        model="m",
        style="chat_json",
        base_url="https://x/v1",
        api_key_env="X_KEY",
    )
    with pytest.raises(ValueError, match='need endpoint = "openai" or "anthropic"'):
        TranslationProfile(name="a", endpoint="local", model="m", style="chat_json", base_url="https://x/v1")
    with pytest.raises(ValueError, match="http:// or https://"):
        TranslationProfile(name="b", endpoint="openai", model="m", style="chat_json", base_url="ftp://x")
    with pytest.raises(ValueError, match="environment variable name"):
        TranslationProfile(name="c", endpoint="anthropic", model="m", style="chat_json", api_key_env="MY-KEY")

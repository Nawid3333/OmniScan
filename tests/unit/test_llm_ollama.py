"""Unit tests for the Ollama client — all HTTP via httpx.MockTransport, no network."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from omniscan.core.config import OllamaConfig, Secrets
from omniscan.llm.ollama import ChatResponse, OllamaClient, OllamaError, OllamaRateLimitError, RunningModel

API_KEY = "test-key-123"
CHAT_BODY = {
    "model": "gemma3:27b",
    "created_at": "2026-09-18T12:00:00.000000Z",
    "message": {"role": "assistant", "content": "Hello, world!"},
    "done": True,
    "done_reason": "stop",
    "total_duration": 8_912_345_678,
    "prompt_eval_count": 42,
    "eval_count": 17,
}
PS_BODY = {
    "models": [
        {
            "name": "gemma3:27b",
            "model": "gemma3:27b",
            "size": 1_700_000_000,
            "size_vram": 1_700_000_000,
            "expires_at": "2026-09-18T12:05:00.000000Z",
        },
        {"name": "qwen3:8b", "model": "qwen3:8b", "size": 500_000_000, "size_vram": 0},
    ]
}


class Recorder:
    """MockTransport handler that replays scripted responses and records every request."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self.sleeps: list[float] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content) if request.content else {})
        return self.responses.pop(0) if self.responses else httpx.Response(200, json=CHAT_BODY)

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)


def make_client(
    responses: list[httpx.Response] | None = None,
    *,
    api_key: str | None = API_KEY,
) -> tuple[OllamaClient, Recorder]:
    rec = Recorder(responses or [httpx.Response(200, json=CHAT_BODY)])
    http = httpx.Client(transport=httpx.MockTransport(rec.handler))
    # model_construct bypasses all settings sources (env file + env vars) so tests stay deterministic.
    secrets = Secrets.model_construct(ollama_api_key=SecretStr(api_key) if api_key is not None else None)
    return OllamaClient(OllamaConfig(), secrets, http=http, sleep=rec.sleep), rec


def test_chat_happy_path_maps_every_field() -> None:
    client, rec = make_client()
    result = client.chat("gemma3:27b", [{"role": "user", "content": "hi"}])
    assert isinstance(result, ChatResponse)
    assert result.content == "Hello, world!"
    assert result.model == "gemma3:27b"
    assert result.done is True
    assert result.total_duration_ns == 8_912_345_678
    assert result.prompt_eval_count == 42
    assert result.eval_count == 17
    assert result.raw == CHAT_BODY
    assert rec.calls == 1
    assert rec.requests[0].url == "http://localhost:11434/api/chat"


def test_chat_cloud_uses_cloud_url_and_bearer_auth() -> None:
    client, rec = make_client()
    client.chat("gemma3:27b", [{"role": "user", "content": "hi"}], cloud=True)
    assert rec.requests[0].url == "https://ollama.com/api/chat"
    assert rec.requests[0].headers["Authorization"] == f"Bearer {API_KEY}"


def test_chat_cloud_without_api_key_makes_no_request() -> None:
    client, rec = make_client(api_key=None)
    with pytest.raises(ValueError, match="OLLAMA_API_KEY is not set"):
        client.chat("gemma3:27b", [{"role": "user", "content": "hi"}], cloud=True)
    assert rec.calls == 0


def test_chat_retries_500s_then_succeeds() -> None:
    client, rec = make_client(
        [
            httpx.Response(500, text="boom"),
            httpx.Response(500, text="boom"),
            httpx.Response(200, json=CHAT_BODY),
        ]
    )
    result = client.chat("gemma3:27b", [{"role": "user", "content": "hi"}])
    assert result.content == "Hello, world!"
    assert rec.calls == 3
    assert len(rec.sleeps) == 2
    assert all(delay > 0 for delay in rec.sleeps)


def test_chat_429_exhausts_retries_and_raises_rate_limit_error() -> None:
    client, rec = make_client([httpx.Response(429, text="rate limited")] * 5)
    with pytest.raises(OllamaRateLimitError) as exc_info:
        client.chat("gemma3:27b", [{"role": "user", "content": "hi"}], max_retries=3)
    assert exc_info.value.status_code == 429
    assert exc_info.value.body == "rate limited"
    assert rec.calls == 3
    assert len(rec.sleeps) == 2


def test_chat_429_honors_retry_after_header() -> None:
    client, rec = make_client([httpx.Response(429, text="slow down", headers={"Retry-After": "7"})] * 2)
    with pytest.raises(OllamaRateLimitError):
        client.chat("gemma3:27b", [{"role": "user", "content": "hi"}], max_retries=2)
    assert len(rec.sleeps) == 1
    assert rec.sleeps[0] >= 7.0


def test_chat_404_is_not_retried() -> None:
    client, rec = make_client([httpx.Response(404, json={"error": "model not found"})])
    with pytest.raises(OllamaError) as exc_info:
        client.chat("missing-model", [{"role": "user", "content": "hi"}])
    assert exc_info.value.status_code == 404
    body = exc_info.value.body
    assert body is not None and "model not found" in body
    assert not isinstance(exc_info.value, OllamaRateLimitError)
    assert rec.calls == 1
    assert rec.sleeps == []


def test_chat_format_schema_passed_verbatim() -> None:
    client, rec = make_client()
    schema = {
        "type": "object",
        "properties": {"translation": {"type": "string"}},
        "required": ["translation"],
    }
    client.chat("gemma3:27b", [{"role": "user", "content": "hi"}], format=schema)
    assert rec.bodies[0]["format"] == schema


def test_chat_keep_alive_zero_present_and_omission_absent() -> None:
    client, rec = make_client()
    client.chat("gemma3:27b", [{"role": "user", "content": "hi"}], keep_alive=0)
    assert rec.bodies[0]["keep_alive"] == 0

    client, rec = make_client()
    client.chat("gemma3:27b", [{"role": "user", "content": "hi"}])
    assert "keep_alive" not in rec.bodies[0]


def test_ps_parses_running_models() -> None:
    client, rec = make_client([httpx.Response(200, json=PS_BODY)])
    models = client.ps()
    assert models == [
        RunningModel(name="gemma3:27b", size_vram=1_700_000_000, expires_at="2026-09-18T12:05:00.000000Z"),
        RunningModel(name="qwen3:8b", size_vram=0, expires_at=None),
    ]
    assert rec.requests[0].url == "http://localhost:11434/api/ps"


def test_close_closes_owned_client_only() -> None:
    owned = OllamaClient(OllamaConfig(), Secrets.model_construct(ollama_api_key=SecretStr(API_KEY)))
    owned.close()
    assert owned._http.is_closed
    owned.close()  # safe to call twice

    injected, _ = make_client()
    injected.close()
    assert not injected._http.is_closed


def test_context_manager_closes_on_exit() -> None:
    with OllamaClient(OllamaConfig(), Secrets.model_construct(ollama_api_key=SecretStr(API_KEY))) as client:
        assert isinstance(client, OllamaClient)
    assert client._http.is_closed


def test_chat_retries_transport_errors() -> None:
    """Connection-level failures retry with backoff like HTTP 5xx do."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("connection refused")

    sleeps: list[float] = []

    def sleep(delay: float) -> None:
        sleeps.append(delay)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = OllamaClient(
        OllamaConfig(), Secrets.model_construct(ollama_api_key=SecretStr(API_KEY)), http=http, sleep=sleep
    )
    with pytest.raises(OllamaError) as exc_info:
        client.chat("gemma3:27b", [{"role": "user", "content": "hi"}], max_retries=3)
    assert exc_info.value.status_code is None
    assert len(calls) == 3
    assert len(sleeps) == 2


def backoff_bounds(attempt: int) -> tuple[float, float]:
    delay = min(1.0 * (2**attempt), 30.0)
    return delay, delay * 1.25


def test_backoff_delay_curve_matches_spec() -> None:
    for attempt in range(6):
        low, high = backoff_bounds(attempt)
        for _ in range(50):
            delay = OllamaClient._backoff_delay(attempt)
            assert low <= delay <= high
    capped_low, capped_high = backoff_bounds(10)
    for _ in range(50):
        assert capped_low <= OllamaClient._backoff_delay(10, None) <= capped_high
    for _ in range(50):
        delay = OllamaClient._backoff_delay(0, 7.0)
        assert 7.0 <= delay <= 7.0 * 1.25

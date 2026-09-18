"""Tests for queue notifications: webhook payloads, error swallowing, combining (card B23)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import httpx
import pytest

from omniscan.queue.notify import combine, log_notifier, webhook_notifier
from omniscan.queue.store import Job


def make_job(**overrides: object) -> Job:
    """A finished job as the store would return it, with fields overridden."""
    fields: dict[str, object] = {
        "id": 7,
        "series": "시리즈",
        "chapters": ("Chapter 1", "Chapter 2"),
        "stages": ("ingest", "slice"),
        "force": False,
        "priority": 0,
        "status": "done",
        "attempts": 1,
        "max_attempts": 2,
        "error": None,
        "created_at": datetime.now(UTC),
        "started_at": None,
        "finished_at": None,
    }
    fields.update(overrides)
    return Job(**fields)  # type: ignore[arg-type]


def mock_client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler)


def test_webhook_posts_the_job_payload() -> None:
    bodies: list[dict[str, object]] = []
    job = make_job()

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200)

    notifier = webhook_notifier("http://example/hook", client=mock_client(httpx.MockTransport(handler)))
    notifier("job_done", job)
    notifier("queue_empty", None)

    assert bodies == [
        {
            "event": "job_done",
            "job": {
                "id": 7,
                "series": "시리즈",
                "chapters": ["Chapter 1", "Chapter 2"],
                "stages": ["ingest", "slice"],
                "status": "done",
                "attempts": 1,
                "max_attempts": 2,
                "error": None,
            },
        },
        {"event": "queue_empty", "job": None},
    ]


def test_webhook_payload_chapters_null() -> None:
    bodies: list[dict[str, object]] = []
    job = make_job(
        id=1,
        series="S",
        chapters=None,
        stages=("slice",),
        status="failed",
        attempts=2,
        error="RuntimeError: boom",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200)

    webhook_notifier("http://example/hook", client=mock_client(httpx.MockTransport(handler)))(
        "job_failed", job
    )
    assert bodies == [
        {
            "event": "job_failed",
            "job": {
                "id": 1,
                "series": "S",
                "chapters": None,
                "stages": ["slice"],
                "status": "failed",
                "attempts": 2,
                "max_attempts": 2,
                "error": "RuntimeError: boom",
            },
        }
    ]


def test_webhook_swallows_500() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    notifier = webhook_notifier("http://example/hook", client=mock_client(httpx.MockTransport(handler)))
    notifier("job_done", make_job())  # must not raise


def test_webhook_swallows_connect_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    notifier = webhook_notifier("http://example/hook", client=mock_client(httpx.MockTransport(handler)))
    notifier("job_done", make_job())  # must not raise


def test_webhook_uses_the_timeout_when_creating_its_own_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without `client=`, a short-lived httpx.Client is created with the given timeout."""
    created: dict[str, object] = {}
    real_init = httpx.Client.__init__

    def spy_init(self: httpx.Client, **kwargs: object) -> None:
        created.update(kwargs)
        real_init(self, transport=httpx.MockTransport(lambda request: httpx.Response(200)))

    monkeypatch.setattr(httpx.Client, "__init__", spy_init)
    notifier = webhook_notifier("http://example/hook", timeout=1.5)
    notifier("queue_empty", None)
    assert created == {"timeout": 1.5}


def test_combine_calls_all_notifiers_in_order_despite_a_raising_one() -> None:
    calls: list[str] = []
    job = make_job()

    def raising(event: str, job: Job | None) -> None:
        calls.append("raising")
        raise RuntimeError("boom")

    def second(event: str, job: Job | None) -> None:
        calls.append(f"second:{event}")

    combine(raising, second)("job_done", job)
    assert calls == ["raising", "second:job_done"]


def test_log_notifier_emits_info(caplog: pytest.LogCaptureFixture) -> None:
    job = make_job(id=3, status="failed", error="RuntimeError: boom")
    with caplog.at_level(logging.INFO, logger="omniscan.queue"):
        log_notifier("job_failed", job)
        log_notifier("queue_empty", None)

    messages = [record.message for record in caplog.records]
    assert len(messages) == 2
    assert all(record.levelno == logging.INFO for record in caplog.records)
    assert "job_failed" in messages[0]
    assert "3" in messages[0] and "시리즈" in messages[0] and "failed" in messages[0]
    assert "RuntimeError: boom" in messages[0]
    assert "queue_empty" in messages[1]

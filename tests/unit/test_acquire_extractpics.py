"""Tests for omniscan.acquire.extractpics (no network; scripted httpx.MockTransport, fake clock/sleep)."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from omniscan.acquire.download import USER_AGENT
from omniscan.acquire.drm import DrmPlatformError
from omniscan.acquire.extractor import (
    AuthError,
    ExtractedImage,
    ExtractError,
    Extraction,
    Extractor,
    QuotaError,
)
from omniscan.acquire.extractpics import ExtractPicsClient

PAGE_URL = "https://x.test/ch1"


def response(status: int, body: Any = None, headers: dict[str, str] | None = None) -> httpx.Response:
    """A scripted mock response; a str body goes out as text, anything else as JSON."""
    if isinstance(body, str):
        return httpx.Response(status, text=body, headers=headers)
    return httpx.Response(status, json=body if body is not None else {}, headers=headers)


def envelope(
    extraction_id: str,
    status: str,
    *,
    page_url: str | None = PAGE_URL,
    images: list[dict[str, str]] | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """The extract.pics response envelope as measured live."""
    data: dict[str, Any] = {
        "id": extraction_id,
        "status": status,
        "created_at": "2026-09-20",
        "project_id": "p",
    }
    if page_url is not None:
        data["url"] = page_url
    if images is not None:
        data["images"] = images
    if message is not None:
        data["message"] = message
    return {"data": data}


class Api:
    """Scripted MockTransport backend keyed by 'METHOD path'; requests recorded in call order."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._scripted: dict[str, list[Any]] = {}

    def script(self, key: str, *outcomes: Any) -> None:
        """Queue outcomes for a key: httpx.Response or an exception; the last one repeats forever."""
        self._scripted[key] = list(outcomes)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self._scripted[f"{request.method} {request.url.path}"]
        outcome = outcome.pop(0) if len(outcome) > 1 else outcome[0]
        if isinstance(outcome, httpx.Response):
            return outcome
        raise outcome


class FakeTime:
    """Sleep recorder whose clock advances by exactly the slept amount."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now


def make_client(api: Api, fake_time: FakeTime, **kwargs: Any) -> ExtractPicsClient:
    """A client over the scripted API using the fake clock/sleep."""
    http = httpx.Client(transport=httpx.MockTransport(api.handler), timeout=30.0)
    return ExtractPicsClient("test-key", client=http, sleep=fake_time.sleep, clock=fake_time.clock, **kwargs)


def test_happy_path() -> None:
    images = [{"id": f"i{i}", "url": f"https://cdn.test/{i}.jpg"} for i in (1, 2, 3)]
    api = Api()
    api.script("POST /v0/extractions", response(201, envelope("abc", "pending")))
    api.script(
        "GET /v0/extractions/abc",
        response(200, envelope("abc", "running")),
        response(200, envelope("abc", "done", images=images)),
    )
    fake_time = FakeTime()
    client = make_client(api, fake_time)

    extraction = client.extract(PAGE_URL)

    assert extraction == Extraction(
        id="abc",
        page_url=PAGE_URL,
        images=tuple(ExtractedImage(url=f"https://cdn.test/{i}.jpg") for i in (1, 2, 3)),
        credits=1,
    )
    post = api.requests[0]
    assert (post.method, post.url.path) == ("POST", "/v0/extractions")
    assert post.headers["Authorization"] == "test-key"
    assert post.headers["Accept"] == "application/json"
    assert post.headers["User-Agent"] == USER_AGENT
    assert json.loads(post.content) == {"url": PAGE_URL, "mode": "basic"}
    assert [request.url.path for request in api.requests[1:]] == [
        "/v0/extractions/abc",
        "/v0/extractions/abc",
    ]
    assert fake_time.sleeps == [2.5, 2.5]
    client.close()


def test_advanced_mode_sends_the_mode_and_costs_two_credits() -> None:
    images = [{"id": "i1", "url": "https://cdn.test/1.jpg"}]
    api = Api()
    api.script("POST /v0/extractions", response(201, envelope("abc", "done", images=images)))
    client = make_client(api, FakeTime())

    extraction = client.extract(PAGE_URL, mode="advanced")

    assert extraction.credits == 2
    assert json.loads(api.requests[0].content) == {"url": PAGE_URL, "mode": "advanced"}


def test_timeouts_301_for_advanced_181_for_basic() -> None:
    api = Api()
    api.script("POST /v0/extractions", response(201, envelope("abc", "running")))
    api.script("GET /v0/extractions/abc", response(200, envelope("abc", "running")))

    advanced = make_client(api, FakeTime(), poll_interval_s=1.0)
    with pytest.raises(ExtractError, match=r"timed out after 300s") as advanced_error:
        advanced.extract(PAGE_URL, mode="advanced")
    assert "abc" in str(advanced_error.value)

    basic = make_client(api, FakeTime(), poll_interval_s=1.0)
    with pytest.raises(ExtractError, match=r"timed out after 180s") as basic_error:
        basic.extract(PAGE_URL)
    assert "abc" in str(basic_error.value)


def test_polling_stops_at_the_timeout_moment() -> None:
    api = Api()
    api.script("POST /v0/extractions", response(201, envelope("abc", "running")))
    api.script("GET /v0/extractions/abc", response(200, envelope("abc", "running")))
    fake_time = FakeTime()
    client = make_client(api, fake_time, poll_interval_s=1.0)

    with pytest.raises(ExtractError):
        client.extract(PAGE_URL)

    assert fake_time.now == 181.0  # 180 polls of 1 s, then the basic default timeout of 180 s is exceeded


def test_drops_duplicate_and_urlless_images_and_falls_back_for_page_url() -> None:
    images: list[dict[str, str]] = [
        {"id": "i1", "url": "https://cdn.test/a.jpg"},
        {"id": "i2", "url": "https://cdn.test/a.jpg"},  # exact duplicate
        {"id": "i3"},  # no url at all
        {"id": "i4", "url": "https://cdn.test/b.jpg"},
    ]
    api = Api()
    api.script("POST /v0/extractions", response(200, envelope("abc", "done", page_url=None, images=images)))
    client = make_client(api, FakeTime())

    extraction = client.extract(PAGE_URL)

    assert extraction.images == (
        ExtractedImage(url="https://cdn.test/a.jpg"),
        ExtractedImage(url="https://cdn.test/b.jpg"),
    )
    assert extraction.page_url == PAGE_URL


def test_failed_status_reports_status_and_message() -> None:
    api = Api()
    api.script("POST /v0/extractions", response(201, envelope("abc", "pending")))
    api.script("GET /v0/extractions/abc", response(200, envelope("abc", "failed", message="render blew up")))
    client = make_client(api, FakeTime())

    with pytest.raises(ExtractError, match="ended with status 'failed': render blew up"):
        client.extract(PAGE_URL)


def test_unknown_status_reports_the_status() -> None:
    api = Api()
    api.script("POST /v0/extractions", response(201, envelope("abc", "error")))
    client = make_client(api, FakeTime())

    with pytest.raises(ExtractError, match="ended with status 'error'"):
        client.extract(PAGE_URL)


def test_401_and_402_fail_once_without_retry() -> None:
    api = Api()
    api.script("POST /v0/extractions", response(401, {"message": "Unauthenticated."}))
    with pytest.raises(AuthError, match="rejected the API key"):
        make_client(api, FakeTime()).extract(PAGE_URL)
    assert len(api.requests) == 1

    quota = Api()
    quota.script("POST /v0/extractions", response(402))
    with pytest.raises(QuotaError, match="no credits"):
        make_client(quota, FakeTime()).extract(PAGE_URL)
    assert len(quota.requests) == 1


def test_422_message_contains_both_texts() -> None:
    api = Api()
    api.script(
        "POST /v0/extractions",
        response(
            422,
            {"message": "The url format is invalid.", "errors": {"url": ["The url format is invalid."]}},
        ),
    )

    with pytest.raises(ExtractError) as excinfo:
        make_client(api, FakeTime()).extract(PAGE_URL)

    assert str(excinfo.value) == "The url format is invalid.: The url format is invalid."


def test_404_while_polling() -> None:
    api = Api()
    api.script("POST /v0/extractions", response(201, envelope("abc", "pending")))
    api.script(
        "GET /v0/extractions/abc", response(404, {"message": "No query results for model [Extraction]."})
    )

    with pytest.raises(ExtractError, match=r"extraction abc not found"):
        make_client(api, FakeTime()).extract(PAGE_URL)


def test_other_4xx_reports_status_and_body() -> None:
    api = Api()
    api.script("POST /v0/extractions", response(418, "I'm a teapot"))

    with pytest.raises(ExtractError) as excinfo:
        make_client(api, FakeTime()).extract(PAGE_URL)

    assert "HTTP 418" in str(excinfo.value)
    assert "teapot" in str(excinfo.value)


def test_malformed_2xx_bodies() -> None:
    api = Api()
    api.script("POST /v0/extractions", response(200, "<html>boom</html>"))
    with pytest.raises(ExtractError, match=r"unexpected response from extract\.pics: <html>boom</html>"):
        make_client(api, FakeTime()).extract(PAGE_URL)

    empty = Api()
    empty.script("POST /v0/extractions", response(200, {}))
    with pytest.raises(ExtractError, match=r"unexpected response from extract\.pics"):
        make_client(empty, FakeTime()).extract(PAGE_URL)


def test_5xx_retries_then_succeeds() -> None:
    images = [{"id": "i1", "url": "https://cdn.test/1.jpg"}]
    api = Api()
    api.script(
        "POST /v0/extractions",
        response(503),
        response(503),
        response(201, envelope("abc", "done", images=images)),
    )
    fake_time = FakeTime()
    client = make_client(api, fake_time, poll_interval_s=0.0)

    assert client.extract(PAGE_URL).images == (ExtractedImage(url="https://cdn.test/1.jpg"),)
    assert fake_time.sleeps == [0.5, 1.0]


def test_5xx_exhaustion_raises_extract_error() -> None:
    api = Api()
    api.script("POST /v0/extractions", response(503))
    fake_time = FakeTime()

    with pytest.raises(ExtractError, match="503"):
        make_client(api, fake_time).extract(PAGE_URL)

    assert len(api.requests) == 4
    assert fake_time.sleeps == [0.5, 1.0, 2.0]


def test_connect_error_retries_then_succeeds_and_exhausts() -> None:
    images = [{"id": "i1", "url": "https://cdn.test/1.jpg"}]
    api = Api()
    api.script(
        "POST /v0/extractions",
        httpx.ConnectError("refused"),
        httpx.ConnectError("refused"),
        response(201, envelope("abc", "done", images=images)),
    )
    fake_time = FakeTime()
    assert make_client(api, fake_time).extract(PAGE_URL).id == "abc"
    assert fake_time.sleeps == [0.5, 1.0]

    down = Api()
    down.script("POST /v0/extractions", httpx.ConnectError("refused"))
    with pytest.raises(ExtractError, match="refused"):
        make_client(down, FakeTime()).extract(PAGE_URL)
    assert len(down.requests) == 4


def test_429_honours_and_caps_retry_after() -> None:
    api = Api()
    api.script(
        "POST /v0/extractions",
        response(429, headers={"Retry-After": "7"}),
        response(201, envelope("abc", "done")),
    )
    fake_time = FakeTime()
    make_client(api, fake_time).extract(PAGE_URL)
    assert fake_time.sleeps == [7.0]

    huge = Api()
    huge.script(
        "POST /v0/extractions",
        response(429, headers={"Retry-After": "900"}),
        response(201, envelope("abc", "done")),
    )
    huge_time = FakeTime()
    make_client(huge, huge_time).extract(PAGE_URL)
    assert huge_time.sleeps == [60.0]


def test_429_without_header_uses_the_backoff_fallback() -> None:
    api = Api()
    api.script(
        "POST /v0/extractions",
        response(429),
        response(429),
        response(201, envelope("abc", "done")),
    )
    fake_time = FakeTime()

    make_client(api, fake_time).extract(PAGE_URL)

    assert fake_time.sleeps == [5.0, 10.0]


def test_429_forever_raises_quota_error() -> None:
    api = Api()
    api.script("POST /v0/extractions", response(429))
    fake_time = FakeTime()

    with pytest.raises(QuotaError, match="rate limit still exceeded after 3 retries"):
        make_client(api, fake_time).extract(PAGE_URL)

    assert len(api.requests) == 4
    assert fake_time.sleeps == [5.0, 10.0, 15.0]


def test_drm_platform_never_touches_the_transport() -> None:
    api = Api()

    with pytest.raises(DrmPlatformError, match="Naver Webtoon"):
        make_client(api, FakeTime()).extract("https://comic.naver.com/webtoon/detail?titleId=1&no=1")

    assert api.requests == []


def test_rate_limiter_keeps_three_requests_per_window() -> None:
    api = Api()
    api.script("POST /v0/extractions", response(201, envelope("abc", "pending")))
    api.script(
        "GET /v0/extractions/abc",
        response(200, envelope("abc", "done", images=[{"id": "i1", "url": "https://cdn.test/1.jpg"}])),
    )
    fake_time = FakeTime()
    times: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        times.append(fake_time.clock())
        return api.handler(request)

    http = httpx.Client(transport=httpx.MockTransport(handler), timeout=30.0)
    client = ExtractPicsClient(
        "test-key",
        client=http,
        requests_per_minute=3,
        poll_interval_s=0.0,
        sleep=fake_time.sleep,
        clock=fake_time.clock,
    )

    for _ in range(4):
        client.extract(PAGE_URL)

    assert times == [0.0, 0.0, 0.0, 60.0, 60.0, 60.0, 120.0, 120.0]
    for start in times:
        assert sum(1 for moment in times if start <= moment < start + 60) <= 3


def test_zero_requests_per_minute_is_rejected() -> None:
    with pytest.raises(ValueError, match="requests_per_minute"):
        ExtractPicsClient("test-key", requests_per_minute=0)


def test_api_key_never_leaks(caplog: pytest.LogCaptureFixture) -> None:
    failures = [
        response(401, {"message": "Unauthenticated."}),
        response(402),
        response(422, {"message": "invalid", "errors": {"url": ["invalid"]}}),
        response(503),
    ]
    with caplog.at_level(logging.DEBUG):
        for outcome in failures:
            api = Api()
            api.script("POST /v0/extractions", outcome)
            with pytest.raises(ExtractError) as excinfo:
                make_client(api, FakeTime()).extract(PAGE_URL)
            assert "test-key" not in str(excinfo.value)

        transport = Api()
        transport.script("POST /v0/extractions", httpx.ConnectError("refused"))
        with pytest.raises(ExtractError) as transport_error:
            make_client(transport, FakeTime()).extract(PAGE_URL)
        assert "test-key" not in str(transport_error.value)

        assert "test-key" not in repr(make_client(Api(), FakeTime()))
        assert not any("test-key" in record.getMessage() for record in caplog.records)


def test_extractor_protocol_and_context_manager() -> None:
    api = Api()
    api.script(
        "POST /v0/extractions",
        response(201, envelope("abc", "done", images=[{"id": "i1", "url": "https://cdn.test/1.jpg"}])),
    )
    fake_time = FakeTime()
    http = httpx.Client(transport=httpx.MockTransport(api.handler), timeout=30.0)
    client = ExtractPicsClient("test-key", client=http, sleep=fake_time.sleep, clock=fake_time.clock)
    protocol_variable: Extractor = client  # pyright-checked structural compatibility

    with client:
        assert client.extract(PAGE_URL).id == "abc"
    assert not http.is_closed  # a client the wrapper does not own stays open
    assert protocol_variable is client

    owned = ExtractPicsClient("test-key")
    with owned:
        pass
    assert owned._client.is_closed  # a client the wrapper created itself is closed

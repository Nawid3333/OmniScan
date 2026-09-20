"""extract.pics REST client: one chapter-page URL in, every image URL of that page out (owned contract: `Extractor`)."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Any, Self

import httpx

from omniscan.acquire.download import USER_AGENT
from omniscan.acquire.drm import check_allowed
from omniscan.acquire.extractor import (
    AuthError,
    ExtractedImage,
    ExtractError,
    Extraction,
    Mode,
    QuotaError,
    credit_cost,
)

API_BASE = "https://api.extract.pics/v0"
DEFAULT_TIMEOUT_S: dict[Mode, float] = {"basic": 180.0, "advanced": 300.0}


class ExtractPicsClient:
    """extract.pics REST client implementing `Extractor`."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        base_url: str = API_BASE,
        poll_interval_s: float = 2.5,
        timeout_s: float | None = None,
        max_retries: int = 3,
        requests_per_minute: int = 20,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure the client; `requests_per_minute` must be at least 1."""
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least 1")
        self._api_key = api_key
        self._client = httpx.Client(timeout=30.0) if client is None else client
        self._owns_client = client is None
        self._base_url = base_url.rstrip("/")
        self._poll_interval_s = poll_interval_s
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._requests_per_minute = requests_per_minute
        self._sleep = sleep
        self._clock = clock
        self._sent_at: deque[float] = deque()

    def __repr__(self) -> str:
        """The base URL only; the API key must never appear anywhere the client is printed."""
        return f"ExtractPicsClient(base_url={self._base_url!r})"

    def extract(self, url: str, *, mode: Mode = "basic") -> Extraction:
        """Run one extraction to completion; raises ExtractError (AuthError, QuotaError) on failure."""
        check_allowed(url)
        timeout_s = DEFAULT_TIMEOUT_S[mode] if self._timeout_s is None else self._timeout_s
        body, data = self._data(self._send("POST", "/extractions", json={"url": url, "mode": mode}))
        extraction_id = data["id"]
        status = data["status"]
        start = self._clock()
        while status != "done":
            if status not in ("pending", "running"):
                raise ExtractError(_ended_message(body, data, extraction_id))
            self._sleep(self._poll_interval_s)
            if self._clock() - start > timeout_s:
                raise ExtractError(f"extraction {extraction_id} timed out after {timeout_s:.0f}s")
            body, data = self._data(
                self._send("GET", f"/extractions/{extraction_id}", extraction_id=extraction_id)
            )
            status = data["status"]
        return Extraction(
            id=extraction_id,
            page_url=data.get("url") or url,
            images=_images(data),
            credits=credit_cost(1, mode),
        )

    def close(self) -> None:
        """Close the HTTP client, but only when this client created it itself."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        """The headers every extract.pics request carries; never logged."""
        return {"Authorization": self._api_key, "Accept": "application/json", "User-Agent": USER_AGENT}

    def _send(
        self, method: str, path: str, *, extraction_id: str | None = None, **kwargs: Any
    ) -> httpx.Response:
        """One throttled HTTP exchange with the retry policy; raises ExtractError (AuthError, QuotaError)."""
        attempt = 0
        while True:
            self._throttle()
            try:
                response = self._client.request(
                    method, self._base_url + path, headers=self._headers(), **kwargs
                )
            except httpx.TransportError as exc:
                if attempt >= self._max_retries:
                    raise ExtractError(f"extract.pics request failed: {exc}") from exc
                self._sleep(0.5 * 2**attempt)
                attempt += 1
                continue
            status = response.status_code
            if status == 401:
                raise AuthError("extract.pics rejected the API key")
            if status == 402:
                raise QuotaError("extract.pics reports no credits left")
            if status == 429:
                if attempt >= self._max_retries:
                    raise QuotaError(
                        f"extract.pics rate limit still exceeded after {self._max_retries} retries"
                    )
                self._sleep(_retry_after_wait(response, attempt))
                attempt += 1
                continue
            if status >= 500:
                if attempt >= self._max_retries:
                    raise ExtractError(
                        f"extract.pics answered HTTP {status} after {self._max_retries} retries"
                    )
                self._sleep(0.5 * 2**attempt)
                attempt += 1
                continue
            if status == 422:
                raise ExtractError(_unprocessible_message(response))
            if status == 404 and extraction_id is not None:
                raise ExtractError(f"extraction {extraction_id} not found")
            if status >= 400:
                raise ExtractError(f"extract.pics answered HTTP {status}: {response.text[:200]}")
            return response

    def _data(self, response: httpx.Response) -> tuple[dict[str, Any], dict[str, Any]]:
        """(body, data) of a 2xx extraction envelope; raises ExtractError for a malformed body."""
        try:
            body: Any = response.json()
        except ValueError:
            body = None
        data = body.get("data") if isinstance(body, dict) else None
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("id"), str)
            or not isinstance(data.get("status"), str)
        ):
            raise ExtractError(f"unexpected response from extract.pics: {response.text[:200]}")
        return (body if isinstance(body, dict) else {}), data

    def _throttle(self) -> None:
        """Keep at most `requests_per_minute` requests inside any 60 s window by sleeping first."""
        while True:
            now = self._clock()
            while self._sent_at and now - self._sent_at[0] >= 60:
                self._sent_at.popleft()
            if len(self._sent_at) < self._requests_per_minute:
                self._sent_at.append(now)
                return
            self._sleep(60 - (now - self._sent_at[0]))


def _images(data: dict[str, Any]) -> tuple[ExtractedImage, ...]:
    """ExtractedImage tuple of an extraction's image list, without duplicates and without URL-less entries."""
    raw = data.get("images")
    if not isinstance(raw, list):
        return ()
    images: list[ExtractedImage] = []
    seen: set[str] = set()
    for entry in raw:
        url = entry.get("url") if isinstance(entry, dict) else None
        if not isinstance(url, str) or url in seen:
            continue
        seen.add(url)
        images.append(ExtractedImage(url=url))
    return tuple(images)


def _ended_message(body: dict[str, Any], data: dict[str, Any], extraction_id: str) -> str:
    """Failure message for an extraction that left pending/running with an unusable status."""
    message = f"extraction {extraction_id} ended with status {data['status']!r}"
    for source in (body, data):
        extra = source.get("message")
        if isinstance(extra, str) and extra:
            return f"{message}: {extra}"
    return message


def _retry_after_wait(response: httpx.Response, attempt: int) -> float:
    """Seconds to wait after a 429: the capped Retry-After header, or the back-off fallback."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return 5.0 * (attempt + 1)
    try:
        return float(min(int(raw), 60))
    except ValueError:
        return 5.0 * (attempt + 1)


def _unprocessible_message(response: httpx.Response) -> str:
    """'message: first error text' of a 422 body, or the raw body text when it is not JSON."""
    try:
        body: Any = response.json()
    except ValueError:
        return response.text[:200]
    if not isinstance(body, dict):
        return response.text[:200]
    message = body.get("message")
    errors = body.get("errors")
    first: str | None = None
    if isinstance(errors, dict):
        for values in errors.values():
            if isinstance(values, list) and values and isinstance(values[0], str):
                first = values[0]
                break
            if isinstance(values, str):
                first = values
                break
    if message is None and first is None:
        return response.text[:200]
    return ": ".join(str(part) for part in (message, first) if part)

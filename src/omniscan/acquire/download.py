"""Ordered, resumable download of one chapter's image list with non-page filtering and a manifest."""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from omniscan.acquire.drm import check_allowed
from omniscan.acquire.filters import image_looks_like_page, url_looks_like_page

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
SUPPORTED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MANIFEST_NAME = "acquire.json"


class AcquireError(RuntimeError):
    """A chapter download finished with failures; kept files and manifest let a re-run continue."""


@dataclass(frozen=True, slots=True)
class ImageRef:
    """One image to download: its URL and the Referer to send when fetching it."""

    url: str
    referer: str | None = None


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Outcome of one download_chapter call: files present after it, plus per-outcome counts."""

    files: list[str]
    downloaded: int
    skipped: int
    rejected: int


@dataclass(frozen=True, slots=True)
class _Saved:
    """A validated image: its manifest entry plus the bytes the calling thread writes."""

    entry: dict[str, Any]
    data: bytes


@dataclass(frozen=True, slots=True)
class _Rejected:
    """An image the content filter dropped."""

    reason: str


@dataclass(frozen=True, slots=True)
class _Failed:
    """An image that could not be downloaded or decoded."""

    message: str


type _Outcome = _Saved | _Rejected | _Failed


class _RefError(Exception):
    """Internal per-image failure message; never escapes download_chapter."""


def download_chapter(
    refs: Sequence[ImageRef],
    dest: Path,
    *,
    client: httpx.Client | None = None,
    concurrency: int = 4,
    retries: int = 3,
    backoff_s: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    apply_filters: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> DownloadResult:
    """Download refs in order into dest, resumably, filtered, with a manifest; raises on failures."""
    for ref in refs:
        check_allowed(ref.url)
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / MANIFEST_NAME
    old_images, old_rejected = _load_manifest(manifest_path)
    total = len(refs)
    digits = max(3, len(str(total)))
    images: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    failures: list[tuple[int, str]] = []
    done = 0
    skipped = 0
    rejected_count = 0
    downloaded = 0
    progress: Callable[[int, int], None] = on_progress if on_progress is not None else _noop
    if client is None:
        client = httpx.Client(follow_redirects=True, timeout=30.0)
        owns_client = True
    else:
        owns_client = False
    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            pending: dict[Future[_Outcome], tuple[int, ImageRef]] = {}
            for position, ref in enumerate(refs, start=1):
                key = (ref.url, position)
                carried = old_images.get(key)
                if carried is not None and _file_matches(dest, carried):
                    images.append(dict(carried))
                    skipped += 1
                elif key in old_rejected:
                    rejected.append(dict(old_rejected[key]))
                    rejected_count += 1
                else:
                    reason = _url_rejection(apply_filters, ref.url)
                    if reason is not None:
                        rejected.append({"position": position, "url": ref.url, "reason": reason})
                        rejected_count += 1
                    else:
                        future = pool.submit(
                            _process_one,
                            ref,
                            position,
                            digits,
                            apply_filters,
                            client,
                            retries,
                            backoff_s,
                            sleep,
                        )
                        pending[future] = (position, ref)
                        continue
                done += 1
                progress(done, total)
            for future in as_completed(pending):
                position, ref = pending[future]
                outcome = future.result()
                if isinstance(outcome, _Saved):
                    images.append(outcome.entry)
                    _write_image(dest, str(outcome.entry["file"]), outcome.data)
                    downloaded += 1
                    done += 1
                elif isinstance(outcome, _Rejected):
                    rejected.append({"position": position, "url": ref.url, "reason": outcome.reason})
                    rejected_count += 1
                    done += 1
                else:
                    failures.append((position, outcome.message))
                    done += 1
                progress(done, total)
    finally:
        if owns_client:
            client.close()
    _write_manifest(manifest_path, images, rejected)
    if failures:
        detail = "; ".join(f"#{position} {message}" for position, message in sorted(failures))
        raise AcquireError(f"{len(failures)} of {total} image(s) failed: {detail}")
    files = [str(entry["file"]) for entry in sorted(images, key=lambda entry: entry["position"])]
    return DownloadResult(files=files, downloaded=downloaded, skipped=skipped, rejected=rejected_count)


def _noop(done: int, total: int) -> None:
    """Discard progress updates when the caller passes no callback."""


def _url_rejection(apply_filters: bool, url: str) -> str | None:
    """Rejection reason from the URL heuristics, or None when the URL may be requested."""
    if not apply_filters:
        return None
    ok, reason = url_looks_like_page(url)
    return None if ok else reason


def _process_one(
    ref: ImageRef,
    position: int,
    digits: int,
    apply_filters: bool,
    client: httpx.Client,
    retries: int,
    backoff_s: float,
    sleep: Callable[[float], None],
) -> _Outcome:
    """Download, validate and filter one image; a failure never escapes as an exception."""
    try:
        data = _fetch(ref, client, retries, backoff_s, sleep)
        if len(data) > MAX_IMAGE_BYTES:
            raise _RefError("too large")
        fmt, width, height = _decode(data)
    except _RefError as exc:
        return _Failed(str(exc))
    if apply_filters:
        ok, reason = image_looks_like_page(width, height, len(data))
        if not ok:
            return _Rejected(reason)
    entry = {
        "position": position,
        "url": ref.url,
        "file": f"{position:0{digits}d}{SUPPORTED_FORMATS[fmt]}",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "width": width,
        "height": height,
    }
    return _Saved(entry, data)


def _fetch(
    ref: ImageRef,
    client: httpx.Client,
    retries: int,
    backoff_s: float,
    sleep: Callable[[float], None],
) -> bytes:
    """Response bytes of one GET, retried for transport errors, 429 and 5xx; raises _RefError otherwise."""
    headers = {"User-Agent": USER_AGENT}
    if ref.referer:
        headers["Referer"] = ref.referer
    attempt = 0
    while True:
        try:
            response = client.get(ref.url, headers=headers)
        except httpx.TransportError as exc:
            if attempt >= retries:
                raise _RefError(f"network error: {exc}") from exc
            sleep(backoff_s * 2**attempt)
            attempt += 1
            continue
        if response.status_code == 200:
            return response.content
        retryable = response.status_code == 429 or response.status_code >= 500
        if not retryable or attempt >= retries:
            raise _RefError(f"HTTP {response.status_code}")
        if response.status_code == 429:
            retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
            if retry_after is None:
                sleep(backoff_s * 2**attempt)
            else:
                sleep(min(retry_after, 30))
        else:
            sleep(backoff_s * 2**attempt)
        attempt += 1


def _retry_after_seconds(value: str | None) -> int | None:
    """Seconds from an integer Retry-After header; None when absent or not an integer."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _decode(data: bytes) -> tuple[str, int, int]:
    """(format, width, height) of image bytes; raises _RefError for undecodable or unsupported data."""
    fmt: str | None = None
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
            fmt = probe.format
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except Exception as exc:
        raise _RefError(f"not a supported image ({fmt or 'undecodable'})") from exc
    if fmt not in SUPPORTED_FORMATS:
        raise _RefError(f"not a supported image ({fmt})")
    return fmt, width, height


def _load_manifest(
    path: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    """(images, rejected) entries of an earlier run keyed by (url, position); empty when unreadable or invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return {}, {}
    if not isinstance(data, dict):
        return {}, {}
    return _image_entries(data.get("images")), _rejected_entries(data.get("rejected"))


def _image_entries(raw: object) -> dict[tuple[str, int], dict[str, Any]]:
    """Valid-shaped image entries of a manifest keyed by (url, position)."""
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    if not isinstance(raw, list):
        return entries
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("position"), int):
            continue
        if not all(isinstance(item.get(key), str) for key in ("url", "file", "sha256")):
            continue
        if not all(isinstance(item.get(key), int) for key in ("bytes", "width", "height")):
            continue
        entries[(item["url"], item["position"])] = item
    return entries


def _rejected_entries(raw: object) -> dict[tuple[str, int], dict[str, Any]]:
    """Valid-shaped rejected entries of a manifest keyed by (url, position)."""
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    if not isinstance(raw, list):
        return entries
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("position"), int):
            continue
        if not all(isinstance(item.get(key), str) for key in ("url", "reason")):
            continue
        entries[(item["url"], item["position"])] = item
    return entries


def _file_matches(dest: Path, entry: dict[str, Any]) -> bool:
    """True when the manifest's recorded file exists with exactly the recorded byte size."""
    try:
        return (dest / str(entry["file"])).stat().st_size == int(entry["bytes"])
    except OSError:
        return False


def _write_image(dest: Path, name: str, data: bytes) -> None:
    """Write image bytes atomically: first to '<name>.part', then moved onto the final name."""
    part = dest / (name + ".part")
    part.write_bytes(data)
    part.replace(dest / name)


def _write_manifest(path: Path, images: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    """Atomically write acquire.json with both entry lists sorted by position."""
    payload = {
        "images": sorted(images, key=lambda entry: entry["position"]),
        "rejected": sorted(rejected, key=lambda entry: entry["position"]),
    }
    part = path.parent / (path.name + ".part")
    part.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    part.replace(path)

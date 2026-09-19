"""Tests for omniscan.acquire.download (no network; httpx.MockTransport; Pillow-made fixtures)."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image

from omniscan.acquire import download as download_mod
from omniscan.acquire.download import (
    USER_AGENT,
    AcquireError,
    DownloadResult,
    ImageRef,
    download_chapter,
)
from omniscan.acquire.drm import DrmPlatformError

WIDTH, HEIGHT = 800, 1200


def noise_image(
    fmt: str, seed: int = 0, width: int = WIDTH, height: int = HEIGHT, **save_kwargs: Any
) -> bytes:
    """A >8 kB image of seeded numpy noise in the given Pillow format."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


class Server:
    """Scripted MockTransport backend: per-URL outcome lists, requests recorded in order."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._scripted: dict[str, list[Any]] = {}

    def script(self, url: str, *outcomes: Any) -> None:
        """Queue outcomes for a URL: bytes (200), int (status), (status, headers) or an exception."""
        self._scripted[url] = list(outcomes)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), follow_redirects=True, timeout=30.0)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcomes = self._scripted[str(request.url)]
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(outcome, bytes):
            return httpx.Response(200, content=outcome)
        if isinstance(outcome, tuple):
            status, headers = outcome
            return httpx.Response(status, headers=headers)
        if isinstance(outcome, int):
            return httpx.Response(outcome)
        raise outcome


def urls_for(count: int, suffix: str = "jpg") -> list[str]:
    """Count URLs of chapter 1 on a fake CDN."""
    return [f"https://cdn.example.org/ch1/{i:03d}.{suffix}" for i in range(1, count + 1)]


def expected_manifest_entry(position: int, url: str, fmt_ext: str, data: bytes) -> dict[str, Any]:
    """The exact manifest entry download_chapter must produce for one saved image."""
    return {
        "position": position,
        "url": url,
        "file": f"{position:03d}{fmt_ext}",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "width": WIDTH,
        "height": HEIGHT,
    }


def test_happy_path_writes_position_order_with_content_extensions(tmp_path: Path) -> None:
    urls = urls_for(5)
    payload = [
        noise_image("JPEG", seed=1),
        noise_image("PNG", seed=2),
        noise_image("WEBP", seed=3),
        noise_image("JPEG", seed=4),
        noise_image("PNG", seed=5),
    ]
    server = Server()
    for url, data in zip(urls, payload, strict=True):
        server.script(url, data)
    sleeps: list[float] = []
    dest = tmp_path / "Chapter 1"

    result = download_chapter(
        [ImageRef(url=url) for url in urls], dest, client=server.client(), sleep=sleeps.append
    )

    assert result == DownloadResult(
        files=["001.jpg", "002.png", "003.webp", "004.jpg", "005.png"], downloaded=5, skipped=0, rejected=0
    )
    assert [p.name for p in sorted(dest.iterdir())] == [
        "001.jpg",
        "002.png",
        "003.webp",
        "004.jpg",
        "005.png",
        "acquire.json",
    ]
    assert [(dest / name).read_bytes() for name in result.files] == payload
    manifest = json.loads((dest / "acquire.json").read_text(encoding="utf-8"))
    extensions = [".jpg", ".png", ".webp", ".jpg", ".png"]
    assert manifest == {
        "images": [
            expected_manifest_entry(i, url, ext, data)
            for i, (url, ext, data) in enumerate(zip(urls, extensions, payload, strict=True), start=1)
        ],
        "rejected": [],
    }
    assert sleeps == []


def test_manifest_text_is_independent_of_concurrency(tmp_path: Path) -> None:
    urls = urls_for(5)
    payload = [
        noise_image("JPEG", seed=1),
        noise_image("PNG", seed=2),
        noise_image("WEBP", seed=3),
        noise_image("JPEG", seed=4),
        noise_image("PNG", seed=5),
    ]
    refs = [ImageRef(url=url) for url in urls]
    texts = []
    for concurrency in (1, 4):
        server = Server()
        for url, data in zip(urls, payload, strict=True):
            server.script(url, data)
        dest = tmp_path / f"c{concurrency}"
        result = download_chapter(
            refs, dest, client=server.client(), concurrency=concurrency, sleep=lambda _s: None
        )
        texts.append((dest / "acquire.json").read_text(encoding="utf-8"))
        assert result.downloaded == 5
    assert texts[0] == texts[1]


def test_on_progress_reaches_total_and_is_monotonic(tmp_path: Path) -> None:
    urls = urls_for(3)
    server = Server()
    for i, url in enumerate(urls, start=1):
        server.script(url, noise_image("JPEG", seed=i))
    calls: list[tuple[int, int]] = []

    download_chapter(
        [ImageRef(url=url) for url in urls],
        tmp_path / "c",
        client=server.client(),
        concurrency=3,
        on_progress=lambda done, total: calls.append((done, total)),
        sleep=lambda _s: None,
    )

    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_referer_and_user_agent_headers(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=noise_image("JPEG", seed=1))

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=30.0)
    refs = [
        ImageRef(url="https://cdn.example.org/1.jpg", referer="https://example.org/a/1"),
        ImageRef(url="https://cdn.example.org/2.jpg"),
    ]

    download_chapter(refs, tmp_path / "c", client=client, sleep=lambda _s: None)

    assert [r.headers["User-Agent"] for r in seen] == [USER_AGENT, USER_AGENT]
    assert seen[0].headers["Referer"] == "https://example.org/a/1"
    assert seen[1].headers.get("Referer") is None


def test_retries_on_500_then_success(tmp_path: Path) -> None:
    urls = urls_for(1)
    data = noise_image("JPEG", seed=1)
    server = Server()
    server.script(urls[0], 500, 500, data)
    sleeps: list[float] = []

    result = download_chapter(
        [ImageRef(url=urls[0])], tmp_path / "c", client=server.client(), sleep=sleeps.append
    )

    assert result.downloaded == 1
    assert sleeps == [0.5, 1.0]
    assert len(server.requests) == 3


def test_429_uses_integer_retry_after(tmp_path: Path) -> None:
    urls = urls_for(1)
    server = Server()
    server.script(urls[0], (429, {"Retry-After": "2"}), noise_image("JPEG", seed=1))
    sleeps: list[float] = []

    result = download_chapter(
        [ImageRef(url=urls[0])], tmp_path / "c", client=server.client(), sleep=sleeps.append
    )

    assert result.downloaded == 1
    assert sleeps == [2.0]


def test_429_retry_after_is_capped_at_30(tmp_path: Path) -> None:
    urls = urls_for(1)
    server = Server()
    server.script(urls[0], (429, {"Retry-After": "999"}), noise_image("JPEG", seed=1))
    sleeps: list[float] = []

    download_chapter([ImageRef(url=urls[0])], tmp_path / "c", client=server.client(), sleep=sleeps.append)

    assert sleeps == [30.0]


def test_retries_on_connect_error_then_success(tmp_path: Path) -> None:
    urls = urls_for(1)
    server = Server()
    server.script(urls[0], httpx.ConnectError("refused"), noise_image("JPEG", seed=1))
    sleeps: list[float] = []

    result = download_chapter(
        [ImageRef(url=urls[0])], tmp_path / "c", client=server.client(), sleep=sleeps.append
    )

    assert result.downloaded == 1
    assert sleeps == [0.5]


def test_404_fails_once_without_retry(tmp_path: Path) -> None:
    urls = urls_for(1)
    server = Server()
    server.script(urls[0], 404)
    sleeps: list[float] = []

    with pytest.raises(AcquireError, match="#1 HTTP 404"):
        download_chapter([ImageRef(url=urls[0])], tmp_path / "c", client=server.client(), sleep=sleeps.append)

    assert len(server.requests) == 1
    assert sleeps == []


def test_retries_exhausted_then_acquire_error(tmp_path: Path) -> None:
    urls = urls_for(1)
    server = Server()
    server.script(urls[0], 500)
    sleeps: list[float] = []

    with pytest.raises(AcquireError, match="#1 HTTP 500"):
        download_chapter(
            [ImageRef(url=urls[0])], tmp_path / "c", client=server.client(), retries=2, sleep=sleeps.append
        )

    assert len(server.requests) == 3
    assert sleeps == [0.5, 1.0]


def test_one_failure_keeps_the_others_and_manifest(tmp_path: Path) -> None:
    urls = urls_for(3)
    payload = [noise_image("JPEG", seed=i) for i in (1, 3)]
    server = Server()
    server.script(urls[0], payload[0])
    server.script(urls[1], 404)
    server.script(urls[2], payload[1])
    dest = tmp_path / "c"

    with pytest.raises(AcquireError, match=re.escape("1 of 3 image(s) failed: #2 HTTP 404")):
        download_chapter(
            [ImageRef(url=url) for url in urls], dest, client=server.client(), sleep=lambda _s: None
        )

    assert sorted(p.name for p in dest.iterdir()) == ["001.jpg", "003.jpg", "acquire.json"]
    assert list(dest.glob("*.part")) == []
    manifest = json.loads((dest / "acquire.json").read_text(encoding="utf-8"))
    assert manifest == {
        "images": [
            expected_manifest_entry(1, urls[0], ".jpg", payload[0]),
            expected_manifest_entry(3, urls[2], ".jpg", payload[1]),
        ],
        "rejected": [],
    }


def test_second_identical_call_makes_no_requests(tmp_path: Path) -> None:
    urls = urls_for(3)
    payload = [noise_image("JPEG", seed=1), noise_image("JPEG", seed=2), noise_image("JPEG", seed=3)]
    server = Server()
    for url, data in zip(urls, payload, strict=True):
        server.script(url, data)
    dest = tmp_path / "c"
    assert (
        download_chapter(
            [ImageRef(url=url) for url in urls], dest, client=server.client(), sleep=lambda _s: None
        ).downloaded
        == 3
    )

    second = Server()
    for url, data in zip(urls, payload, strict=True):
        second.script(url, data)
    result = download_chapter(
        [ImageRef(url=url) for url in urls], dest, client=second.client(), sleep=lambda _s: None
    )

    assert second.requests == []
    assert result == DownloadResult(
        files=["001.jpg", "002.jpg", "003.jpg"], downloaded=0, skipped=3, rejected=0
    )


def test_resume_after_deleted_file_downloads_only_it(tmp_path: Path) -> None:
    urls = urls_for(3)
    payload = [noise_image("JPEG", seed=1), noise_image("JPEG", seed=2), noise_image("JPEG", seed=3)]
    server = Server()
    for url, data in zip(urls, payload, strict=True):
        server.script(url, data)
    dest = tmp_path / "c"
    download_chapter([ImageRef(url=url) for url in urls], dest, client=server.client(), sleep=lambda _s: None)
    (dest / "002.jpg").unlink()

    re_server = Server()
    for url, data in zip(urls, payload, strict=True):
        re_server.script(url, data)
    result = download_chapter(
        [ImageRef(url=url) for url in urls], dest, client=re_server.client(), sleep=lambda _s: None
    )

    assert [str(r.url) for r in re_server.requests] == [urls[1]]
    assert result == DownloadResult(
        files=["001.jpg", "002.jpg", "003.jpg"], downloaded=1, skipped=2, rejected=0
    )


def test_resume_after_truncated_file_downloads_it_again(tmp_path: Path) -> None:
    urls = urls_for(3)
    payload = [noise_image("JPEG", seed=1), noise_image("JPEG", seed=2), noise_image("JPEG", seed=3)]
    server = Server()
    for url, data in zip(urls, payload, strict=True):
        server.script(url, data)
    dest = tmp_path / "c"
    download_chapter([ImageRef(url=url) for url in urls], dest, client=server.client(), sleep=lambda _s: None)
    (dest / "001.jpg").write_bytes((dest / "001.jpg").read_bytes()[:100])

    re_server = Server()
    for url, data in zip(urls, payload, strict=True):
        re_server.script(url, data)
    result = download_chapter(
        [ImageRef(url=url) for url in urls], dest, client=re_server.client(), sleep=lambda _s: None
    )

    assert [str(r.url) for r in re_server.requests] == [urls[0]]
    assert result.downloaded == 1
    assert result.skipped == 2
    assert (dest / "001.jpg").read_bytes() == payload[0]


def test_resume_leaves_unmanifested_files_alone(tmp_path: Path) -> None:
    urls = urls_for(2)
    server = Server()
    for i, url in enumerate(urls, start=1):
        server.script(url, noise_image("JPEG", seed=i))
    dest = tmp_path / "c"
    dest.mkdir()
    stray = dest / "999.jpg"
    stray.write_bytes(b"not a chapter page")

    download_chapter([ImageRef(url=url) for url in urls], dest, client=server.client(), sleep=lambda _s: None)

    assert stray.read_bytes() == b"not a chapter page"


def test_resume_after_failed_run_downloads_only_missing(tmp_path: Path) -> None:
    urls = urls_for(3)
    payload = [noise_image("JPEG", seed=1), noise_image("JPEG", seed=2), noise_image("JPEG", seed=3)]
    first = Server()
    first.script(urls[0], payload[0])
    first.script(urls[1], 500)
    first.script(urls[2], payload[2])
    dest = tmp_path / "c"

    with pytest.raises(AcquireError):
        download_chapter(
            [ImageRef(url=url) for url in urls], dest, client=first.client(), sleep=lambda _s: None
        )

    second = Server()
    for url, data in zip(urls, payload, strict=True):
        second.script(url, data)
    result = download_chapter(
        [ImageRef(url=url) for url in urls], dest, client=second.client(), sleep=lambda _s: None
    )

    assert [str(r.url) for r in second.requests] == [urls[1]]
    assert result == DownloadResult(
        files=["001.jpg", "002.jpg", "003.jpg"], downloaded=1, skipped=2, rejected=0
    )


def test_url_and_content_filters(tmp_path: Path) -> None:
    urls = [*urls_for(3), "https://cdn.example.org/ch1/logo.png"]
    payload = [
        noise_image("JPEG", seed=1),
        noise_image("JPEG", seed=2, width=200, height=500),
        noise_image("JPEG", seed=3),
        noise_image("PNG", seed=4),
    ]
    server = Server()
    for url, data in zip(urls, payload, strict=True):
        server.script(url, data)
    dest = tmp_path / "c"
    sleeps: list[float] = []

    result = download_chapter(
        [ImageRef(url=url) for url in urls], dest, client=server.client(), sleep=sleeps.append
    )

    assert result == DownloadResult(files=["001.jpg", "003.jpg"], downloaded=2, skipped=0, rejected=2)
    assert [str(r.url) for r in server.requests] == urls[:3]
    manifest_text = (dest / "acquire.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["rejected"] == [
        {"position": 2, "url": urls[1], "reason": "too narrow"},
        {"position": 4, "url": urls[3], "reason": "url contains 'logo'"},
    ]

    second = Server()
    for url, data in zip(urls, payload, strict=True):
        second.script(url, data)
    again = download_chapter(
        [ImageRef(url=url) for url in urls], dest, client=second.client(), sleep=sleeps.append
    )

    assert second.requests == []
    assert again == DownloadResult(files=["001.jpg", "003.jpg"], downloaded=0, skipped=2, rejected=2)
    assert (dest / "acquire.json").read_text(encoding="utf-8") == manifest_text


def test_apply_filters_false_keeps_everything(tmp_path: Path) -> None:
    urls = [*urls_for(3), "https://cdn.example.org/ch1/logo.png"]
    payload = [
        noise_image("JPEG", seed=1),
        noise_image("JPEG", seed=2, width=200, height=500),
        noise_image("JPEG", seed=3),
        noise_image("PNG", seed=4),
    ]
    server = Server()
    for url, data in zip(urls, payload, strict=True):
        server.script(url, data)
    sleeps: list[float] = []

    result = download_chapter(
        [ImageRef(url=url) for url in urls],
        tmp_path / "c",
        client=server.client(),
        apply_filters=False,
        sleep=sleeps.append,
    )

    assert result == DownloadResult(
        files=["001.jpg", "002.jpg", "003.jpg", "004.png"], downloaded=4, skipped=0, rejected=0
    )
    assert len(server.requests) == 4


def test_invalid_content_is_reported_and_not_written(tmp_path: Path) -> None:
    urls = urls_for(3)
    payloads = [
        b"<html><body>blocked</body></html>",
        noise_image("GIF", seed=2, width=400, height=600),
        noise_image("JPEG", seed=3)[:40],
    ]
    server = Server()
    for url, data in zip(urls, payloads, strict=True):
        server.script(url, data)
    dest = tmp_path / "c"

    with pytest.raises(AcquireError) as excinfo:
        download_chapter(
            [ImageRef(url=url) for url in urls], dest, client=server.client(), sleep=lambda _s: None
        )

    assert str(excinfo.value) == (
        "3 of 3 image(s) failed: "
        "#1 not a supported image (undecodable); "
        "#2 not a supported image (GIF); "
        "#3 not a supported image (undecodable)"
    )
    assert [p.name for p in dest.iterdir()] == ["acquire.json"]
    manifest = json.loads((dest / "acquire.json").read_text(encoding="utf-8"))
    assert manifest == {"images": [], "rejected": []}


def test_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    urls = urls_for(1)
    server = Server()
    server.script(urls[0], noise_image("JPEG", seed=1))
    monkeypatch.setattr(download_mod, "MAX_IMAGE_BYTES", 1000)
    dest = tmp_path / "c"

    with pytest.raises(AcquireError, match="#1 too large"):
        download_chapter([ImageRef(url=urls[0])], dest, client=server.client(), sleep=lambda _s: None)

    assert [p.name for p in dest.iterdir()] == ["acquire.json"]


def test_drm_refused_before_any_request(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request may be made to a DRM platform")

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=30.0)
    dest = tmp_path / "Chapter 1"

    with pytest.raises(DrmPlatformError, match="Naver Webtoon"):
        download_chapter(
            [
                ImageRef(url="https://comic.naver.com/webtoon/1"),
                ImageRef(url="https://cdn.example.org/1.jpg"),
            ],
            dest,
            client=client,
            sleep=lambda _s: None,
        )

    assert not dest.exists()


def test_invalid_manifest_is_treated_as_empty(tmp_path: Path) -> None:
    urls = urls_for(2)
    payload = [noise_image("JPEG", seed=1), noise_image("JPEG", seed=2)]
    server = Server()
    for url, data in zip(urls, payload, strict=True):
        server.script(url, data)
    dest = tmp_path / "c"
    download_chapter([ImageRef(url=url) for url in urls], dest, client=server.client(), sleep=lambda _s: None)
    (dest / "acquire.json").write_bytes(b"{not json")

    re_server = Server()
    for url, data in zip(urls, payload, strict=True):
        re_server.script(url, data)
    result = download_chapter(
        [ImageRef(url=url) for url in urls], dest, client=re_server.client(), sleep=lambda _s: None
    )

    assert [str(r.url) for r in re_server.requests] == urls
    assert result.downloaded == 2
    assert (dest / "002.jpg").read_bytes() == payload[1]


def test_nested_dest_is_created(tmp_path: Path) -> None:
    urls = urls_for(1)
    server = Server()
    server.script(urls[0], noise_image("JPEG", seed=1))
    dest = tmp_path / "library" / "Series" / "Chapter 1"

    result = download_chapter([ImageRef(url=urls[0])], dest, client=server.client(), sleep=lambda _s: None)

    assert (dest / "001.jpg").exists()
    assert result.files == ["001.jpg"]


def test_earlier_rejections_are_fetched_when_the_filters_are_switched_off(tmp_path: Path) -> None:
    urls = [*urls_for(3), "https://cdn.example.org/ch1/logo.png"]
    payload = [
        noise_image("JPEG", seed=1),
        noise_image("JPEG", seed=2, width=200, height=500),
        noise_image("JPEG", seed=3),
        noise_image("PNG", seed=4),
    ]
    dest = tmp_path / "c"
    sleeps: list[float] = []
    first = Server()
    for url, data in zip(urls, payload, strict=True):
        first.script(url, data)
    download_chapter([ImageRef(url=url) for url in urls], dest, client=first.client(), sleep=sleeps.append)

    second = Server()
    for url, data in zip(urls, payload, strict=True):
        second.script(url, data)
    result = download_chapter(
        [ImageRef(url=url) for url in urls],
        dest,
        client=second.client(),
        apply_filters=False,
        sleep=sleeps.append,
    )

    assert [str(r.url) for r in second.requests] == [urls[1], urls[3]]  # only the formerly rejected refs
    assert result == DownloadResult(
        files=["001.jpg", "002.jpg", "003.jpg", "004.png"], downloaded=2, skipped=2, rejected=0
    )
    assert json.loads((dest / "acquire.json").read_text(encoding="utf-8"))["rejected"] == []

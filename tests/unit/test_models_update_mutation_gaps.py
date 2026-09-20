"""Tests closing the gaps found by the mutation review (card Q4).

Each test kills a mutant that survived the existing suite; see `tests/mutants/models_update/`
and `docs/reports/Q4.md`. The tests are self-contained on purpose: they pin one behaviour each.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from omniscan.models.catalog import ModelEntry
from omniscan.models.download import download_model
from omniscan.update.download import download_update
from omniscan.update.github import Asset, ReleaseInfo, UpdateError, list_app_releases
from omniscan.update.version import parse_version


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def file_entry(body: bytes = b"abcd") -> ModelEntry:
    return ModelEntry(
        id="inpaint-big-lama",
        name="LaMa",
        kind="inpaint",
        format="file",
        size_mb=1,
        license="Apache-2.0",
        description="d",
        mirror_url="https://mirror/big-lama.pt",
        sha256=hashlib.sha256(body).hexdigest(),
        bytes=len(body),
        upstream_url="https://upstream/big-lama.pt",
        install_path="lama/big-lama.pt",
    )


def llm_entry() -> ModelEntry:
    return ModelEntry(
        id="llm-x",
        name="X",
        kind="llm",
        format="ollama",
        size_mb=1,
        license="Gemma terms",
        description="d",
        ollama_name="x:1b",
    )


# ---------------------------------------------------------------- models: download order


def test_file_download_tries_the_mirror_before_upstream(tmp_path: Path) -> None:
    body = b"abcd"
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(200, content=body)

    result = download_model(file_entry(body), tmp_path, client=mock_client(handler))
    assert result == "mirror"
    assert hosts == ["mirror"]  # the upstream URL is never touched when the mirror works


def test_stream_without_content_length_reports_an_unknown_total(tmp_path: Path) -> None:
    body = b"abcd"
    entry = file_entry(body)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=iter([body]))  # streamed: no Content-Length header

    progress: list[tuple[str, int, int | None]] = []
    result = download_model(
        entry,
        tmp_path,
        client=mock_client(handler),
        on_progress=lambda mid, done, total: progress.append((mid, done, total)),
    )
    assert result == "mirror"
    assert (tmp_path / "lama" / "big-lama.pt").read_bytes() == body
    assert progress[-1] == (entry.id, len(body), None)


# ---------------------------------------------------------------- models: ollama


def test_ollama_pull_succeeds_when_the_stream_ends_without_a_success_line(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text='{"status": "pulling", "completed": 3, "total": 10}\n',
            headers={"content-type": "application/x-ndjson"},
        )

    result = download_model(llm_entry(), tmp_path, client=mock_client(handler), ollama_url="http://ollama")
    assert result == "ollama"


# ---------------------------------------------------------------- catalog validation


def test_validate_for_format_names_the_first_missing_field() -> None:
    entry = ModelEntry(
        id="det",
        name="Det",
        kind="vision",
        format="zip",
        size_mb=1,
        license="Apache-2.0",
        description="d",
        bytes=1,
        upstream_repo="org/det",
        upstream_revision="rev",
    )
    with pytest.raises(ValueError, match="needs mirror_url"):
        entry.validate_for_format()


def test_validate_for_format_rejects_uppercase_sha256() -> None:
    entry = ModelEntry(
        id="det",
        name="Det",
        kind="vision",
        format="zip",
        size_mb=1,
        license="Apache-2.0",
        description="d",
        mirror_url="https://mirror/det.zip",
        sha256="A" * 64,
        bytes=1,
        upstream_repo="org/det",
        upstream_revision="rev",
    )
    with pytest.raises(ValueError, match="64-hex"):
        entry.validate_for_format()


# ---------------------------------------------------------------- updater: listing


def test_http_400_is_reported_with_the_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400)

    with pytest.raises(UpdateError) as exc_info:
        list_app_releases(mock_client(handler), "owner/repo")
    assert str(exc_info.value) == "GitHub answered HTTP 400"


# ---------------------------------------------------------------- updater: staging


ASSET_NAME = "omniscan-windows-x64.zip"
PARTS = tuple(bytes([65 + i]) * 100 for i in range(10))
ASSET_BYTES = b"".join(PARTS)
ASSET_SHA256 = hashlib.sha256(ASSET_BYTES).hexdigest()
SUMS_TEXT = f"{ASSET_SHA256}  {ASSET_NAME}\n"


def make_release(asset_name: str = ASSET_NAME) -> ReleaseInfo:
    return ReleaseInfo(
        tag="v1.0.0",
        version=parse_version("v1.0.0"),
        prerelease=False,
        notes="",
        published_at=None,
        assets=(
            Asset(asset_name, f"https://gh/releases/v1.0.0/{asset_name}", len(ASSET_BYTES)),
            Asset("SHA256SUMS", "https://gh/releases/v1.0.0/SHA256SUMS", len(SUMS_TEXT)),
        ),
    )


def sums_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=SUMS_TEXT)


def test_staging_reports_http_400(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        if name == "SHA256SUMS":
            return sums_handler(request)
        return httpx.Response(400)

    with pytest.raises(UpdateError, match="HTTP 400 while downloading"):
        download_update(mock_client(handler), make_release(), tmp_path, key="windows-x64")
    assert not list(tmp_path.rglob("*.part"))


def test_progress_reports_done_first_total_second(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        if name == "SHA256SUMS":
            return sums_handler(request)
        return httpx.Response(200, content=iter(PARTS), headers={"Content-Length": str(len(ASSET_BYTES))})

    progress: list[tuple[int, int | None]] = []
    download_update(
        mock_client(handler),
        make_release(),
        tmp_path,
        key="windows-x64",
        on_progress=lambda done, total: progress.append((done, total)),
    )
    dones = [done for done, _ in progress]
    assert dones[0] == len(PARTS[0])  # cumulative done bytes, never the total
    assert len(set(dones)) == len(dones)
    assert progress[-1] == (len(ASSET_BYTES), len(ASSET_BYTES))


def test_staging_streams_without_content_length(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        if name == "SHA256SUMS":
            return sums_handler(request)
        return httpx.Response(200, content=iter(PARTS))  # streamed: no Content-Length header

    staged = download_update(mock_client(handler), make_release(), tmp_path, key="windows-x64")
    assert staged.path == tmp_path / "v1.0.0" / ASSET_NAME
    assert staged.path.read_bytes() == ASSET_BYTES

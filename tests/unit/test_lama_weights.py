"""Tests for LaMa weight download + verification (card C6b, part 1) — the HTTP transport is mocked."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from omniscan.core.config import InpaintConfig
from omniscan.inpaint.lama_weights import ensure_lama_weights, lama_path, sha256_file

CONTENT = b"fake lama torchscript weights"


def cfg() -> InpaintConfig:
    """A config pointing at a fake URL/file with the hash of CONTENT."""
    return InpaintConfig(
        lama_url="https://example.com/models/big-lama.pt",
        lama_sha256=hashlib.sha256(CONTENT).hexdigest(),
        lama_file="fake-lama.pt",
    )


def serving(content: bytes, status: int = 200) -> httpx.MockTransport:
    """A transport that answers every request with `content` and `status`."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=content)

    return httpx.MockTransport(handler)


def refusing(request: httpx.Request) -> httpx.Response:
    """A transport handler that fails the test: no request may be made."""
    raise AssertionError(f"no request expected, got {request.method} {request.url}")


def test_lama_path_layout(tmp_path: Path) -> None:
    assert lama_path(tmp_path / "models", cfg()) == tmp_path / "models" / "lama" / "fake-lama.pt"


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"hello world")
    assert sha256_file(path) == hashlib.sha256(b"hello world").hexdigest()


def test_existing_file_with_right_hash_is_returned_without_request(tmp_path: Path) -> None:
    path = lama_path(tmp_path, cfg())
    path.parent.mkdir(parents=True)
    path.write_bytes(CONTENT)
    client = httpx.Client(transport=httpx.MockTransport(refusing))
    try:
        assert ensure_lama_weights(tmp_path, cfg(), client=client) == path
    finally:
        client.close()
    assert path.read_bytes() == CONTENT


def test_missing_file_is_downloaded_atomically(tmp_path: Path) -> None:
    path = lama_path(tmp_path, cfg())
    client = httpx.Client(transport=serving(CONTENT))
    try:
        assert ensure_lama_weights(tmp_path, cfg(), client=client) == path
    finally:
        client.close()
    assert path.read_bytes() == CONTENT
    lama_dir = tmp_path / "lama"
    assert [p.name for p in lama_dir.iterdir()] == ["fake-lama.pt"]  # no .part left behind


def test_wrong_hash_from_server_raises_and_cleans_up(tmp_path: Path) -> None:
    bad = b"corrupted weights"
    client = httpx.Client(transport=serving(bad))
    try:
        with pytest.raises(RuntimeError) as excinfo:
            ensure_lama_weights(tmp_path, cfg(), client=client)
    finally:
        client.close()
    message = str(excinfo.value)
    assert cfg().lama_sha256 in message and hashlib.sha256(bad).hexdigest() in message
    assert not lama_path(tmp_path, cfg()).exists()
    assert list((tmp_path / "lama").glob("*.part")) == []


def test_http_error_status_raises(tmp_path: Path) -> None:
    client = httpx.Client(transport=serving(b"", status=404))
    try:
        with pytest.raises(RuntimeError, match="LaMa download failed: HTTP 404"):
            ensure_lama_weights(tmp_path, cfg(), client=client)
    finally:
        client.close()
    assert not lama_path(tmp_path, cfg()).exists()


def test_stale_wrong_hash_file_is_replaced(tmp_path: Path) -> None:
    path = lama_path(tmp_path, cfg())
    path.parent.mkdir(parents=True)
    path.write_bytes(b"stale garbage")
    client = httpx.Client(transport=serving(CONTENT))
    try:
        assert ensure_lama_weights(tmp_path, cfg(), client=client) == path
    finally:
        client.close()
    assert path.read_bytes() == CONTENT
    assert list((tmp_path / "lama").glob("*.part")) == []

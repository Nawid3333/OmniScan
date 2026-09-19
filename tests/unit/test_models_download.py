"""Tests for omniscan.models.download (card U2a, parts 3): mirror, fallback, ollama, remove."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from omniscan.models import download as dl
from omniscan.models.catalog import ModelEntry
from omniscan.models.download import (
    ModelDownloadError,
    download_model,
    remove_model,
    verify_models,
)
from omniscan.models.store import MARKER_NAME

MIRROR = "https://mirror"


def make_zip_bytes(model_id: str, *names: str, payload: str = "weights") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{model_id}/config.json", "{}")
        for name in names:
            zf.writestr(f"{model_id}/{name}", payload)
    return buf.getvalue()


def zip_entry(body: bytes, model_id: str = "det") -> ModelEntry:
    return ModelEntry(
        id=model_id,
        name="Det",
        kind="vision",
        required=True,
        format="zip",
        size_mb=1,
        license="Apache-2.0",
        description="d",
        mirror_url=f"{MIRROR}/{model_id}.zip",
        sha256=hashlib.sha256(body).hexdigest(),
        bytes=len(body),
        upstream_repo="org/det",
        upstream_revision="rev1",
    )


def file_entry(body: bytes = b"abcd") -> ModelEntry:
    return ModelEntry(
        id="inpaint-big-lama",
        name="LaMa",
        kind="inpaint",
        format="file",
        size_mb=1,
        license="Apache-2.0",
        description="d",
        mirror_url=f"{MIRROR}/big-lama.pt",
        sha256=hashlib.sha256(body).hexdigest(),
        bytes=len(body),
        upstream_url="https://upstream/big-lama.pt",
        install_path="lama/big-lama.pt",
    )


def llm_entry(fmt: str = "ollama") -> ModelEntry:
    return ModelEntry(
        id="llm-x",
        name="X",
        kind="llm",
        format=fmt,  # type: ignore[arg-type]
        size_mb=1,
        license="Gemma terms",
        description="d",
        ollama_name="x:1b",
    )


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


class RecordingFake:
    """Fake huggingface_hub.snapshot_download: records calls, writes one file, returns the dir."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = fail

    def __call__(self, repo_id: str, **kwargs: Any) -> str:
        self.calls.append({"repo_id": repo_id, **kwargs})
        if self.fail:
            raise RuntimeError("hf down")
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "model.safetensors").write_text("upstream weights", encoding="utf-8")
        return str(local_dir)


def serving(body: bytes, *, mirror_status: int = 200, upstream_status: int = 404):
    """Handler serving `body` from the mirror and 404 elsewhere (upstream GETs get upstream_status)."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.host}{request.url.path}")
        if request.url.host == "mirror":
            if mirror_status != 200:
                return httpx.Response(mirror_status)
            return httpx.Response(200, headers={"content-length": str(len(body))}, content=body)
        if request.url.host == "upstream":
            if upstream_status != 200:
                return httpx.Response(upstream_status)
            return httpx.Response(200, headers={"content-length": str(len(body))}, content=body)
        raise AssertionError(f"unexpected host {request.url.host}")

    handler.calls = calls  # type: ignore[attr-defined]
    return handler


# ---------------------------------------------------------------- mirror success


def test_download_zip_from_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl, "_CHUNK", 8)  # several progress steps for a small body
    body = make_zip_bytes("det", "model.safetensors", payload="0123456789")
    handler = serving(body)
    progress: list[tuple[str, int, int | None]] = []

    result = download_model(
        zip_entry(body),
        tmp_path,
        client=mock_client(handler),
        on_progress=lambda mid, done, total: progress.append((mid, done, total)),
    )

    assert result == "mirror"
    folder = tmp_path / "det"
    assert (folder / "config.json").read_text(encoding="utf-8") == "{}"
    assert (folder / "model.safetensors").read_text(encoding="utf-8") == "0123456789"
    marker = json.loads((folder / MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["source"] == "mirror"
    assert marker["sha256"] == hashlib.sha256(body).hexdigest()
    assert marker["revision"] == "rev1"
    assert list(tmp_path.rglob("*.part")) == []  # nothing left over
    done_values = [p[1] for p in progress]
    assert done_values == sorted(done_values) and len(set(done_values)) == len(done_values)
    assert progress[-1] == ("det", len(body), len(body))  # monotonic, ends at the total


def test_second_download_is_noop_without_requests(tmp_path: Path) -> None:
    body = make_zip_bytes("det", "model.safetensors")
    entry = zip_entry(body)
    handler = serving(body)
    client = mock_client(handler)
    assert download_model(entry, tmp_path, client=client) == "mirror"
    before = len(handler.calls)  # type: ignore[attr-defined]
    assert download_model(entry, tmp_path, client=client) == "already installed"
    assert len(handler.calls) == before  # type: ignore[attr-defined]


def test_download_file_entry_from_mirror(tmp_path: Path) -> None:
    body = b"abcd"
    result = download_model(
        file_entry(body), tmp_path, client=mock_client(serving(body, upstream_status=500))
    )
    assert result == "mirror"
    assert (tmp_path / "lama" / "big-lama.pt").read_bytes() == body
    assert list(tmp_path.rglob("*.part")) == []


def test_file_entry_already_installed_makes_no_request(tmp_path: Path) -> None:
    entry = file_entry(b"abcd")
    target = tmp_path / "lama" / "big-lama.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"abcd")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    assert download_model(entry, tmp_path, client=mock_client(handler)) == "already installed"


# ---------------------------------------------------------------- integrity and fallback


def test_wrong_bytes_fall_back_to_upstream(tmp_path: Path) -> None:
    body = make_zip_bytes("det", "model.safetensors")
    entry = zip_entry(body)
    fake = RecordingFake()
    client = mock_client(serving(b"corrupted" + body, upstream_status=404))  # wrong bytes on mirror

    result = download_model(entry, tmp_path, client=client, hf_download=fake)

    assert result == "upstream"
    assert list(tmp_path.rglob("*.part")) == []  # the failed .part is deleted
    assert len(fake.calls) == 1
    assert fake.calls[0]["repo_id"] == "org/det"
    assert fake.calls[0]["revision"] == "rev1"
    assert fake.calls[0]["local_dir"] == str(tmp_path / "det")
    marker = json.loads((tmp_path / "det" / MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["source"] == "upstream"


def test_mirror_404_falls_back_to_upstream(tmp_path: Path) -> None:
    body = make_zip_bytes("det", "model.safetensors")
    fake = RecordingFake()
    result = download_model(
        zip_entry(body), tmp_path, client=mock_client(serving(body, mirror_status=404)), hf_download=fake
    )
    assert result == "upstream"
    assert (tmp_path / "det" / "model.safetensors").is_file()


def test_both_sources_failing_lists_both_messages(tmp_path: Path) -> None:
    body = make_zip_bytes("det", "model.safetensors")
    fake = RecordingFake(fail=True)
    with pytest.raises(ModelDownloadError) as exc_info:
        download_model(
            zip_entry(body),
            tmp_path,
            client=mock_client(serving(body, mirror_status=404)),
            hf_download=fake,
        )
    message = str(exc_info.value)
    assert "mirror: HTTP 404" in message
    assert "upstream: huggingface org/det: RuntimeError: hf down" in message
    assert list(tmp_path.rglob("*.part")) == []


def test_file_upstream_hash_mismatch_is_a_hard_error(tmp_path: Path) -> None:
    entry = file_entry(b"abcd")  # expected hash of b"abcd"
    bad = b"XXXX"  # right size, wrong bytes
    client = mock_client(serving(bad, mirror_status=404, upstream_status=200))  # upstream: wrong bytes
    with pytest.raises(ModelDownloadError) as exc_info:
        download_model(entry, tmp_path, client=client)
    assert "sha256 mismatch" in str(exc_info.value)
    assert not (tmp_path / "lama" / "big-lama.pt").exists()  # the file is (never) installed
    assert list(tmp_path.rglob("*.part")) == []


def test_zip_slip_relative_member(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("det/ok.txt", "fine")
        zf.writestr("../evil.txt", "escaped")
    body = buf.getvalue()
    with pytest.raises(ModelDownloadError, match="zip-slip"):
        download_model(zip_entry(body), tmp_path, client=mock_client(serving(body)))
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "det").exists()  # nothing was extracted
    assert list(tmp_path.rglob("*.part")) == []


def test_zip_slip_absolute_member(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("/abs/evil.txt", "escaped")
    body = buf.getvalue()
    with pytest.raises(ModelDownloadError, match="zip-slip"):
        download_model(zip_entry(body), tmp_path, client=mock_client(serving(body)))
    assert list(tmp_path.rglob("*.part")) == []


# ---------------------------------------------------------------- ollama


def ndjson_client(lines: list[dict[str, Any]], calls: list[tuple[str, Any]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((f"{request.method} {request.url.path}", body))
        payload = "\n".join(json.dumps(line) for line in lines)
        return httpx.Response(200, text=payload, headers={"content-type": "application/x-ndjson"})

    return mock_client(handler)


def test_ollama_pull_reports_progress(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    lines: list[dict[str, Any]] = [
        {"status": "pulling manifest"},
        {"status": "pulling", "completed": 5, "total": 10},
        {"status": "pulling", "completed": 10, "total": 10},
        {"status": "success"},
    ]
    progress: list[tuple[str, int, int | None]] = []
    result = download_model(
        llm_entry(),
        tmp_path,
        client=ndjson_client(lines, calls),
        ollama_url="http://ollama",
        on_progress=lambda mid, done, total: progress.append((mid, done, total)),
    )
    assert result == "ollama"
    assert progress == [("llm-x", 5, 10), ("llm-x", 10, 10)]
    assert calls == [("POST /api/pull", {"model": "x:1b", "stream": True})]


def test_ollama_error_line_fails(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    lines: list[dict[str, Any]] = [{"status": "pulling"}, {"error": "model not found"}]
    with pytest.raises(ModelDownloadError, match="model not found"):
        download_model(llm_entry(), tmp_path, client=ndjson_client(lines, calls), ollama_url="http://ollama")


def test_ollama_http_error_fails(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with pytest.raises(ModelDownloadError, match="HTTP 500"):
        download_model(llm_entry(), tmp_path, client=mock_client(handler), ollama_url="http://ollama")


def test_cloud_entry_has_nothing_to_download(tmp_path: Path) -> None:
    with pytest.raises(ModelDownloadError, match="Ollama Cloud"):
        download_model(llm_entry("cloud"), tmp_path, client=mock_client(serving(b"")))


def test_remove_ollama_sends_delete(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, json.loads(request.content) if request.content else None))
        return httpx.Response(200, json={"deleted": True})

    assert remove_model(llm_entry(), tmp_path, client=mock_client(handler), ollama_url="http://ollama")
    assert calls == [("DELETE", {"model": "x:1b"})]


# ---------------------------------------------------------------- remove and verify


def test_remove_zip_folder(tmp_path: Path) -> None:
    body = make_zip_bytes("det", "model.safetensors")
    entry = zip_entry(body)
    download_model(entry, tmp_path, client=mock_client(serving(body)))
    assert remove_model(entry, tmp_path) is True
    assert not (tmp_path / "det").exists()
    assert remove_model(entry, tmp_path) is False


def test_remove_file_and_empty_parent(tmp_path: Path) -> None:
    entry = file_entry(b"abcd")
    target = tmp_path / "lama" / "big-lama.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"abcd")
    assert remove_model(entry, tmp_path) is True
    assert not target.exists()
    assert not (tmp_path / "lama").exists()  # emptied parent folder is removed too
    assert remove_model(entry, tmp_path) is False


def test_remove_cloud_is_never_a_remove(tmp_path: Path) -> None:
    assert remove_model(llm_entry("cloud"), tmp_path) is False


def test_verify_models_returns_statuses(tmp_path: Path) -> None:
    body = make_zip_bytes("det", "model.safetensors")
    entry = zip_entry(body)
    statuses = verify_models([entry, llm_entry("cloud")], tmp_path)
    assert statuses == {"det": "missing", "llm-x": "cloud"}
    download_model(entry, tmp_path, client=mock_client(serving(body)))
    assert verify_models([entry], tmp_path) == {"det": "installed"}

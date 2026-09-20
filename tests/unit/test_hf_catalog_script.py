"""Tests for scripts/hf_catalog.py (card O1a, part 2): canned Hugging Face JSON, no network."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from omniscan.models.catalog import ModelEntry

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "hf_catalog.py"
_spec = importlib.util.spec_from_file_location("hf_catalog", _SCRIPT)
assert _spec is not None and _spec.loader is not None
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

REPO = "org/rec"
HEAD = "d" * 40
OTHER_REVISION = "e" * 40
LFS_SHA = "a" * 64
CONFIG_BYTES = b'{"arch": "x"}'
CONFIG_SHA = hashlib.sha256(CONFIG_BYTES).hexdigest()

METADATA: dict[str, Any] = {
    "id": REPO,
    "sha": HEAD,
    "cardData": {"license": "apache-2.0", "language": ["en", "zh"]},
    "siblings": [{"rfilename": "model.safetensors"}, {"rfilename": "config.json"}],
}
TREE: list[dict[str, Any]] = [
    {
        "type": "file",
        "path": "model.safetensors",
        "size": 4_000_000,
        "lfs": {"oid": LFS_SHA, "size": 4_000_000},
    },
    {"type": "file", "path": "config.json", "size": 500},
    {"type": "directory", "path": "sub", "size": 0},
    {"type": "file", "path": "sub/nested.bin", "size": 9_000_000},  # subfolder: never listed
    {"type": "file", "path": "README.md", "size": 300},  # not a model file
    {"type": "file", "path": ".gitattributes", "size": 20},  # not a model file
    {"type": "file", "path": "poster.png", "size": 9_000_000},  # not a model file
]


class FakeFetch:
    """Canned Hugging Face API: metadata + tree JSON, raw bytes for the small config file.

    The canned `model.safetensors` is LFS, so the fake never serves it — a download attempt for it
    (an AssertionError) fails the test.
    """

    def __init__(self, *, status: int | None = None) -> None:
        self.status = status
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        if self.status is not None:
            raise script.FetchError(f"HTTP {self.status} for {url}")
        api = f"https://huggingface.co/api/models/{REPO}"
        if url == f"{api}?blobs=true":
            return json.dumps(METADATA).encode()
        if url.startswith(f"{api}/tree/") and url.endswith("?recursive=true"):
            return json.dumps(TREE).encode()
        if "/resolve/" in url and url.endswith("/config.json"):
            return CONFIG_BYTES
        raise AssertionError(f"unexpected fetch: {url}")


def run(*args: str, fetch: Callable[[str], bytes] | None = None) -> tuple[int, str, str]:
    """Run the script's main and return (code, stdout, stderr) without touching the network."""
    import contextlib
    import io

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = script.main(list(args), fetch=fetch if fetch is not None else FakeFetch())
    return code, out.getvalue(), err.getvalue()


def parse_entry(block: str) -> ModelEntry:
    """tomllib-parse a printed block and build the (format-validated) catalog entry."""
    entry = ModelEntry(**tomllib.loads(block)["model"][0])
    entry.validate_for_format()  # must not raise
    return entry


def test_block_parses_and_is_a_valid_model_entry(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _err = run(
        REPO, "--id", "ocr-rec-x", "--role", "recognizer", "--family", "ppocrv6", "--size-class", "small"
    )
    assert code == 0
    entry = parse_entry(out)
    assert entry.id == "ocr-rec-x"
    assert entry.name == "rec"
    assert entry.kind == "ocr" and entry.format == "hf" and entry.required is False
    assert entry.role == "recognizer" and entry.family == "ppocrv6" and entry.size_class == "small"
    assert entry.upstream_repo == REPO and entry.upstream_revision == HEAD
    assert entry.license == "apache-2.0"
    assert entry.langs == ["en", "zh"]
    assert entry.size_mb == 5  # ceil((4_000_000 + 500) / 1_000_000) of the listed files only
    assert entry.files == {"model.safetensors": LFS_SHA, "config.json": CONFIG_SHA}
    assert "sub/nested.bin" not in entry.files  # subfolder, README, .gitattributes, images are gone


def test_lfs_oid_is_used_without_downloading_the_file() -> None:
    fetch = FakeFetch()
    code, out, _err = run(REPO, fetch=fetch)
    assert code == 0
    assert parse_entry(out).files["model.safetensors"] == LFS_SHA
    assert not any("/resolve/" in url and "/model.safetensors" in url for url in fetch.urls)
    assert any("/resolve/" in url and url.endswith("/config.json") for url in fetch.urls)  # small: hashed


def test_revision_overrides_the_head_sha() -> None:
    fetch = FakeFetch()
    code, out, _err = run(REPO, "--revision", OTHER_REVISION, fetch=fetch)
    assert code == 0
    entry = parse_entry(out)
    assert entry.upstream_revision == OTHER_REVISION
    assert any(f"/tree/{OTHER_REVISION}?recursive=true" in url for url in fetch.urls)
    assert not any(f"/tree/{HEAD}" in url for url in fetch.urls)


def test_unknown_repo_exits_1_with_a_clear_error(capsys: pytest.CaptureFixture[str]) -> None:
    code, _out, err = run("org/missing", fetch=FakeFetch(status=404))
    assert code == 1
    assert "error" in err and "404" in err
    assert not capsys.readouterr().out.strip()


def test_default_id_and_name_derive_from_the_repo() -> None:
    class Repo(FakeFetch):
        def __call__(self, url: str) -> bytes:  # the canned URLs are repo-independent except the id
            return super().__call__(url.replace("PP-OCRv6_small_rec_safetensors", "rec"))

    code, out, _err = run("org/PP-OCRv6_small_rec_safetensors", fetch=Repo())
    assert code == 0
    entry = parse_entry(out)
    assert entry.id == "pp-ocrv6-small-rec"  # _safetensors stripped, like the name
    assert entry.name == "PP-OCRv6 small rec"


def test_no_model_files_is_an_error_not_a_block(capsys: pytest.CaptureFixture[str]) -> None:
    fetch = FakeFetch()
    empty_tree = json.dumps([{"type": "file", "path": "README.md", "size": 10}]).encode()

    def serving(url: str) -> bytes:
        fetch.urls.append(url)
        if "?blobs=true" in url:
            return json.dumps(METADATA).encode()
        return empty_tree

    code, _out, err = run(REPO, fetch=serving)
    assert code == 1
    assert "no model files" in err

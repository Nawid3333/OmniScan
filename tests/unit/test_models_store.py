"""Tests for omniscan.models.store (card U2a, part 2: statuses and resolve_model_path)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from omniscan.core.config import Config, PathsConfig
from omniscan.models.catalog import ModelEntry
from omniscan.models.store import (
    MARKER_NAME,
    file_sha256,
    install_path,
    model_status,
    resolve_model_path,
)

SHA = "b" * 64
OTHER_SHA = "c" * 64


def zip_entry(model_id: str = "det") -> ModelEntry:
    return ModelEntry(
        id=model_id,
        name="Det",
        kind="vision",
        required=True,
        format="zip",
        size_mb=1,
        license="Apache-2.0",
        description="d",
        used_by=["detect.repo"],
        mirror_url=f"https://mirror/{model_id}.zip",
        sha256=SHA,
        bytes=10,
        upstream_repo="org/det",
        upstream_revision="rev1",
    )


def file_entry() -> ModelEntry:
    return ModelEntry(
        id="inpaint-big-lama",
        name="LaMa",
        kind="inpaint",
        format="file",
        size_mb=1,
        license="Apache-2.0",
        description="d",
        mirror_url="https://mirror/big-lama.pt",
        sha256=SHA,
        bytes=4,
        upstream_url="https://upstream/big-lama.pt",
        install_path="lama/big-lama.pt",
    )


def ollama_entry(fmt: str = "ollama") -> ModelEntry:
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


def make_zip_bytes(model_id: str, payload: str = "hello") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{model_id}/weights.bin", payload)
        zf.writestr(f"{model_id}/meta.json", "{}")
    return buf.getvalue()


def install_fake_zip(models_dir: Path, model_id: str, *, sha: str = SHA, marker: dict | None = None) -> None:
    folder = models_dir / model_id
    folder.mkdir(parents=True)
    (folder / "weights.bin").write_text("hello", encoding="utf-8")
    body = marker if marker is not None else {"id": model_id, "sha256": sha, "source": "mirror"}
    (folder / MARKER_NAME).write_text(json.dumps(body), encoding="utf-8")


def cfg(tmp_path: Path) -> Config:
    return Config(paths=PathsConfig(models_dir=tmp_path / "models"))


# ---------------------------------------------------------------- install_path


def test_install_path_per_format(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    assert install_path(zip_entry(), models_dir) == models_dir / "det"
    assert install_path(file_entry(), models_dir) == models_dir / "lama" / "big-lama.pt"
    assert install_path(ollama_entry(), models_dir) is None
    assert install_path(ollama_entry("cloud"), models_dir) is None


# ---------------------------------------------------------------- zip statuses


def test_zip_missing_folder(tmp_path: Path) -> None:
    assert model_status(zip_entry(), tmp_path, ollama_names=None) == "missing"


def test_zip_folder_without_marker_is_corrupt(tmp_path: Path) -> None:
    (tmp_path / "det").mkdir()
    assert model_status(zip_entry(), tmp_path, ollama_names=None) == "corrupt"


def test_zip_wrong_hash_marker_is_corrupt(tmp_path: Path) -> None:
    install_fake_zip(tmp_path, "det", sha=OTHER_SHA)
    assert model_status(zip_entry(), tmp_path, ollama_names=None) == "corrupt"


def test_zip_unparsable_marker_is_corrupt(tmp_path: Path) -> None:
    (tmp_path / "det").mkdir()
    (tmp_path / "det" / MARKER_NAME).write_text("{not json", encoding="utf-8")
    assert model_status(zip_entry(), tmp_path, ollama_names=None) == "corrupt"


def test_zip_matching_marker_is_installed(tmp_path: Path) -> None:
    install_fake_zip(tmp_path, "det")
    assert model_status(zip_entry(), tmp_path, ollama_names=None) == "installed"


# ---------------------------------------------------------------- file statuses


def test_file_missing(tmp_path: Path) -> None:
    assert model_status(file_entry(), tmp_path, ollama_names=None) == "missing"


def test_file_wrong_size_is_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "lama" / "big-lama.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"too long!")
    assert model_status(file_entry(), tmp_path, ollama_names=None) == "corrupt"


def test_file_wrong_hash_is_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "lama" / "big-lama.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"abc")  # right size (4), wrong hash
    assert model_status(file_entry(), tmp_path, ollama_names=None) == "corrupt"


def test_file_matching_is_installed(tmp_path: Path) -> None:
    path = tmp_path / "lama" / "big-lama.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"abcd")
    entry = file_entry().model_copy(update={"sha256": file_sha256(path)})
    assert model_status(entry, tmp_path, ollama_names=None) == "installed"


# ---------------------------------------------------------------- ollama / cloud statuses


def test_ollama_installed_missing_unknown(tmp_path: Path) -> None:
    entry = ollama_entry()
    assert model_status(entry, tmp_path, ollama_names={"x:1b"}) == "installed"
    assert model_status(entry, tmp_path, ollama_names={"y:2b"}) == "missing"
    assert model_status(entry, tmp_path, ollama_names=None) == "unknown"


def test_cloud_status(tmp_path: Path) -> None:
    assert model_status(ollama_entry("cloud"), tmp_path, ollama_names=None) == "cloud"


# ---------------------------------------------------------------- resolve_model_path


def test_resolve_model_path_only_when_installed(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    catalog = [zip_entry(), file_entry(), ollama_entry()]
    assert resolve_model_path("det", config, catalog) is None  # missing
    assert resolve_model_path("nope", config, catalog) is None  # unknown id
    assert resolve_model_path("llm-x", config, catalog) is None  # ollama -> never a path
    install_fake_zip(config.paths.models_dir, "det")
    assert resolve_model_path("det", config, catalog) == config.paths.models_dir / "det"


def test_resolve_model_path_loads_the_real_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from omniscan.models import catalog as catalog_module

    config = cfg(tmp_path)
    monkeypatch.setattr(catalog_module, "default_catalog_path", lambda: tmp_path / "none.toml")
    assert resolve_model_path("detector-comic-text-bubble", config) is None  # not installed here

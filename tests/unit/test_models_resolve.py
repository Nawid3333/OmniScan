"""Tests for omniscan.models.resolve (card U2b, part 1: a repo id -> the installed folder)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omniscan.models.catalog import ModelEntry
from omniscan.models.resolve import local_model_source
from omniscan.models.store import MARKER_NAME, file_sha256

SHA = "b" * 64
OTHER_SHA = "c" * 64


def zip_entry(model_id: str, repo: str) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        name="M",
        kind="vision",
        required=True,
        format="zip",
        size_mb=1,
        license="Apache-2.0",
        description="d",
        mirror_url=f"https://mirror/{model_id}.zip",
        sha256=SHA,
        bytes=10,
        upstream_repo=repo,
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
        upstream_repo="org/lama",
        upstream_url="https://upstream/big-lama.pt",
        install_path="lama/big-lama.pt",
    )


def ollama_entry() -> ModelEntry:
    return ModelEntry(
        id="llm-x",
        name="X",
        kind="llm",
        format="ollama",
        size_mb=1,
        license="Gemma terms",
        description="d",
        upstream_repo="org/ollama",
        ollama_name="x:1b",
    )


def install_fake_zip(models_dir: Path, model_id: str, *, sha: str = SHA) -> None:
    folder = models_dir / model_id
    folder.mkdir(parents=True)
    (folder / "weights.bin").write_text("hello", encoding="utf-8")
    (folder / MARKER_NAME).write_text(json.dumps({"id": model_id, "sha256": sha}), encoding="utf-8")


# ---------------------------------------------------------------- zip entries


def test_installed_repo_resolves_to_the_folder(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    install_fake_zip(models_dir, "det")
    catalog = [zip_entry("det", "org/det")]
    assert local_model_source("org/det", models_dir, catalog) == str(models_dir / "det")


def test_missing_folder_returns_none(tmp_path: Path) -> None:
    assert local_model_source("org/det", tmp_path / "models", [zip_entry("det", "org/det")]) is None


def test_corrupt_marker_returns_none(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    install_fake_zip(models_dir, "det", sha=OTHER_SHA)  # wrong hash in the marker: corrupt
    assert local_model_source("org/det", models_dir, [zip_entry("det", "org/det")]) is None


def test_unknown_repo_returns_none(tmp_path: Path) -> None:
    assert local_model_source("org/other", tmp_path / "models", [zip_entry("det", "org/det")]) is None


def test_file_and_ollama_entries_never_match(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    file = file_entry()
    lama = models_dir / "lama" / "big-lama.pt"
    lama.parent.mkdir(parents=True)
    lama.write_bytes(b"abcd")  # installed (right size, right hash)
    catalog = [file.model_copy(update={"sha256": file_sha256(lama)}), ollama_entry()]
    assert local_model_source("org/lama", models_dir, catalog) is None
    assert local_model_source("org/ollama", models_dir, catalog) is None


def test_two_entries_with_different_repos_resolve_independently(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    catalog = [zip_entry("a", "org/a"), zip_entry("b", "org/b")]
    install_fake_zip(models_dir, "b")
    assert local_model_source("org/a", models_dir, catalog) is None
    assert local_model_source("org/b", models_dir, catalog) == str(models_dir / "b")


# ---------------------------------------------------------------- catalog loading


def test_unreadable_catalog_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from omniscan.models import catalog as catalog_module

    broken = tmp_path / "models.toml"
    broken.write_text("[[model]]\nid = 'x'\nname = 'X'\n", encoding="utf-8")  # missing format fields
    monkeypatch.setattr(catalog_module, "default_catalog_path", lambda: broken)
    monkeypatch.setattr(catalog_module, "machine_catalog_path", lambda: tmp_path / "none.toml")
    assert local_model_source("org/det", tmp_path) is None


def test_shipped_catalog_repo_not_installed_returns_none(tmp_path: Path) -> None:
    assert local_model_source("ogkalu/comic-text-and-bubble-detector", tmp_path) is None

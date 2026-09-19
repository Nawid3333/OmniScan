"""Tests for omniscan.models.catalog (card U2a, parts 1-2)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from omniscan.core.config import DetectConfig, InpaintConfig, OcrConfig
from omniscan.models import catalog
from omniscan.models.catalog import ModelEntry, load_catalog

IDS = (
    "detector-comic-text-bubble",
    "ocr-det-ppocrv5-server",
    "ocr-rec-korean-ppocrv5-mobile",
    "inpaint-big-lama",
    "llm-translategemma-12b",
    "llm-gemma4-12b",
    "llm-gemma4-31b",
    "llm-gemma4-31b-cloud",
)

MIRROR_BASE = "https://github.com/Nawid3333/OmniScan/releases/download/models-v1/"
SHA = "a" * 64

REPO_TOML = """\
[[model]]
id = "detector"
name = "Detector"
kind = "vision"
required = true
format = "zip"
size_mb = 159
license = "Apache-2.0"
description = "detector"
used_by = ["detect.repo"]
mirror_url = "{base}detector.zip"
sha256 = "{sha}"
bytes = 100
upstream_repo = "org/detector"
upstream_revision = "rev1"

[[model]]
id = "lama"
name = "LaMa"
kind = "inpaint"
format = "file"
size_mb = 206
license = "Apache-2.0"
description = "inpainter"
used_by = []
mirror_url = "{base}big-lama.pt"
sha256 = "{sha}"
bytes = 200
upstream_url = "https://example.com/big-lama.pt"
install_path = "lama/big-lama.pt"
"""

MACHINE_TOML = """\
[[model]]
id = "detector"
name = "Detector (pinned)"
size_mb = 200
license = "Apache-2.0"
description = "detector, machine-pinned"
used_by = []
kind = "vision"
required = true
format = "zip"
mirror_url = "{base}detector.zip"
sha256 = "{sha2}"
bytes = 150
upstream_repo = "org/detector"
upstream_revision = "rev2"

[[model]]
id = "extra"
name = "Extra"
kind = "llm"
format = "ollama"
size_mb = 1
license = "Gemma terms"
description = "extra model"
used_by = []
ollama_name = "extra:1b"
"""


def _write(path: Path, text: str, **subs: str) -> Path:
    path.write_text(text.format(base=MIRROR_BASE, **subs), encoding="utf-8")
    return path


# ---------------------------------------------------------------- the real shipped catalog


def test_real_catalog_entries_in_order_and_valid() -> None:
    entries = load_catalog()
    assert [e.id for e in entries] == list(IDS)
    assert len({e.id for e in entries}) == len(entries)
    for entry in entries:
        entry.validate_for_format()  # must not raise
    for entry in entries:
        if entry.format in ("zip", "file"):
            assert entry.sha256 is not None and len(entry.sha256) == 64
            assert all(c in "0123456789abcdef" for c in entry.sha256)
            assert entry.mirror_url is not None and entry.mirror_url.startswith(MIRROR_BASE)
            assert entry.bytes is not None and entry.bytes > 0


def test_real_catalog_required_flags() -> None:
    entries = load_catalog()
    assert [e.required for e in entries] == [True, True, True, False, False, False, False, False]


def test_real_catalog_matches_pipeline_contracts() -> None:
    entries = {e.id: e for e in load_catalog()}
    assert entries["inpaint-big-lama"].sha256 == InpaintConfig().lama_sha256
    assert entries["inpaint-big-lama"].upstream_url == InpaintConfig().lama_url
    assert entries["inpaint-big-lama"].install_path == "lama/big-lama.pt"
    assert entries["detector-comic-text-bubble"].upstream_repo == DetectConfig().repo
    assert entries["ocr-det-ppocrv5-server"].upstream_repo == OcrConfig().det_repo
    assert entries["ocr-rec-korean-ppocrv5-mobile"].upstream_repo == OcrConfig().rec_repo


def test_real_catalog_ollama_names_match_profiles_and_judge() -> None:
    repo = Path(__file__).resolve().parents[2]
    with (repo / "config" / "translation_profiles.toml").open("rb") as fh:
        profiles = tomllib.load(fh)["profiles"]
    with (repo / "config" / "judge.toml").open("rb") as fh:
        judge = tomllib.load(fh)["judge"]
    known = {p["model"] for p in profiles.values()} | {judge["model"]}
    entries = {e.id: e for e in load_catalog()}
    for model_id in ("llm-translategemma-12b", "llm-gemma4-12b", "llm-gemma4-31b-cloud"):
        assert entries[model_id].ollama_name in known
    assert entries["llm-gemma4-31b-cloud"].used_by == ["gemma4-31b-cloud", "judge"]


# ---------------------------------------------------------------- validate_for_format


def test_zip_entry_without_sha256_names_the_field() -> None:
    entry = ModelEntry(
        id="det",
        name="Det",
        kind="vision",
        format="zip",
        size_mb=1,
        license="Apache-2.0",
        description="d",
        mirror_url=f"{MIRROR_BASE}det.zip",
        bytes=1,
        upstream_repo="org/det",
        upstream_revision="rev",
    )
    with pytest.raises(ValueError, match="sha256"):
        entry.validate_for_format()


def test_ollama_entry_without_ollama_name_names_the_field() -> None:
    entry = ModelEntry(
        id="m",
        name="M",
        kind="llm",
        format="ollama",
        size_mb=1,
        license="Gemma terms",
        description="d",
    )
    with pytest.raises(ValueError, match="ollama_name"):
        entry.validate_for_format()


# ---------------------------------------------------------------- merging and validation


def test_machine_file_replaces_entry_and_adds_new(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _write(tmp_path / "repo.toml", REPO_TOML, sha=SHA)
    machine = _write(tmp_path / "machine.toml", MACHINE_TOML, sha=SHA, sha2="b" * 64)
    monkeypatch.setattr(catalog, "default_catalog_path", lambda: repo)
    monkeypatch.setattr(catalog, "machine_catalog_path", lambda: machine)

    entries = load_catalog()
    assert [e.id for e in entries] == ["detector", "lama", "extra"]
    by_id = {e.id: e for e in entries}
    assert by_id["detector"].name == "Detector (pinned)"  # replaced by the machine entry
    assert by_id["detector"].sha256 == "b" * 64
    assert by_id["lama"].name == "LaMa"  # untouched
    assert by_id["extra"].ollama_name == "extra:1b"  # machine-only addition goes last


def test_machine_file_may_be_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _write(tmp_path / "repo.toml", REPO_TOML, sha=SHA)
    monkeypatch.setattr(catalog, "default_catalog_path", lambda: repo)
    monkeypatch.setattr(catalog, "machine_catalog_path", lambda: tmp_path / "missing.toml")
    assert [e.id for e in load_catalog()] == ["detector", "lama"]


def test_explicit_path_reads_only_that_file(tmp_path: Path) -> None:
    repo = _write(tmp_path / "repo.toml", REPO_TOML, sha=SHA)
    assert [e.id for e in load_catalog(repo)] == ["detector", "lama"]


def test_duplicate_ids_in_one_file_raise(tmp_path: Path) -> None:
    text = REPO_TOML.format(base=MIRROR_BASE, sha=SHA)
    path = tmp_path / "cat.toml"
    path.write_text(text + text[: text.index("install_path")], encoding="utf-8")  # lama repeated
    with pytest.raises(ValueError, match="duplicate model id 'detector'"):
        load_catalog(path)


def test_unknown_key_raises(tmp_path: Path) -> None:
    text = REPO_TOML.format(base=MIRROR_BASE, sha=SHA).replace(
        'upstream_revision = "rev1"', 'upstream_revision = "rev1"\nbogus = 1'
    )
    path = tmp_path / "cat.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="bogus"):
        load_catalog(path)


def test_zip_entry_without_sha256_raises_on_load(tmp_path: Path) -> None:
    text = REPO_TOML.format(base=MIRROR_BASE, sha=SHA).replace(
        f'sha256 = "{SHA}"\nbytes = 100\n', "bytes = 100\n", 1
    )
    path = tmp_path / "cat.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        load_catalog(path)

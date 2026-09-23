"""Tests for omniscan.models.catalog (cards U2a and O1a, parts 1-2)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

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
    # --- PP-OCRv6 (card O1a)
    "ocr-det-ppocrv6-tiny",
    "ocr-det-ppocrv6-small",
    "ocr-det-ppocrv6-medium",
    "ocr-rec-ppocrv6-tiny",
    "ocr-rec-ppocrv6-small",
    "ocr-rec-ppocrv6-medium",
    # --- PP-OCRv5 additions (card O1a)
    "ocr-det-ppocrv5-mobile",
    "ocr-rec-ppocrv5-server",
    "ocr-rec-ppocrv5-mobile",
    "ocr-rec-en-ppocrv5-mobile",
    "ocr-rec-latin-ppocrv5-mobile",
    "ocr-rec-eslav-ppocrv5-mobile",
    "ocr-rec-th-ppocrv5-mobile",
    "ocr-rec-el-ppocrv5-mobile",
    "ocr-rec-arabic-ppocrv5-mobile",
    "ocr-rec-cyrillic-ppocrv5-mobile",
    "ocr-rec-devanagari-ppocrv5-mobile",
    "ocr-rec-te-ppocrv5-mobile",
    "ocr-rec-ta-ppocrv5-mobile",
    # --- PaddleOCR-VL and GGUF (card O1a)
    "ocr-vl-1.6",
    "ocr-vl-1.5",
    "ocr-vl",
    "ocr-vl-1.6-gguf",
    "ocr-vl-1.5-gguf",
    # --- manga-ocr (card O1a)
    "ocr-rec-manga-ocr-base",
    "ocr-rec-manga-ocr-2025",
)

LEGACY_IDS = IDS[:8]  # the entries that shipped before O1a
V6_IDS = tuple(i for i in IDS if "ppocrv6" in i)
V5_LANG_RECS = (
    ("ocr-rec-en-ppocrv5-mobile", ["en"]),
    ("ocr-rec-latin-ppocrv5-mobile", []),
    ("ocr-rec-eslav-ppocrv5-mobile", []),
    ("ocr-rec-th-ppocrv5-mobile", ["th"]),
    ("ocr-rec-el-ppocrv5-mobile", ["el"]),
    ("ocr-rec-arabic-ppocrv5-mobile", ["ar"]),
    ("ocr-rec-cyrillic-ppocrv5-mobile", []),
    ("ocr-rec-devanagari-ppocrv5-mobile", []),
    ("ocr-rec-te-ppocrv5-mobile", ["te"]),
    ("ocr-rec-ta-ppocrv5-mobile", ["ta"]),
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


# ---------------------------------------------------------------- hardware requirement fields (H1)


def test_real_catalog_requirement_fields() -> None:
    entries = {e.id: e for e in load_catalog()}
    det = entries["detector-comic-text-bubble"]
    assert (det.min_vram_gb, det.min_ram_gb, det.cpu_speed) == (1.0, 4.0, "slow")
    assert (entries["ocr-det-ppocrv5-server"].min_vram_gb, entries["ocr-det-ppocrv5-server"].cpu_speed) == (
        0.8,
        "ok",
    )
    assert (
        entries["ocr-rec-korean-ppocrv5-mobile"].min_vram_gb,
        entries["ocr-rec-korean-ppocrv5-mobile"].cpu_speed,
    ) == (
        0.5,
        "ok",
    )
    lama = entries["inpaint-big-lama"]
    assert (lama.min_vram_gb, lama.min_ram_gb, lama.cpu_speed) == (2.0, 8.0, "slow")
    assert lama.notes == "fp32 only; the first inference per window shape warms up for 10-25 s"
    translategemma = entries["llm-translategemma-12b"]
    assert (translategemma.min_vram_gb, translategemma.min_ram_gb, translategemma.cpu_speed) == (
        9.0,
        16.0,
        "unusable",
    )
    assert (entries["llm-gemma4-12b"].min_vram_gb, entries["llm-gemma4-12b"].cpu_speed) == (8.5, "unusable")
    assert (entries["llm-gemma4-31b"].min_vram_gb, entries["llm-gemma4-31b"].min_ram_gb) == (21.0, 32.0)
    cloud = entries["llm-gemma4-31b-cloud"]
    assert (cloud.cpu_ok, cloud.cpu_speed, cloud.min_vram_gb) == (True, "fast", 0.0)
    assert cloud.notes == "needs internet and an Ollama account"
    assert all(e.backends == [] for e in entries.values())  # empty = all backends


def test_requirement_fields_default_when_absent() -> None:
    entry = ModelEntry(
        id="m",
        name="M",
        kind="vision",
        format="zip",
        size_mb=1,
        license="Apache-2.0",
        description="d",
        mirror_url=f"{MIRROR_BASE}m.zip",
        sha256=SHA,
        bytes=1,
        upstream_repo="org/m",
        upstream_revision="rev",
    )
    assert entry.min_vram_gb is None
    assert entry.min_ram_gb is None
    assert entry.backends == []
    assert entry.cpu_ok is True
    assert entry.cpu_speed == "ok"
    assert entry.notes == ""


def test_invalid_cpu_speed_raises_value_error() -> None:
    with pytest.raises(ValueError, match="cpu_speed"):
        ModelEntry(
            id="m",
            name="M",
            kind="vision",
            format="zip",
            size_mb=1,
            license="Apache-2.0",
            description="d",
            cpu_speed="very fast",  # type: ignore[arg-type]  # rejected by pydantic at runtime
        )


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


# ---------------------------------------------------------------- format hf (card O1a)

HF_REVISION = "c" * 40
HF_FILES = {"model.safetensors": "a" * 64, "config.json": "b" * 64}


def hf_entry(**overrides: Any) -> ModelEntry:
    """A valid hf entry; every keyword overrides one field."""
    data: dict[str, Any] = {
        "id": "ocr-rec-x",
        "name": "X",
        "kind": "ocr",
        "format": "hf",
        "size_mb": 2,
        "license": "Apache-2.0",
        "description": "d",
        "upstream_repo": "org/rec",
        "upstream_revision": HF_REVISION,
        "files": dict(HF_FILES),
    }
    data.update(overrides)
    return ModelEntry(**data)


def test_hf_entry_is_valid_with_field_defaults() -> None:
    entry = hf_entry()
    entry.validate_for_format()  # must not raise
    assert entry.role is None
    assert entry.family == ""
    assert entry.size_class == ""
    assert entry.langs == []
    assert entry.recommended_for == []
    assert entry.notes == ""


def test_hf_entry_missing_upstream_repo_names_the_field() -> None:
    with pytest.raises(ValueError, match="needs upstream_repo"):
        hf_entry(upstream_repo=None).validate_for_format()


def test_hf_entry_missing_revision_names_the_field() -> None:
    with pytest.raises(ValueError, match="needs upstream_revision"):
        hf_entry(upstream_revision=None).validate_for_format()


def test_hf_entry_short_revision_needs_40_hex() -> None:
    with pytest.raises(ValueError, match="40-hex upstream_revision"):
        hf_entry(upstream_revision="c" * 39).validate_for_format()


def test_hf_entry_empty_files_needs_files() -> None:
    with pytest.raises(ValueError, match="needs files"):
        hf_entry(files={}).validate_for_format()


def test_hf_entry_bad_file_hash_names_the_file() -> None:
    entry = hf_entry(files={"model.safetensors": "a" * 64, "config.json": "nothex"})
    with pytest.raises(ValueError, match=r"files\['config\.json'\] needs a 64-hex sha256"):
        entry.validate_for_format()


def test_hf_entry_zero_size_mb_needs_size_mb() -> None:
    with pytest.raises(ValueError, match="needs size_mb > 0"):
        hf_entry(size_mb=0).validate_for_format()


def test_hf_entry_loads_from_toml_files_table(tmp_path: Path) -> None:
    path = tmp_path / "cat.toml"
    path.write_text(
        "\n".join(
            [
                "[[model]]",
                'id = "ocr-rec-x"',
                'name = "X"',
                'kind = "ocr"',
                'format = "hf"',
                "size_mb = 2",
                'license = "Apache-2.0"',
                'description = "d"',
                'upstream_repo = "org/rec"',
                f'upstream_revision = "{HF_REVISION}"',
                "",
                "[model.files]",
                f'"model.safetensors" = "{"a" * 64}"',
                f'"config.json" = "{"b" * 64}"',
            ]
        ),
        encoding="utf-8",
    )
    entry = load_catalog(path)[0]
    assert entry.files == HF_FILES
    entry.validate_for_format()  # must not raise


# ---------------------------------------------------------------- descriptive fields in the real catalog


def test_real_catalog_roles() -> None:
    entries = {e.id: e for e in load_catalog()}
    assert entries["detector-comic-text-bubble"].role == "detector"
    assert entries["ocr-det-ppocrv5-server"].role == "text_line_detector"
    assert entries["ocr-rec-korean-ppocrv5-mobile"].role == "recognizer"
    assert entries["inpaint-big-lama"].role == "inpaint"
    for model_id in LEGACY_IDS[4:]:
        assert entries[model_id].role == "llm"


def test_real_catalog_recommended_for_matches_the_measured_defaults() -> None:
    entries = load_catalog()
    recommended = {e.id: e.recommended_for for e in entries if e.recommended_for}
    assert recommended == {
        "ocr-det-ppocrv5-server": ["ko"],
        "ocr-rec-korean-ppocrv5-mobile": ["ko"],
        # card O1c, 2026-09-23: best chrF on Japanese of any candidate and faster than the prior
        # ja default, despite the upstream model card listing only en/zh support
        "ocr-det-ppocrv6-medium": ["ja"],
        "ocr-rec-ppocrv6-medium": ["ja"],
    }
    korean_rec = next(e for e in entries if e.id == "ocr-rec-korean-ppocrv5-mobile")
    assert korean_rec.langs == ["ko"]
    assert korean_rec.family == "ppocrv5" and korean_rec.size_class == "mobile"


def test_real_catalog_hf_entries_are_upstream_only_with_file_hashes() -> None:
    entries = [e for e in load_catalog() if e.format == "hf"]
    assert len(entries) == 26
    for entry in entries:
        assert entry.mirror_url is None and entry.sha256 is None
        assert entry.upstream_repo and entry.upstream_revision
        assert len(entry.upstream_revision) == 40  # validate_for_format checks the hex
        assert entry.files and entry.size_mb > 0
        assert all(len(sha) == 64 and set(sha) <= set("0123456789abcdef") for sha in entry.files.values())


def test_real_catalog_v6_entries_are_en_zh_only() -> None:
    entries = {e.id: e for e in load_catalog()}
    assert set(V6_IDS) <= set(entries)
    for model_id in V6_IDS:
        entry = entries[model_id]
        assert entry.role == ("text_line_detector" if "det" in model_id else "recognizer")
        assert entry.family == "ppocrv6" and entry.format == "hf"
        assert entry.langs == ["en", "zh"] and "ko" not in entry.langs
    for model_id in (i for i in V6_IDS if "rec" in i):
        assert "Korean" in entries[model_id].notes
    assert {entries[i].size_class for i in V6_IDS} == {"tiny", "small", "medium"}


def test_real_catalog_v5_language_recognisers() -> None:
    entries = {e.id: e for e in load_catalog()}
    for model_id, langs in V5_LANG_RECS:
        entry = entries[model_id]
        assert entry.role == "recognizer" and entry.family == "ppocrv5" and entry.format == "hf"
        assert entry.langs == langs
        if not langs:
            assert "repo name names the script/family" in entry.notes
    base_rec = entries["ocr-rec-ppocrv5-mobile"]
    assert base_rec.langs == ["en", "zh"] and "zh-Hant" in base_rec.notes
    assert entries["ocr-det-ppocrv5-mobile"].role == "text_line_detector"
    assert entries["ocr-rec-ppocrv5-server"].size_class == "server"


def test_real_catalog_vl_gguf_and_manga_entries() -> None:
    entries = {e.id: e for e in load_catalog()}
    for model_id in ("ocr-vl-1.6", "ocr-vl-1.5", "ocr-vl"):
        entry = entries[model_id]
        assert entry.role == "vlm_ocr" and entry.family == "paddleocr-vl" and entry.format == "hf"
        assert entry.langs == ["en", "zh", "multilingual"] and entry.size_mb > 1000
    for model_id in ("ocr-vl-1.6-gguf", "ocr-vl-1.5-gguf"):
        entry = entries[model_id]
        assert entry.role == "vlm_ocr" and entry.format == "hf"
        assert "llama.cpp" in entry.notes
        assert all(name.endswith((".gguf", ".jinja")) for name in entry.files)
    for model_id in ("ocr-rec-manga-ocr-base", "ocr-rec-manga-ocr-2025"):
        entry = entries[model_id]
        assert entry.role == "recognizer" and entry.family == "manga-ocr" and entry.langs == ["ja"]


def test_real_catalog_required_flags() -> None:
    required = [e.id for e in load_catalog() if e.required]
    assert required == [LEGACY_IDS[0], LEGACY_IDS[1], LEGACY_IDS[2]]  # the pre-O1a Korean stack only

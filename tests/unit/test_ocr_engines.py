"""model_source / load_kwargs / engine_rec_model against a hand-made catalog (card O1b, tests 1-2)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

from omniscan.core.config import OcrConfig
from omniscan.models.catalog import ModelEntry
from omniscan.models.store import MARKER_NAME
from omniscan.ocr.engines import (
    DEFAULT_REC_MODEL,
    ModelSource,
    engine_rec_model,
    load_kwargs,
    model_entry,
    model_source,
)

REVISION = "d" * 40


def hf_entry(model_id: str, repo: str) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        name="M",
        kind="ocr",
        format="hf",
        size_mb=1,
        license="Apache-2.0",
        description="d",
        upstream_repo=repo,
        upstream_revision=REVISION,
        files={"model.safetensors": hashlib.sha256(b"weights").hexdigest()},
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


def install_fake_hf(models_dir: Path, model_id: str) -> None:
    folder = models_dir / model_id
    folder.mkdir(parents=True)
    (folder / "model.safetensors").write_text("weights", encoding="utf-8")
    marker = {
        "id": model_id,
        "source": "upstream",
        "revision": REVISION,
        "files_verified": 1,
        "file_sizes": {"model.safetensors": 7},  # len(b"weights")
    }
    (folder / MARKER_NAME).write_text(json.dumps(marker), encoding="utf-8")


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[list[ModelEntry], Path]:
    entries = [hf_entry("ocr-rec-a", "org/a"), hf_entry("ocr-rec-b", "org/b"), ollama_entry()]
    install_fake_hf(tmp_path / "models", "ocr-rec-a")
    monkeypatch.setattr("omniscan.ocr.engines.load_catalog", lambda: entries)
    return entries, tmp_path / "models"


# ---------------------------------------------------------------- model_source (test 1)


def test_installed_model_resolves_to_its_folder(catalog: tuple[list[ModelEntry], Path]) -> None:
    _entries, models_dir = catalog
    source = model_source("ocr-rec-a", models_dir)
    assert source == ModelSource(
        repo="org/a", revision=REVISION, local=str(models_dir / "ocr-rec-a"), id="ocr-rec-a"
    )


def test_missing_model_falls_back_to_the_pinned_hub(catalog: tuple[list[ModelEntry], Path]) -> None:
    _entries, models_dir = catalog
    source = model_source("ocr-rec-b", models_dir)
    assert source.local is None and source.repo == "org/b" and source.revision == REVISION


def test_models_dir_none_never_looks_installed(catalog: tuple[list[ModelEntry], Path]) -> None:
    _entries, _models_dir = catalog
    assert model_source("ocr-rec-a", None).local is None


def test_unknown_model_id_raises(catalog: tuple[list[ModelEntry], Path]) -> None:
    _entries, _models_dir = catalog
    with pytest.raises(ValueError, match="unknown OCR model 'nope'"):
        model_source("nope", None)


def test_non_hugging_face_model_raises(catalog: tuple[list[ModelEntry], Path]) -> None:
    _entries, _models_dir = catalog
    with pytest.raises(ValueError, match="llm-x is not a Hugging Face model"):
        model_source("llm-x", None)


def test_model_entry_lookup(catalog: tuple[list[ModelEntry], Path]) -> None:
    entries, _models_dir = catalog
    assert model_entry("ocr-rec-b", entries).upstream_repo == "org/b"
    with pytest.raises(ValueError, match="unknown OCR model 'nope'"):
        model_entry("nope", entries)


# ---------------------------------------------------------------- load_kwargs (test 2)


def test_load_kwargs_installed(
    catalog: tuple[list[ModelEntry], Path], caplog: pytest.LogCaptureFixture
) -> None:
    _entries, models_dir = catalog
    source = model_source("ocr-rec-a", models_dir)
    with caplog.at_level(logging.INFO):
        assert load_kwargs(source, models_dir=models_dir) == (
            str(models_dir / "ocr-rec-a"),
            {"local_files_only": True},
        )
    assert f"loading org/a from {models_dir / 'ocr-rec-a'}" in caplog.text


def test_load_kwargs_hub_with_revision(
    catalog: tuple[list[ModelEntry], Path], caplog: pytest.LogCaptureFixture
) -> None:
    _entries, models_dir = catalog
    source = model_source("ocr-rec-b", models_dir)
    with caplog.at_level(logging.WARNING):
        assert load_kwargs(source, models_dir=models_dir) == ("org/b", {"revision": REVISION})
    warnings = [r for r in caplog.records if r.name == "omniscan.ocr.engines"]
    assert len(warnings) == 1
    assert "omniscan models download ocr-rec-b" in warnings[0].message  # the warning names the id


def test_load_kwargs_hub_without_revision(catalog: tuple[list[ModelEntry], Path]) -> None:
    entries, models_dir = catalog
    source = model_source(
        "ocr-rec-b", models_dir, [entries[1].model_copy(update={"upstream_revision": None})]
    )
    assert load_kwargs(source, models_dir=models_dir) == ("org/b", {})


def test_load_kwargs_without_models_dir_never_warns(
    catalog: tuple[list[ModelEntry], Path], caplog: pytest.LogCaptureFixture
) -> None:
    _entries, _models_dir = catalog
    source = model_source("ocr-rec-b", None)
    assert load_kwargs(source, models_dir=None) == ("org/b", {"revision": REVISION})
    assert "not installed" not in caplog.text


# ---------------------------------------------------------------- engine_rec_model


def test_engine_rec_model() -> None:
    assert engine_rec_model(OcrConfig(engine="manga_ocr")) == "ocr-rec-manga-ocr-2025"
    assert engine_rec_model(OcrConfig(engine="manga_ocr", rec_model="ocr-rec-manga-ocr-base")) == (
        "ocr-rec-manga-ocr-base"
    )
    assert engine_rec_model(OcrConfig(engine="ppocr")) is None
    assert engine_rec_model(OcrConfig(engine="ppocr", rec_model="ocr-rec-ppocrv6-medium")) == (
        "ocr-rec-ppocrv6-medium"
    )
    assert engine_rec_model(OcrConfig(engine="paddleocr_vl")) == "ocr-vl-1.6"
    assert engine_rec_model(OcrConfig(engine="paddleocr_vl", rec_model="ocr-vl-1.5")) == "ocr-vl-1.5"


def test_default_rec_model_table() -> None:
    assert DEFAULT_REC_MODEL == {"manga_ocr": "ocr-rec-manga-ocr-2025", "paddleocr_vl": "ocr-vl-1.6"}

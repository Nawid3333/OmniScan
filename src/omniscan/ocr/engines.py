"""OCR engine support: catalog model ids -> Hugging Face sources for the loaders (card O1b).

`ocr.det_model` / `ocr.rec_model` name catalog entries; `model_source` maps an id to the installed
folder (offline) or the pinned hub repo, `load_kwargs` turns that into `from_pretrained` arguments,
and `engine_rec_model` resolves an engine's recogniser id (`cfg.rec_model` or the engine default).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omniscan.core.config import OcrConfig
from omniscan.models.catalog import ModelEntry, load_catalog
from omniscan.models.resolve import local_model_source

log = logging.getLogger(__name__)

DEFAULT_REC_MODEL: dict[str, str] = {
    "manga_ocr": "ocr-rec-manga-ocr-2025",  # paddleocr_vl is added by card O1d
}


@dataclass(frozen=True, slots=True)
class ModelSource:
    """Where a catalog model loads from: the pinned HF repo, or the installed folder when present."""

    repo: str  # Hugging Face repo id of the catalog entry
    revision: str | None  # pinned commit (catalog upstream_revision) or None
    local: str | None  # installed folder (as str) or None when not installed
    id: str  # catalog id (names the model in warnings; card O1b deviation: added to this dataclass)


def model_entry(model_id: str, catalog: Sequence[ModelEntry] | None = None) -> ModelEntry:
    """The catalog entry for `model_id`; an unknown id is a ValueError."""
    entries = load_catalog() if catalog is None else catalog
    for entry in entries:
        if entry.id == model_id:
            return entry
    raise ValueError(f"unknown OCR model {model_id!r}")


def model_source(
    model_id: str, models_dir: Path | None, catalog: Sequence[ModelEntry] | None = None
) -> ModelSource:
    """Map a catalog model id to its HF repo/revision and installed folder (None when not installed)."""
    entries = catalog if catalog is not None else load_catalog()
    entry = model_entry(model_id, entries)
    if entry.format not in ("hf", "zip") or entry.upstream_repo is None:
        raise ValueError(f"{model_id} is not a Hugging Face model")
    local = local_model_source(entry.upstream_repo, models_dir, entries) if models_dir is not None else None
    return ModelSource(repo=entry.upstream_repo, revision=entry.upstream_revision, local=local, id=entry.id)


def load_kwargs(source: ModelSource, *, models_dir: Path | None) -> tuple[str, dict[str, Any]]:
    """(path_or_repo, from_pretrained kwargs) for `source`: the installed folder, else the pinned hub."""
    if source.local is not None:
        log.info("loading %s from %s", source.repo, source.local)
        return source.local, {"local_files_only": True}
    if models_dir is not None:
        log.warning(
            "model %s is not installed in %s; using the Hugging Face hub/cache. "
            'Run "omniscan models download %s" to install it.',
            source.id,
            models_dir,
            source.id,
        )
    return source.repo, {"revision": source.revision} if source.revision else {}


def engine_rec_model(cfg: OcrConfig) -> str | None:
    """The recogniser/crop-reader catalog id for cfg: `ocr.rec_model` or the engine's default."""
    return cfg.rec_model or DEFAULT_REC_MODEL.get(cfg.engine)

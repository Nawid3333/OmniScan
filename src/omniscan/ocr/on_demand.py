"""Read one region again with the chapter's OCR engine: the Studio's *Read again* and `omniscan edit ocr`.

After a box was moved or drawn by hand, or when a reading looks wrong, the configured engine reads just that
region: its box plus a margin is cut from the raw pages (Pillow, like the editing tools' crops) and handed to
the same reading functions the `ocr` stage uses — line detection and recognition for ppocr, one crop for the
crop readers. Nothing is written here: the reading is a suggestion the editor keeps as the region's source
text (a hand edit through edits/store.py) or drops. The models come from the vision group of a VRAM manager,
with exclusive GPU access for the read, like the stages. Torch is imported only when a region is read, so the
web app and the CLI stay importable without it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from omniscan.cleanup.strip import strip_crop
from omniscan.core.config import Config
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import BBox, IngestArtifact, Region
from omniscan.edits import store

MARGIN_PX = 24  # room around the box, so the line detector sees whole lines at its edges


@dataclass(frozen=True, slots=True)
class Reading:
    """What the OCR read in one region."""

    region_id: str
    text: str
    confidence: float
    engine: str


def crop_around(box: BBox, width: int, height: int, margin: int = MARGIN_PX) -> BBox:
    """`box` grown by `margin` on every side, inside a strip of `width` x `height`."""
    return BBox(
        x0=max(0, box.x0 - margin),
        y0=max(0, box.y0 - margin),
        x1=min(width, box.x1 + margin),
        y1=min(height, box.y1 + margin),
    )


def in_crop(region: Region, crop: BBox) -> Region:
    """A bare copy of `region` (no lines or text) with its box in the pixels of `crop`."""
    box = region.bbox
    return Region(
        id=region.id,
        slice_index=region.slice_index,
        kind=region.kind,
        lang=region.lang,
        orientation=region.orientation,
        bbox=BBox(x0=box.x0 - crop.x0, y0=box.y0 - crop.y0, x1=box.x1 - crop.x0, y1=box.y1 - crop.y0),
    )


def region_crop(paths: ChapterPaths, region_id: str) -> tuple[Region, BBox, np.ndarray]:
    """The current region `region_id`, the strip box around it and those pixels (uint8 [h, w, 3]).
    EditNotFoundError for an unknown region, FileNotFoundError without ingest.json or a page file."""
    region = next((r for r in store.current_regions(paths) if r.id == region_id), None)
    if region is None:
        raise store.EditNotFoundError(f"region {region_id!r} not found in ocr.json")
    ingest_path = paths.artifact("ingest.json")
    if not ingest_path.is_file():
        raise FileNotFoundError("ingest.json missing — run the slice stage first")
    ingest = IngestArtifact.load(ingest_path)
    crop = crop_around(region.bbox, ingest.strip_width, ingest.strip_height)
    return region, crop, strip_crop(paths, ingest, crop)


def read_region(paths: ChapterPaths, region_id: str, cfg: Config, models: Mapping[str, Any]) -> Reading:
    """Read region `region_id` with the engine `cfg` names and its loaded `models` (a vision group)."""
    import torch  # deferred: only reading needs it

    from omniscan.gpu.device import resolve_device
    from omniscan.ocr.engines import engine_rec_model
    from omniscan.ocr.pipeline import read_region_crops, read_regions

    region, crop, pixels = region_crop(paths, region_id)
    tile = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).to(
        resolve_device(cfg.gpu.device)
    )
    local = in_crop(region, crop)
    if cfg.ocr.engine == "ppocr":
        engine = cfg.ocr.rec_model or cfg.ocr.rec_repo.split("/")[-1]
        read, _ = read_regions(
            tile,
            [local],
            models["line_detector"],
            models["recognizer"],
            cfg.ocr,
            direction=cfg.detect.reading_direction,
            engine=engine,
        )
    else:
        rec_model = engine_rec_model(cfg.ocr)
        if rec_model is None:
            raise ValueError(f"OCR engine '{cfg.ocr.engine}' needs ocr.rec_model")
        engine = rec_model
        read, _ = read_region_crops(tile, [local], models["reader"], cfg.ocr, engine=engine)
    (result,) = read
    return Reading(region_id=region.id, text=result.text, confidence=result.confidence, engine=engine)


@contextmanager
def vision_models(cfg: Config) -> Iterator[Mapping[str, Any]]:
    """The vision model group for one on-demand read, with exclusive GPU access; everything is released after."""
    from omniscan.gpu.groups import VISION_GROUP, build_vram_manager
    from omniscan.gpu.lock import acquire_gpu_lock, release_gpu_lock

    lock = acquire_gpu_lock() if cfg.gpu.device != "cpu" else None
    try:
        manager = build_vram_manager(cfg)
        try:
            yield manager.acquire(VISION_GROUP)
        finally:
            manager.release()
    finally:
        if lock is not None:
            release_gpu_lock(lock)


def read_region_now(paths: ChapterPaths, region_id: str, cfg: Config) -> Reading:
    """`read_region` with the vision models loaded just for this read (the web API's and the CLI's default)."""
    with vision_models(cfg) as models:
        return read_region(paths, region_id, cfg, models)

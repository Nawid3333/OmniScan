"""OCR stage wrapper: ingest.json + slices.json + regions.json + chapter JPEGs -> ocr.json (text filled).

The strip is cut into overlapping tiles (only tiles that touch a region), the line detector runs in
batches, lines are merged across tiles, assigned to regions, cropped and read by the recognizer, and
the results are assembled into the regions of ocr.json via ocr/pipeline.py. With `ocr.engine` set to
a crop-reading engine (manga_ocr today, paddleocr_vl later) every region is read as one whole crop
instead and no line detection runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from omniscan.core.config import Config
from omniscan.core.paths import list_images
from omniscan.core.schemas import IngestArtifact, RegionsArtifact
from omniscan.core.stage import ChapterContext
from omniscan.gpu.groups import VISION_GROUP
from omniscan.ingest.strip import load_strip
from omniscan.ocr.engines import engine_rec_model
from omniscan.ocr.pipeline import read_region_crops, read_regions


class OcrStage:
    """Read the text of detected regions (ppocr or a crop-reading engine; satisfies core.stage.Stage)."""

    name: ClassVar[str] = "ocr"
    version: ClassVar[int] = (
        2  # 2: strips decoded before the CUDA staging-buffer fix (2026-09-19) held duplicated pages
    )
    gpu_group: ClassVar[str | None] = VISION_GROUP

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        """Files whose content determines this stage's output (raw images and/or upstream artifacts)."""
        return [
            ctx.paths.artifact("ingest.json"),
            ctx.paths.artifact("slices.json"),
            ctx.paths.artifact("regions.json"),
            *list_images(ctx.paths.raw_dir),
        ]

    def outputs(self, ctx: ChapterContext) -> list[str]:
        """Artifact names (relative to the chapter work dir) this stage writes."""
        return ["ocr.json"]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        return {**cfg.ocr.model_dump(), "reading_direction": cfg.detect.reading_direction}

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        regions_path = ctx.paths.artifact("regions.json")
        if not regions_path.is_file():
            raise FileNotFoundError("regions.json missing — run the detect stage first")
        ingest_path = ctx.paths.artifact("ingest.json")
        if not ingest_path.is_file():
            raise FileNotFoundError("ingest.json missing — run the slice stage first")
        ingest = IngestArtifact.load(ingest_path)
        regions = RegionsArtifact.load(regions_path)
        strip = load_strip(ctx, ingest)
        cfg = ctx.cfg
        if cfg.ocr.engine == "ppocr":
            ocr_regions, metrics = read_regions(
                strip,
                regions.regions,
                models["line_detector"],
                models["recognizer"],
                cfg.ocr,
                direction=cfg.detect.reading_direction,
                engine=cfg.ocr.rec_model or cfg.ocr.rec_repo.split("/")[-1],
            )
        else:
            engine = engine_rec_model(cfg.ocr)
            if engine is None:
                raise ValueError(f"OCR engine '{cfg.ocr.engine}' needs ocr.rec_model")
            ocr_regions, metrics = read_region_crops(
                strip, regions.regions, models["reader"], cfg.ocr, engine=engine
            )
        # false-positive detections read as junk with a low score (or nothing at all): leave those pixels alone
        kept = [r for r in ocr_regions if r.lines and r.confidence >= ctx.cfg.ocr.drop_conf]
        metrics["regions_dropped"] = float(len(ocr_regions) - len(kept))
        RegionsArtifact(regions=kept).save(ctx.paths.artifact("ocr.json"))
        return metrics

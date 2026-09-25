"""Inpaint stage wrapper: ingest/slices/ocr + chapter JPEGs -> inpaint.json + patches.npz (flat fills + LaMa masks)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from omniscan.core.config import Config
from omniscan.core.paths import list_images
from omniscan.core.schemas import IngestArtifact, RegionsArtifact
from omniscan.core.stage import ChapterContext
from omniscan.ingest.strip import load_strip
from omniscan.inpaint.patches import save_patches
from omniscan.inpaint.pipeline import inpaint_regions


class InpaintStage:
    """Flat-fill the text of every OCR region (satisfies core.stage.Stage); no GPU model group needed."""

    name: ClassVar[str] = "inpaint"
    version: ClassVar[int] = (
        4  # 3: glyph-precise masks, sfx only with sfx.mode = "replace"; 4: watermarks erased
    )
    gpu_group: ClassVar[str | None] = None

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        """Files whose content determines this stage's output (raw images and upstream artifacts)."""
        return [
            ctx.paths.artifact("ingest.json"),
            ctx.paths.artifact("slices.json"),
            ctx.paths.artifact("ocr.json"),
            *list_images(ctx.paths.raw_dir),
        ]

    def outputs(self, ctx: ChapterContext) -> list[str]:
        """Artifact names (relative to the chapter work dir) this stage writes."""
        return ["inpaint.json", "patches.npz"]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        return {**cfg.inpaint.model_dump(), "sfx_mode": cfg.sfx.mode}

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        for name, stage in (("ingest.json", "ingest"), ("slices.json", "slice"), ("ocr.json", "ocr")):
            if not ctx.paths.artifact(name).is_file():
                raise FileNotFoundError(f"{name} missing — run the {stage} stage first")
        ingest = IngestArtifact.load(ctx.paths.artifact("ingest.json"))
        strip = load_strip(ctx, ingest)
        regions = RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions
        artifact, patches, metrics = inpaint_regions(
            strip, regions, ctx.cfg.inpaint, sfx_mode=ctx.cfg.sfx.mode
        )
        save_patches(ctx.paths.artifact("patches.npz"), patches)
        artifact.save(ctx.paths.artifact("inpaint.json"))
        return metrics

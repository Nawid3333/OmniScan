"""Inpaint stage wrapper: ingest/slices/ocr + chapter JPEGs -> inpaint.json + patches.npz (flat fill only)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import torch

from omniscan.core.config import Config
from omniscan.core.paths import list_images
from omniscan.core.schemas import IngestArtifact, RegionsArtifact
from omniscan.core.stage import ChapterContext
from omniscan.gpu.codec.select import get_codec
from omniscan.ingest.strip import build_strip, jpeg_paths
from omniscan.inpaint.patches import save_patches
from omniscan.inpaint.pipeline import inpaint_regions


def load_strip(ctx: ChapterContext, ingest: IngestArtifact) -> torch.Tensor:
    """Decode the chapter strip once per pass (memoised under 'strip', shared with the other stages)."""

    def build() -> torch.Tensor:
        codec = get_codec(ctx.cfg)
        try:
            return build_strip(
                ingest,
                jpeg_paths(ingest, ctx.paths.raw_dir, ctx.paths.work_dir / "converted"),
                codec,
            )
        finally:
            close = getattr(codec, "close", None)
            if close is not None:
                close()

    return ctx.lazy("strip", build)


class InpaintStage:
    """Flat-fill the text of every OCR region (satisfies core.stage.Stage); no GPU model group needed."""

    name: ClassVar[str] = "inpaint"
    version: ClassVar[int] = 1
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
        return cfg.inpaint.model_dump()

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        for name, stage in (("ingest.json", "ingest"), ("slices.json", "slice"), ("ocr.json", "ocr")):
            if not ctx.paths.artifact(name).is_file():
                raise FileNotFoundError(f"{name} missing — run the {stage} stage first")
        ingest = IngestArtifact.load(ctx.paths.artifact("ingest.json"))
        strip = load_strip(ctx, ingest)
        regions = RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions
        artifact, patches, metrics = inpaint_regions(strip, regions, ctx.cfg.inpaint)
        save_patches(ctx.paths.artifact("patches.npz"), patches)
        artifact.save(ctx.paths.artifact("inpaint.json"))
        return metrics

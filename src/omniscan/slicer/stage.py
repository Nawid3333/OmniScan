"""Slice stage wrapper: ingest.json + chapter JPEGs -> slices.json (via the shared strip tensor)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from omniscan.core.config import Config
from omniscan.core.schemas import IngestArtifact
from omniscan.core.stage import ChapterContext
from omniscan.gpu.codec.select import get_codec
from omniscan.ingest.strip import build_strip, jpeg_paths
from omniscan.slicer import slice_strip


class SliceStage:
    """Cut the chapter strip into slices (satisfies core.stage.Stage); no GPU model group needed."""

    name: ClassVar[str] = "slice"
    version: ClassVar[int] = 1
    gpu_group: ClassVar[str | None] = None

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        """Files whose content determines this stage's output (raw images and/or upstream artifacts)."""
        return [ctx.paths.artifact("ingest.json")]

    def outputs(self, ctx: ChapterContext) -> list[str]:
        """Artifact names (relative to the chapter work dir) this stage writes."""
        return ["slices.json"]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        return cfg.slicer.model_dump()

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        ingest_path = ctx.paths.artifact("ingest.json")
        if not ingest_path.is_file():
            raise FileNotFoundError("ingest.json missing — run the ingest stage first")
        ingest = IngestArtifact.load(ingest_path)

        codec = get_codec(ctx.cfg)
        try:
            strip = ctx.lazy(
                "strip",
                lambda: build_strip(
                    ingest,
                    jpeg_paths(ingest, ctx.paths.raw_dir, ctx.paths.work_dir / "converted"),
                    codec,
                ),
            )
            slices = slice_strip(strip, ctx.cfg.slicer, ingest.files)
        finally:
            close = getattr(codec, "close", None)
            if close is not None:
                close()

        slices.save(ctx.paths.artifact("slices.json"))
        return {
            "slices": float(len(slices.slices)),
            "blank": float(sum(1 for s in slices.slices if s.blank)),
            "forced": float(sum(1 for s in slices.slices if s.forced_cut)),
            "bands": float(len(slices.bands)),
            "strip_height": float(slices.strip_height),
        }

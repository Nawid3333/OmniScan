"""Ingest stage wrapper: raw chapter images -> normalised JPEGs + ingest.json (strip layout)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from omniscan.core.config import Config
from omniscan.core.paths import list_images
from omniscan.core.stage import ChapterContext
from omniscan.ingest import ingest_chapter


class IngestStage:
    """Resumable wrapper around `ingest_chapter` (satisfies core.stage.Stage)."""

    name: ClassVar[str] = "ingest"
    version: ClassVar[int] = 1
    gpu_group: ClassVar[str | None] = None

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        """Files whose content determines this stage's output (raw images and/or upstream artifacts)."""
        return list_images(ctx.paths.raw_dir)

    def outputs(self, ctx: ChapterContext) -> list[str]:
        """Artifact names (relative to the chapter work dir) this stage writes."""
        return ["ingest.json"]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        return {"quality": 95}

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        result = ingest_chapter(
            ctx.paths.raw_dir,
            ctx.paths.series,
            ctx.paths.chapter,
            ctx.paths.work_dir / "converted",
            quality=95,
        )
        result.artifact.save(ctx.paths.artifact("ingest.json"))
        files = result.artifact.files
        return {
            "files": float(len(files)),
            "converted": float(sum(1 for f in files if f.converted_from is not None)),
            "strip_height": float(result.artifact.strip_height),
        }

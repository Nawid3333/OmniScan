"""inpaint_lama stage: inpaint.json + patches.npz + raw JPEGs -> inpaint_lama.json + patches_lama.npz."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from omniscan.core.config import Config
from omniscan.core.paths import list_images
from omniscan.core.schemas import IngestArtifact, InpaintArtifact
from omniscan.core.stage import ChapterContext
from omniscan.gpu.groups import INPAINT_GROUP
from omniscan.ingest.strip import load_strip
from omniscan.inpaint.lama_pipeline import lama_regions
from omniscan.inpaint.patches import load_patches, save_patches


class LamaStage:
    """LaMa-clean the `needs_lama` regions the flat fill left behind (satisfies core.stage.Stage)."""

    name: ClassVar[str] = "inpaint_lama"
    version: ClassVar[int] = 1
    gpu_group: ClassVar[str | None] = INPAINT_GROUP

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        """Files whose content determines this stage's output (raw images and upstream artifacts)."""
        return [
            ctx.paths.artifact("ingest.json"),
            ctx.paths.artifact("slices.json"),
            ctx.paths.artifact("inpaint.json"),
            ctx.paths.artifact("patches.npz"),
            *list_images(ctx.paths.raw_dir),
        ]

    def outputs(self, ctx: ChapterContext) -> list[str]:
        """Artifact names (relative to the chapter work dir) this stage writes."""
        return ["inpaint_lama.json", "patches_lama.npz"]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        return cfg.inpaint.model_dump()

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        for name in ("ingest.json", "slices.json", "inpaint.json", "patches.npz"):
            if not ctx.paths.artifact(name).is_file():
                raise FileNotFoundError(f"{name} missing — run the inpaint stage first")
        inpaint = InpaintArtifact.load(ctx.paths.artifact("inpaint.json"))
        strip = load_strip(ctx, IngestArtifact.load(ctx.paths.artifact("ingest.json")))
        artifact, patches, metrics = lama_regions(
            strip,
            inpaint,
            load_patches(ctx.paths.artifact("patches.npz")),
            models["lama"],
            ctx.cfg.inpaint,
        )
        save_patches(ctx.paths.artifact("patches_lama.npz"), patches)
        artifact.save(ctx.paths.artifact("inpaint_lama.json"))
        return metrics

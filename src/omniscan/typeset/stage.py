"""Typeset stage wrapper: ocr.json + final.json + inpaint(_lama).json -> layout.json (no GPU, no images)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from omniscan.core.config import Config
from omniscan.core.schemas import (
    RGB,
    FinalArtifact,
    InpaintArtifact,
    LayoutArtifact,
    RegionsArtifact,
)
from omniscan.core.stage import ChapterContext
from omniscan.translate.prompts import translatable
from omniscan.typeset.plan import plan_layout


class TypesetStage:
    """Fit every final English line into its region's target box (satisfies core.stage.Stage)."""

    name: ClassVar[str] = "typeset"
    version: ClassVar[int] = (
        3  # 2: balloon-shaped lines, style presets, chapter-wide sizes, sfx styles; 3: effects keep clear of neighbours
    )
    gpu_group: ClassVar[str | None] = None

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        """Files whose content determines this stage's output (upstream artifacts only)."""
        inputs = [
            ctx.paths.artifact("ocr.json"),
            ctx.paths.artifact("final.json"),
            ctx.paths.artifact("inpaint.json"),
        ]
        lama = ctx.paths.artifact("inpaint_lama.json")  # which sound effects LaMa erased
        return [*inputs, lama] if lama.is_file() else inputs

    def outputs(self, ctx: ChapterContext) -> list[str]:
        """Artifact names (relative to the chapter work dir) this stage writes."""
        return ["layout.json"]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        return {**cfg.typeset.model_dump(), "sfx": cfg.sfx.model_dump()}

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        ocr_path = ctx.paths.artifact("ocr.json")
        if not ocr_path.is_file():
            raise FileNotFoundError("ocr.json missing — run the ocr stage first")
        final_path = ctx.paths.artifact("final.json")
        if not final_path.is_file():
            raise FileNotFoundError("final.json missing — run the judge stage first")
        inpaint_path = ctx.paths.artifact("inpaint.json")
        if not inpaint_path.is_file():
            raise FileNotFoundError("inpaint.json missing — run the inpaint stage first")

        regions = RegionsArtifact.load(ocr_path).regions
        lines = {line.region_id: line.text for line in FinalArtifact.load(final_path).lines}
        inpaint = InpaintArtifact.load(inpaint_path).items
        fills: dict[str, RGB] = {item.region_id: item.fill for item in inpaint if item.fill is not None}
        erased = {item.region_id for item in inpaint if item.method == "flat"}
        lama_path = ctx.paths.artifact("inpaint_lama.json")
        if lama_path.is_file():
            erased |= {item.region_id for item in InpaintArtifact.load(lama_path).items}
        items = plan_layout(regions, lines, fills, ctx.cfg.typeset, sfx=ctx.cfg.sfx, erased=erased)
        LayoutArtifact(items=items).save(ctx.paths.artifact("layout.json"))
        return {
            "items": float(len(items)),
            "overflow": float(sum(1 for item in items if item.overflow)),
            "skipped": float(len(translatable(regions)) - len(items)),
        }

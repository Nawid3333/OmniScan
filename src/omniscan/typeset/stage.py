"""Typeset stage wrapper: ocr.json + final.json + inpaint(_lama).json (+ edits.json) -> layout.json (no GPU,
no images)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from omniscan.core.config import Config
from omniscan.core.schemas import LayoutArtifact, RegionsArtifact
from omniscan.core.stage import ChapterContext
from omniscan.edits.store import EDITS_FILE
from omniscan.translate.prompts import translatable
from omniscan.typeset.chapter import chapter_layout

LAYOUT_AUTO_FILE = "layout_auto.json"  # layout.json as the typesetter set it, before hand lettering


class TypesetStage:
    """Fit every final English line into its region's target box (satisfies core.stage.Stage).

    The typesetter's own items are kept as layout_auto.json; layout.json is them with the chapter's
    hand-set lettering (edits.json) applied. edits.json is an input: typesetting is cheap, so any hand
    edit simply re-letters the chapter.
    """

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
        for name in ("inpaint_lama.json", EDITS_FILE):  # which sound effects LaMa erased; hand lettering
            path = ctx.paths.artifact(name)
            if path.is_file():
                inputs.append(path)
        return inputs

    def outputs(self, ctx: ChapterContext) -> list[str]:
        """Artifact names (relative to the chapter work dir) this stage writes."""
        return ["layout.json", LAYOUT_AUTO_FILE]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        return {**cfg.typeset.model_dump(), "sfx": cfg.sfx.model_dump()}

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        layout = chapter_layout(ctx.paths, ctx.cfg)
        LayoutArtifact(items=layout.auto).save(ctx.paths.artifact(LAYOUT_AUTO_FILE))
        LayoutArtifact(items=layout.items).save(ctx.paths.artifact("layout.json"))
        regions = RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions
        return {
            "items": float(len(layout.items)),
            "overflow": float(sum(1 for item in layout.items if item.overflow)),
            "skipped": float(len(translatable(regions)) - len(layout.auto)),
            "edits_orphaned": float(layout.orphans),
        }

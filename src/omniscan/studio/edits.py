"""Reading and applying a chapter's studio.json (the Translator Studio's manual changes)."""

from __future__ import annotations

from collections.abc import Sequence

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import Region, StudioEdits

STUDIO_NAME = "studio.json"


def load_edits(paths: ChapterPaths) -> StudioEdits:
    """The chapter's manual changes, or an empty set when the Studio never saved any."""
    path = paths.artifact(STUDIO_NAME)
    return StudioEdits.load(path) if path.is_file() else StudioEdits()


def apply_source_edits(regions: Sequence[Region], edits: StudioEdits) -> list[Region]:
    """Regions with corrected source text applied and removed regions dropped (a re-run OCR keeps the fixes)."""
    removed = set(edits.removed)
    return [
        region.model_copy(update={"text": edits.sources[region.id]}) if region.id in edits.sources else region
        for region in regions
        if region.id not in removed
    ]

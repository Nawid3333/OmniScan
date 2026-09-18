"""Resolve a series' watermark regions into strip-space BBoxes for one chapter's ingest artifact."""

from __future__ import annotations

from collections.abc import Sequence

from omniscan.core.schemas import BBox, IngestArtifact
from omniscan.watermark.store import WatermarkRegion


def resolve_watermark_regions(regions: Sequence[WatermarkRegion], ingest: IngestArtifact) -> list[BBox]:
    """Map every (region, source_file) pair to a strip-space BBox, omitting degenerate results."""
    boxes: list[BBox] = []
    for region in regions:
        for source_file in ingest.files:
            x0 = round(region.x0_frac * source_file.width * source_file.scale)
            x1 = round(region.x1_frac * source_file.width * source_file.scale)
            y0 = source_file.y0 + round(region.y0_frac * (source_file.y1 - source_file.y0))
            y1 = source_file.y0 + round(region.y1_frac * (source_file.y1 - source_file.y0))
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append(BBox(x0=x0, y0=y0, x1=x1, y1=y1))
    return boxes

"""Tier-3 promo filter: reclassify detected regions that overlap a series' stored fixed-position
watermark box (see watermark/store.py, watermark/resolve.py) as kind="watermark", and add a
watermark region for every stored box so the inpaint stage erases it even where nothing was detected."""

from __future__ import annotations

from collections.abc import Sequence

from omniscan.core.schemas import BBox, Region, Slice

_MIN_OVERLAP_IOA = 0.5  # a region needs at least half its own area inside a stored watermark zone

# sfx text is short and a spurious collision there is a real risk (same carve-out as F2b's
# ocr/watermark_text.py), and an already-watermark region needs no copy — only these are reclassified
_RECLASSIFY_KINDS = frozenset({"bubble_text", "free_text"})


def _ioa(region_bbox: BBox, watermark_bbox: BBox) -> float:
    """Intersection area over `region_bbox`'s own area (0.0 when `region_bbox` has zero area)."""
    ix = max(0, min(region_bbox.x1, watermark_bbox.x1) - max(region_bbox.x0, watermark_bbox.x0))
    iy = max(0, min(region_bbox.y1, watermark_bbox.y1) - max(region_bbox.y0, watermark_bbox.y0))
    area = region_bbox.width * region_bbox.height
    return (ix * iy) / area if area else 0.0


def region_overlaps_watermark(region_bbox: BBox, watermark_boxes: Sequence[BBox]) -> bool:
    """True if `region_bbox` has at least `_MIN_OVERLAP_IOA` of its own area inside any one of
    `watermark_boxes` (checked independently per box, not their union)."""
    return any(_ioa(region_bbox, box) >= _MIN_OVERLAP_IOA for box in watermark_boxes)


def reclassify_watermark_position_regions(
    regions: Sequence[Region], watermark_boxes: Sequence[BBox]
) -> list[Region]:
    """Regions overlapping a stored watermark box get kind="watermark" (via model_copy, same shape
    as ocr/watermark_text.py's reclassify_watermark_regions): only "bubble_text"/"free_text" are
    eligible (an "sfx" region is left alone even if it overlaps, same reasoning as F2b: sfx text is
    short and a spurious collision there is a real risk); a region already kind="watermark" stays
    one either way. Untouched regions are returned as the exact same object (identity, not just
    equality — callers may rely on this). Empty `watermark_boxes` -> every region unchanged."""
    return [
        region.model_copy(update={"kind": "watermark"})
        if region.kind in _RECLASSIFY_KINDS and region_overlaps_watermark(region.bbox, watermark_boxes)
        else region
        for region in regions
    ]


def add_watermark_zone_regions(
    regions: Sequence[Region], watermark_boxes: Sequence[BBox], slices: Sequence[Slice]
) -> list[Region]:
    """`regions` plus one kind="watermark" region (no text) per stored watermark box, ids continuing
    after the last region's, each in the slice holding its centre after that slice's other regions; a
    box whose centre lies in a blank or filtered slice (or outside every slice) adds nothing."""
    out = list(regions)
    for box in watermark_boxes:
        centre = (box.y0 + box.y1) / 2
        owner = next((s for s in slices if s.y0 <= centre < s.y1), None)
        if owner is None or owner.blank or owner.filtered:
            continue
        out.append(
            Region(
                id=f"r{len(out) + 1:04d}",
                slice_index=owner.index,
                kind="watermark",
                bbox=box,
                reading_order=sum(1 for r in out if r.slice_index == owner.index),
                confidence=1.0,
            )
        )
    return out

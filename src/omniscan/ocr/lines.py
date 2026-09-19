"""Line-level OCR logic: detector boxes -> merged lines -> assignment to detected regions.

Pure functions over plain floats (no torch): boxes are (x0, y0, x1, y1) in strip pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from omniscan.core.schemas import Region
from omniscan.detect.postprocess import Box, area, ioa, iou, reading_order


@dataclass(frozen=True, slots=True)
class LineBox:
    """One text line the OCR line detector found, in strip pixels."""

    box: Box
    score: float


def polygon_box(points: Sequence[Sequence[float]]) -> Box:
    """Bounding box (x0, y0, x1, y1) of corner points (a [4, 2] detection polygon)."""
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def merge_lines(lines: Sequence[LineBox], *, nms_iou: float, contain_thr: float = 0.8) -> list[LineBox]:
    """NMS across tiles by descending score, then drop lines almost contained in a strictly larger survivor.

    A line cut by a tile edge loses against the whole line seen in the neighbouring tile this way.
    Returns the survivors sorted by (y0, x0).
    """
    pool = sorted(lines, key=lambda line: -line.score)
    kept: list[LineBox] = []
    for line in pool:
        if all(iou(line.box, other.box) <= nms_iou for other in kept):
            kept.append(line)
    kept = [
        line
        for line in kept
        if not any(
            other is not line and area(line.box) < area(other.box) and ioa(line.box, other.box) >= contain_thr
            for other in kept
        )
    ]
    return sorted(kept, key=lambda line: (line.box[1], line.box[0]))


def _padded(region: Region, pad_px: int) -> Box:
    return (
        float(region.bbox.x0 - pad_px),
        float(region.bbox.y0 - pad_px),
        float(region.bbox.x1 + pad_px),
        float(region.bbox.y1 + pad_px),
    )


def assign_lines(
    regions: Sequence[Region],
    lines: Sequence[LineBox],
    *,
    min_ioa: float,
    pad_px: int,
    direction: Literal["ltr", "rtl"],
) -> tuple[dict[str, list[LineBox]], list[LineBox]]:
    """Assign every line to the region whose (padded) text box contains it best; unmatchable lines are orphans.

    Returns (by_region, orphans): `by_region` has an entry for every region id, its lines in reading order;
    `orphans` keeps the input order. Ties go to the smaller region area, then the smaller region id.
    """
    by_region: dict[str, list[LineBox]] = {region.id: [] for region in regions}
    candidates = [
        (
            region,
            _padded(region, pad_px),
            float((region.bbox.x1 - region.bbox.x0) * (region.bbox.y1 - region.bbox.y0)),
        )
        for region in regions
    ]
    orphans: list[LineBox] = []
    for line in lines:
        best: tuple[tuple[float, float, str], Region] | None = None
        for region, padded, region_area in candidates:
            share = ioa(line.box, padded)
            if share < min_ioa:
                continue
            key = (-share, region_area, region.id)
            if best is None or key < best[0]:
                best = (key, region)
        if best is None:
            orphans.append(line)
        else:
            by_region[best[1].id].append(line)
    for region_id, member in by_region.items():
        boxes = [line.box for line in member]
        by_region[region_id] = [member[i] for i in reading_order(boxes, direction)]
    return by_region, orphans

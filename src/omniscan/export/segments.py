"""Which strip rows become which output image: the slices, or the output cuts set by hand (pure, no torch)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from omniscan.core.schemas import BBox, Region, Slice


@dataclass(frozen=True, slots=True)
class Segment:
    """Strip rows [y0, y1) written as one output image; `slice_index` = the slice holding row y0."""

    y0: int
    y1: int
    slice_index: int


def _slice_at(slices: Sequence[Slice], y: int) -> int:
    """Index of the slice holding row `y` (the last slice for rows past the end)."""
    for s in slices:
        if s.y0 <= y < s.y1:
            return s.index
    return slices[-1].index if slices else 0


def _split(y0: int, y1: int, max_height: int | None) -> list[tuple[int, int]]:
    """Rows [y0, y1) as equal parts no taller than `max_height` (one part when it fits, or without a limit)."""
    parts = 1 if max_height is None else max(1, math.ceil((y1 - y0) / max_height))
    bounds = [y0 + (y1 - y0) * i // parts for i in range(parts + 1)]
    return list(pairwise(bounds))


def output_segments(
    slices: Sequence[Slice], strip_height: int, cuts: Sequence[int] | None, max_height: int | None = None
) -> list[Segment]:
    """The output images of a chapter in order: one per slice that is not filtered, or — with hand cuts —
    the pieces between the cuts, with the rows of filtered slices taken out (a piece a filtered slice
    splits becomes two). A hand piece taller than `max_height` is split into equal parts no taller than it:
    a JPEG holds at most 65,535 rows, and the slicer's own slices already respect the limit."""
    if cuts is None:
        return [Segment(s.y0, s.y1, s.index) for s in slices if not s.filtered]
    bounds = [0, *sorted({c for c in cuts if 0 < c < strip_height}), strip_height]
    removed = sorted((s.y0, s.y1) for s in slices if s.filtered)
    segments: list[Segment] = []
    for top, bottom in pairwise(bounds):
        pieces = [(top, bottom)]
        for f0, f1 in removed:
            pieces = [
                part
                for y0, y1 in pieces
                for part in ((y0, min(y1, f0)), (max(y0, f1), y1))
                if part[0] < part[1]
            ]
        segments.extend(
            Segment(a, b, _slice_at(slices, a)) for y0, y1 in pieces for a, b in _split(y0, y1, max_height)
        )
    return segments


def cut_crossings(cuts: Sequence[int], regions: Sequence[Region]) -> list[tuple[int, str]]:
    """(cut, region id) for every cut that runs through a region's text box or its bubble — a cut there
    splits the lettering across two images."""
    crossings: list[tuple[int, str]] = []
    for cut in cuts:
        for region in regions:
            boxes: list[BBox] = [region.bbox, *([region.bubble_bbox] if region.bubble_bbox else [])]
            if any(box.y0 < cut < box.y1 for box in boxes):
                crossings.append((cut, region.id))
    return crossings

"""Turn raw tile detections into chapter regions: merge across tiles, pair text with bubbles, order for reading.

Pure functions over plain floats (no torch): boxes are (x0, y0, x1, y1) in strip pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from omniscan.core.schemas import BBox, Region, Slice
from omniscan.detect.tiles import Tile

DetClass = Literal["bubble", "text_bubble", "text_free"]
Box = tuple[float, float, float, float]

_TEXT_IN_BUBBLE_MIN_IOA = 0.6  # a text box is inside a bubble when this share of it lies within the bubble
_MAX_BUBBLE_TO_TEXT_AREA_RATIO = (
    8.0  # a bubble more than this much bigger than its own text is not a real fit
)
_CROSS_CLASS_DUP_IOU = 0.6  # text_free and text_bubble boxes this similar are the same text
_CROSS_CLASS_CONTAIN_IOA = (
    0.85  # a box this much inside the other (either direction) is the same text, even if IoU reads low
)
_SAME_ROW_OVERLAP = 0.5  # vertical overlap (of the smaller box) that puts two regions in one reading row
_EDGE_PX = 4.0  # a box this close to an internal tile edge counts as cut by it


@dataclass(frozen=True, slots=True)
class Det:
    """One detection in strip pixels; `edge` marks a box cut by an internal tile edge (likely truncated)."""

    cls: DetClass
    score: float
    box: Box
    edge: bool = False


def area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def ioa(inner: Box, outer: Box) -> float:
    """Share of `inner`'s area that lies inside `outer`."""
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def tile_det_to_strip(tile: Tile, cls: DetClass, score: float, box: Box) -> Det:
    """Move a detection from tile pixels to strip pixels and flag it if an internal tile edge cut it."""
    x0, y0, x1, y1 = box
    edge = (
        (tile.left and x0 <= _EDGE_PX)
        or (tile.top and y0 <= _EDGE_PX)
        or (tile.right and x1 >= tile.width - _EDGE_PX)
        or (tile.bottom and y1 >= tile.height - _EDGE_PX)
    )
    return Det(cls, score, (x0 + tile.x0, y0 + tile.y0, x1 + tile.x0, y1 + tile.y0), edge)


def merge_detections(
    dets: Sequence[Det], *, nms_iou: float, contain_thr: float, edge_penalty: float
) -> list[Det]:
    """Class-wise NMS across tiles, preferring whole boxes over ones cut by a tile edge.

    Edge boxes are penalised before NMS; afterwards an edge box that lies mostly inside another same-class box
    is dropped as a truncated duplicate. Returns the survivors, highest score first within each class.
    """
    merged: list[Det] = []
    for cls in ("bubble", "text_bubble", "text_free"):
        pool = sorted(
            (replace(d, score=d.score - edge_penalty) if d.edge else d for d in dets if d.cls == cls),
            key=lambda d: -d.score,
        )
        kept: list[Det] = []
        for det in pool:
            if all(iou(det.box, other.box) <= nms_iou for other in kept):
                kept.append(det)
        kept = [
            det
            for det in kept
            if not (
                det.edge
                and any(other is not det and ioa(det.box, other.box) >= contain_thr for other in kept)
            )
        ]
        merged.extend(kept)
    return merged


@dataclass(frozen=True, slots=True)
class _Item:
    kind: Literal["bubble_text", "free_text"]
    box: Box
    bubble: Box | None
    score: float


def _union(boxes: Sequence[Box]) -> Box:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _items(dets: Sequence[Det], merge_bubble_text: bool) -> list[_Item]:
    """Pair text boxes with their bubble; drop cross-class duplicates; merge text boxes sharing a bubble."""
    bubbles = [d for d in dets if d.cls == "bubble"]
    texts_bubble = [d for d in dets if d.cls == "text_bubble"]
    texts_free = list(d for d in dets if d.cls == "text_free")

    dropped_free: set[int] = set()
    dropped_bubble: set[int] = set()
    for fi, free in enumerate(texts_free):
        for bi, tb in enumerate(texts_bubble):
            duplicate = (
                iou(free.box, tb.box) >= _CROSS_CLASS_DUP_IOU
                or ioa(free.box, tb.box) >= _CROSS_CLASS_CONTAIN_IOA
                or ioa(tb.box, free.box) >= _CROSS_CLASS_CONTAIN_IOA
            )
            if bi in dropped_bubble or not duplicate:
                continue
            if free.score > tb.score:
                dropped_bubble.add(bi)
            else:
                dropped_free.add(fi)
                break
    texts_bubble = [d for i, d in enumerate(texts_bubble) if i not in dropped_bubble]
    texts_free = [d for i, d in enumerate(texts_free) if i not in dropped_free]

    items: list[_Item] = []
    grouped: dict[int, list[Det]] = {}
    for tb in texts_bubble:
        candidates = [
            (i, b)
            for i, b in enumerate(bubbles)
            if ioa(tb.box, b.box) >= _TEXT_IN_BUBBLE_MIN_IOA
            and area(b.box) <= _MAX_BUBBLE_TO_TEXT_AREA_RATIO * area(tb.box)
        ]
        if not candidates:
            items.append(_Item("bubble_text", tb.box, None, tb.score))
            continue
        index = min(candidates, key=lambda c: area(c[1].box))[0]
        if merge_bubble_text:
            grouped.setdefault(index, []).append(tb)
        else:
            items.append(_Item("bubble_text", tb.box, bubbles[index].box, tb.score))
    for index, members in grouped.items():
        items.append(
            _Item(
                "bubble_text",
                _union([m.box for m in members]),
                bubbles[index].box,
                max(m.score for m in members),
            )
        )
    items.extend(_Item("free_text", d.box, None, d.score) for d in texts_free)
    return items


def reading_order(boxes: Sequence[Box], direction: Literal["ltr", "rtl"]) -> list[int]:
    """Indices of `boxes` in reading order: rows top to bottom, each row left-to-right (or right-to-left).

    A box joins the current row when it overlaps the row's topmost box vertically by at least half of the
    smaller of the two heights; otherwise it starts a new row.
    """
    remaining = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
    rows: list[list[int]] = []
    for i in remaining:
        if rows:
            anchor = boxes[rows[-1][0]]
            top = max(anchor[1], boxes[i][1])
            bottom = min(anchor[3], boxes[i][3])
            smaller = min(anchor[3] - anchor[1], boxes[i][3] - boxes[i][1])
            if smaller > 0 and (bottom - top) / smaller >= _SAME_ROW_OVERLAP:
                rows[-1].append(i)
                continue
        rows.append([i])
    order: list[int] = []
    for row in rows:
        row.sort(key=(lambda i: boxes[i][0]) if direction == "ltr" else (lambda i: -boxes[i][2]))
        order.extend(row)
    return order


def _slice_of(slices: Sequence[Slice], y: float) -> Slice | None:
    for s in slices:
        if s.y0 <= y < s.y1:
            return s
    return slices[-1] if slices and y >= slices[-1].y1 else None


def _bbox(box: Box, width: int, height: int) -> BBox | None:
    x0, y0 = max(0, int(box[0] // 1)), max(0, int(box[1] // 1))
    x1 = min(width, int(-(-box[2] // 1)))
    y1 = min(height, int(-(-box[3] // 1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def build_regions(
    dets: Sequence[Det],
    slices: Sequence[Slice],
    *,
    strip_width: int,
    strip_height: int,
    merge_bubble_text: bool,
    direction: Literal["ltr", "rtl"],
) -> list[Region]:
    """Regions (text empty) for the merged detections, ids r0001.. in (slice, reading order).

    A region belongs to the slice containing its box centre; regions in blank or filtered slices are dropped.
    """
    per_slice: dict[int, list[tuple[BBox, BBox | None, _Item]]] = {}
    for item in _items(dets, merge_bubble_text):
        box = _bbox(item.box, strip_width, strip_height)
        if box is None:
            continue
        owner = _slice_of(slices, (box.y0 + box.y1) / 2)
        if owner is None or owner.blank or owner.filtered:
            continue
        bubble = _bbox(item.bubble, strip_width, strip_height) if item.bubble is not None else None
        per_slice.setdefault(owner.index, []).append((box, bubble, item))

    regions: list[Region] = []
    for slice_index in sorted(per_slice):
        entries = per_slice[slice_index]
        boxes = [(float(b.x0), float(b.y0), float(b.x1), float(b.y1)) for b, _, _ in entries]
        for rank, i in enumerate(reading_order(boxes, direction)):
            box, bubble, item = entries[i]
            regions.append(
                Region(
                    id=f"r{len(regions) + 1:04d}",
                    slice_index=slice_index,
                    kind=item.kind,
                    bbox=box,
                    bubble_bbox=bubble,
                    reading_order=rank,
                    confidence=round(item.score, 4),
                )
            )
    return regions

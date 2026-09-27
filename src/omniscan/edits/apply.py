"""Hand edits (edits.json) applied to what the pipeline produces: regions after `ocr`, lines after `judge`.

Pure functions over the core schemas (no torch, no I/O). An edit is matched to its region by id and box
overlap, so it follows the region through a re-run that renumbered the regions (see `RegionEdit`).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Literal

from omniscan.core.schemas import (
    BBox,
    ChapterEdits,
    FinalLine,
    OcrLine,
    Region,
    RegionEdit,
    Slice,
    TranslationEdit,
)
from omniscan.detect.postprocess import reading_order
from omniscan.translate.prompts import source_text

MATCH_IOU = 0.5  # an edit applies to a region whose box overlaps the edit's boxes at least this much
MANUAL_ENGINE = "manual"  # OcrLine.engine of a line drawn or typed by hand
SOURCE_CHANGED = "source_changed"  # FinalLine flag: the source text changed after the line was written
ADDED_PREFIX = "m"  # ids of hand-added regions (m0001, …) never collide with the detector's r0001, …


def edit_boxes(edit: RegionEdit | TranslationEdit) -> list[BBox]:
    """The boxes an edit is matched on: its anchor, plus the box it moved the region to."""
    if isinstance(edit, RegionEdit) and edit.bbox is not None:
        return [edit.anchor, edit.bbox]
    return [edit.anchor]


def _overlap(region: Region, boxes: Sequence[BBox]) -> float:
    """Best IoU between the region's box and any of `boxes`."""
    return max(region.bbox.iou(box) for box in boxes)


def match_region(
    region_id: str, boxes: Sequence[BBox], regions: Sequence[Region], taken: Collection[str] = ()
) -> Region | None:
    """The region an edit of (`region_id`, `boxes`) applies to: the one with that id when it still overlaps
    the boxes by MATCH_IOU, else the best-overlapping one not in `taken` (at least MATCH_IOU); else None."""
    free = [region for region in regions if region.id not in taken]
    same = next((region for region in free if region.id == region_id), None)
    if same is not None and _overlap(same, boxes) >= MATCH_IOU:
        return same
    best = max(free, key=lambda region: _overlap(region, boxes), default=None)
    if best is None or _overlap(best, boxes) < MATCH_IOU:
        return None
    return best


def edited_region(region: Region, edit: RegionEdit) -> Region:
    """`region` with the edit's fields applied. A new box replaces the OCR lines by one hand-drawn line of
    that box (so inpaint cleans the whole new area); a new text is trusted (confidence 1)."""
    update: dict[str, object] = {}
    text = region.text if edit.text is None else edit.text
    if edit.kind is not None:
        update["kind"] = edit.kind
    if edit.bubble_bbox is not None:
        update["bubble_bbox"] = edit.bubble_bbox
    if edit.bbox is not None:
        update["bbox"] = edit.bbox
        update["lines"] = [OcrLine(bbox=edit.bbox, text=text, score=1.0, engine=MANUAL_ENGINE)]
    if edit.text is not None:
        update["text"] = edit.text
        update["confidence"] = 1.0
        update["ocr_alt"] = None
    return region.model_copy(update=update)


def slice_at(slices: Sequence[Slice], y: float) -> Slice | None:
    """The slice whose rows contain `y` (the last slice below the strip's end); None without slices."""
    for s in slices:
        if s.y0 <= y < s.y1:
            return s
    return slices[-1] if slices else None


def _centre_y(box: BBox) -> float:
    return (box.y0 + box.y1) / 2


def added_region(edit: RegionEdit, slices: Sequence[Slice]) -> Region | None:
    """The region a hand-added edit describes, in the slice holding its centre; None without slices."""
    box = edit.bbox if edit.bbox is not None else edit.anchor
    owner = slice_at(slices, _centre_y(box))
    if owner is None:
        return None
    text = edit.text or ""
    return Region(
        id=edit.region_id,
        slice_index=owner.index,
        kind=edit.kind or "bubble_text",
        bbox=box,
        bubble_bbox=edit.bubble_bbox,
        lang=edit.lang or "ko",
        lines=[OcrLine(bbox=box, text=text, score=1.0, engine=MANUAL_ENGINE)],
        text=text,
        confidence=1.0,
    )


def reorder(
    regions: Sequence[Region], slice_indices: Collection[int], direction: Literal["ltr", "rtl"]
) -> list[Region]:
    """`regions` with `reading_order` recomputed (as `detect` does) inside the given slices."""
    ranks: dict[str, int] = {}
    for index in slice_indices:
        members = [region for region in regions if region.slice_index == index]
        boxes = [(float(r.bbox.x0), float(r.bbox.y0), float(r.bbox.x1), float(r.bbox.y1)) for r in members]
        for rank, i in enumerate(reading_order(boxes, direction)):
            ranks[members[i].id] = rank
    return [
        region.model_copy(update={"reading_order": ranks[region.id]}) if region.id in ranks else region
        for region in regions
    ]


def match_region_edits(regions: Sequence[Region], edits: ChapterEdits) -> dict[int, str]:
    """Index in `edits.regions` -> id of the pipeline region that edit applies to, in edit order (each region
    claimed once); added edits and edits whose region no longer exists are absent."""
    taken: set[str] = set()
    claims: dict[int, str] = {}
    for i, edit in enumerate(edits.regions):
        if edit.added:
            continue
        region = match_region(edit.region_id, edit_boxes(edit), regions, taken)
        if region is not None:
            taken.add(region.id)
            claims[i] = region.id
    return claims


def apply_region_edits(
    regions: Sequence[Region],
    edits: ChapterEdits,
    slices: Sequence[Slice],
    *,
    direction: Literal["ltr", "rtl"] = "ltr",
) -> tuple[list[Region], int]:
    """The regions with every region edit applied, and the number of edits whose region no longer exists.

    Added regions are inserted, and a pipeline region they cover (IoU >= MATCH_IOU) that no other edit
    claims is dropped: the hand-drawn box wins. A region moved into another slice changes slice. Reading
    order is recomputed only in the slices that gained or moved a region (a removal keeps the others' order).
    """
    claims = match_region_edits(regions, edits)
    edit_of = {region_id: edits.regions[i] for i, region_id in claims.items()}
    orphans = sum(1 for i, edit in enumerate(edits.regions) if not edit.added and i not in claims)
    added = [
        region
        for edit in edits.regions
        if edit.added and not edit.deleted
        for region in (added_region(edit, slices),)
        if region is not None
    ]
    touched: set[int] = {region.slice_index for region in added}
    result: list[Region] = []
    for region in regions:
        edit = edit_of.get(region.id)
        if edit is not None:
            if edit.deleted:
                continue
            edited = edited_region(region, edit)
            if edited.bbox != region.bbox:
                owner = slice_at(slices, _centre_y(edited.bbox))
                if owner is not None and owner.index != edited.slice_index:
                    edited = edited.model_copy(update={"slice_index": owner.index})
                touched.add(edited.slice_index)
            result.append(edited)
        elif not any(region.bbox.iou(new.bbox) >= MATCH_IOU for new in added):
            result.append(region)
    result.extend(added)
    if not touched:
        return result, orphans
    return reorder(result, touched, direction), orphans


def match_translation_edits(regions: Sequence[Region], edits: ChapterEdits) -> dict[int, str]:
    """Index in `edits.translations` -> id of the region that line belongs to (each region claimed once);
    lines whose region no longer exists are absent."""
    taken: set[str] = set()
    claims: dict[int, str] = {}
    for i, edit in enumerate(edits.translations):
        region = match_region(edit.region_id, edit_boxes(edit), regions, taken)
        if region is not None:
            taken.add(region.id)
            claims[i] = region.id
    return claims


def manual_line(edit: TranslationEdit, region: Region) -> FinalLine:
    """The final line a translation edit stands for, flagged when the region's source text has changed."""
    flags = [] if source_text(region) == " ".join(edit.source.split()) else [SOURCE_CHANGED]
    return FinalLine(
        region_id=region.id,
        text=edit.text,
        decision="manual",
        sources=[],
        rationale="edited by hand",
        flags=flags,
    )


def apply_translation_edits(
    lines: Sequence[FinalLine], edits: ChapterEdits, regions: Sequence[Region]
) -> tuple[list[FinalLine], int]:
    """The final lines with every translation edit applied (replacing the region's line, else appended),
    and the number of edits whose region no longer exists."""
    by_id = {region.id: region for region in regions}
    claims = match_translation_edits(regions, edits)
    manual = {
        region_id: manual_line(edits.translations[i], by_id[region_id]) for i, region_id in claims.items()
    }
    result = [manual.pop(line.region_id, line) for line in lines]
    return [*result, *manual.values()], len(edits.translations) - len(claims)

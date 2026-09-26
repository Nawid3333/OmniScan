"""edits.json persistence and the edit operations every editing tool shares (web studio, CLI).

An operation records the edit in edits.json, then rebuilds the chapter's ocr.json from the pipeline's own
reading (`ocr_auto.json`) and final.json from the judge's (`final_auto.json`) with every edit applied — the
same functions the `ocr` and `judge` stages apply when they re-run, so a tool and the pipeline never
disagree. A chapter processed before those files existed gets them from its ocr.json/final.json on its
first edit; a chapter with no OCR yet starts from no regions (everything drawn by hand).
"""

from __future__ import annotations

import shutil
import threading
from typing import Literal

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import (
    BBox,
    ChapterEdits,
    FinalArtifact,
    FinalLine,
    Lang,
    Region,
    RegionEdit,
    RegionKind,
    RegionsArtifact,
    SlicesArtifact,
    TranslationEdit,
)
from omniscan.edits.apply import (
    ADDED_PREFIX,
    apply_region_edits,
    apply_translation_edits,
    match_region_edits,
    match_translation_edits,
)

EDITS_FILE = "edits.json"
OCR_AUTO_FILE = "ocr_auto.json"  # ocr.json as the `ocr` stage read it, before hand edits
FINAL_AUTO_FILE = "final_auto.json"  # final.json as the `judge` stage wrote it, before hand edits
MANUAL_JUDGE = "manual"  # FinalArtifact.judge_model of a final.json no judge has written
MIN_BOX_PX = 2  # a hand-drawn box narrower or lower than this (after clamping to the strip) is refused

Direction = Literal["ltr", "rtl"]

# One edit at a time in this process: the web server runs its handlers on a thread pool, and every
# operation is a read-modify-write of edits.json, ocr.json and final.json.
_LOCK = threading.RLock()


class EditNotFoundError(LookupError):
    """The chapter lacks what an edit needs (an artifact, a region); the message says what."""


def load_edits(paths: ChapterPaths) -> ChapterEdits:
    """The chapter's edits.json, or no edits when it does not exist."""
    path = paths.artifact(EDITS_FILE)
    return ChapterEdits.load(path) if path.is_file() else ChapterEdits()


def save_edits(paths: ChapterPaths, edits: ChapterEdits) -> None:
    """Atomically write the chapter's edits.json."""
    edits.save(paths.artifact(EDITS_FILE))


def load_slices(paths: ChapterPaths) -> SlicesArtifact:
    """The chapter's slices.json (an edit needs the strip's size and slices)."""
    path = paths.artifact("slices.json")
    if not path.is_file():
        raise EditNotFoundError("slices.json not found — run the slice stage first")
    return SlicesArtifact.load(path)


def _ensure_auto(paths: ChapterPaths) -> None:
    """Create ocr_auto.json (from a pre-edit ocr.json, else empty) and final_auto.json (from a pre-edit
    final.json) when missing, before the first edit rewrites ocr.json/final.json."""
    ocr_auto = paths.artifact(OCR_AUTO_FILE)
    if not ocr_auto.is_file():
        current = paths.artifact("ocr.json")
        if current.is_file():
            shutil.copyfile(current, ocr_auto)
        else:
            RegionsArtifact(regions=[]).save(ocr_auto)
    final_auto = paths.artifact(FINAL_AUTO_FILE)
    final = paths.artifact("final.json")
    if not final_auto.is_file() and final.is_file():
        shutil.copyfile(final, final_auto)


def auto_regions(paths: ChapterPaths) -> list[Region]:
    """The pipeline's own regions (ocr_auto.json; none when the chapter has no OCR yet)."""
    path = paths.artifact(OCR_AUTO_FILE)
    return RegionsArtifact.load(path).regions if path.is_file() else []


def current_regions(paths: ChapterPaths) -> list[Region]:
    """The chapter's regions as the rest of the pipeline sees them (ocr.json; none when missing)."""
    path = paths.artifact("ocr.json")
    return RegionsArtifact.load(path).regions if path.is_file() else []


def rebuild(paths: ChapterPaths, edits: ChapterEdits, *, direction: Direction) -> list[Region]:
    """Rewrite ocr.json — and final.json when a judge ran or a line was written by hand — from the pipeline's
    output with every edit applied; returns the regions written."""
    slices = load_slices(paths).slices if edits.regions else []  # only added/moved regions need slices
    regions, _ = apply_region_edits(auto_regions(paths), edits, slices, direction=direction)
    if regions != current_regions(paths) or not paths.artifact("ocr.json").is_file():
        # an unchanged ocr.json keeps its bytes: its hash is the translate stage's input, and rewriting it
        # for an English-only edit would make the next run re-translate the whole chapter
        RegionsArtifact(regions=regions).save(paths.artifact("ocr.json"))
    final_auto = paths.artifact(FINAL_AUTO_FILE)
    if final_auto.is_file():
        base = FinalArtifact.load(final_auto)
    elif edits.translations:
        base = FinalArtifact(judge_model=MANUAL_JUDGE, lines=[])
        base.save(final_auto)  # so a later edit never mistakes the hand-written final.json for a judge's
    else:
        return regions
    lines, _ = apply_translation_edits(base.lines, edits, regions)
    base.model_copy(update={"lines": lines}).save(paths.artifact("final.json"))
    return regions


def _find(regions: list[Region], region_id: str) -> Region:
    """The region with `region_id` (EditNotFoundError when there is none)."""
    region = next((region for region in regions if region.id == region_id), None)
    if region is None:
        raise EditNotFoundError(f"region {region_id!r} not found in ocr.json")
    return region


def clamp_box(box: BBox, slices: SlicesArtifact) -> BBox:
    """`box` clamped into the strip; ValueError when less than MIN_BOX_PX remains on either axis."""
    clamped = BBox(
        x0=min(max(box.x0, 0), slices.strip_width),
        y0=min(max(box.y0, 0), slices.strip_height),
        x1=min(max(box.x1, 0), slices.strip_width),
        y1=min(max(box.y1, 0), slices.strip_height),
    )
    if clamped.width < MIN_BOX_PX or clamped.height < MIN_BOX_PX:
        raise ValueError(f"box {box.x0},{box.y0},{box.x1},{box.y1} is empty inside the strip")
    return clamped


def _region_edit_of(paths: ChapterPaths, edits: ChapterEdits, region_id: str) -> int | None:
    """Index in `edits.regions` of the edit that applies to the region shown as `region_id`, or None."""
    for i, edit in enumerate(edits.regions):
        if edit.added and edit.region_id == region_id:
            return i
    claims = match_region_edits(auto_regions(paths), edits)
    return next((i for i, claimed in claims.items() if claimed == region_id), None)


def _translation_edit_of(paths: ChapterPaths, edits: ChapterEdits, region_id: str) -> int | None:
    """Index in `edits.translations` of the hand-written line of the region shown as `region_id`, or None."""
    claims = match_translation_edits(current_regions(paths), edits)
    return next((i for i, claimed in claims.items() if claimed == region_id), None)


def _reanchor(edits: ChapterEdits, line_index: int | None, box: BBox) -> None:
    """Move a hand-written line's anchor to its region's new box, so the line follows a moved region."""
    if line_index is not None:
        edits.translations[line_index] = edits.translations[line_index].model_copy(update={"anchor": box})


def update_region(
    paths: ChapterPaths,
    region_id: str,
    *,
    direction: Direction,
    kind: RegionKind | None = None,
    bbox: BBox | None = None,
    bubble_bbox: BBox | None = None,
    text: str | None = None,
) -> Region:
    """Change a region's kind, text box, bubble box and/or source text; returns the region as rebuilt."""
    with _LOCK:
        _ensure_auto(paths)
        region = _find(current_regions(paths), region_id)
        slices = load_slices(paths)
        new_box = None if bbox is None else clamp_box(bbox, slices)
        changes: dict[str, object] = {}
        if kind is not None:
            changes["kind"] = kind
        if new_box is not None:
            changes["bbox"] = new_box
        if bubble_bbox is not None:
            changes["bubble_bbox"] = clamp_box(bubble_bbox, slices)
        if text is not None:
            changes["text"] = text
        edits = load_edits(paths)
        line_index = _translation_edit_of(paths, edits, region.id)
        index = _region_edit_of(paths, edits, region.id)
        if index is None:
            edits.regions.append(
                RegionEdit(region_id=region.id, anchor=region.bbox).model_copy(update=changes)
            )
        else:
            edits.regions[index] = edits.regions[index].model_copy(update=changes)
        if new_box is not None:
            _reanchor(edits, line_index, new_box)
        save_edits(paths, edits)
        return _find(rebuild(paths, edits, direction=direction), region.id)


def next_added_id(edits: ChapterEdits) -> str:
    """The next free id for a hand-added region (m0001, m0002, …)."""
    used = {edit.region_id for edit in edits.regions if edit.added}
    number = 1
    while f"{ADDED_PREFIX}{number:04d}" in used:
        number += 1
    return f"{ADDED_PREFIX}{number:04d}"


def add_region(
    paths: ChapterPaths,
    bbox: BBox,
    *,
    direction: Direction,
    kind: RegionKind = "bubble_text",
    text: str = "",
    bubble_bbox: BBox | None = None,
    lang: Lang = "ko",
) -> Region:
    """Add a hand-drawn region (a text area the detector missed, or one to clean); returns it as rebuilt."""
    with _LOCK:
        _ensure_auto(paths)
        slices = load_slices(paths)
        edits = load_edits(paths)
        region_id = next_added_id(edits)
        edits.regions.append(
            RegionEdit(
                region_id=region_id,
                anchor=clamp_box(bbox, slices),
                added=True,
                kind=kind,
                text=text,
                bubble_bbox=None if bubble_bbox is None else clamp_box(bubble_bbox, slices),
                lang=lang,
            )
        )
        save_edits(paths, edits)
        return _find(rebuild(paths, edits, direction=direction), region_id)


def delete_region(paths: ChapterPaths, region_id: str, *, direction: Direction) -> None:
    """Remove a region: a hand-added one is forgotten, a pipeline one is dropped (its text stays on the page,
    untranslated — a false detection) until the deletion is reverted."""
    with _LOCK:
        _ensure_auto(paths)
        region = _find(current_regions(paths), region_id)
        edits = load_edits(paths)
        index = _region_edit_of(paths, edits, region.id)
        if index is not None and edits.regions[index].added:
            del edits.regions[index]
        elif index is None:
            edits.regions.append(RegionEdit(region_id=region.id, anchor=region.bbox, deleted=True))
        else:
            edits.regions[index] = edits.regions[index].model_copy(update={"deleted": True})
        save_edits(paths, edits)
        rebuild(paths, edits, direction=direction)


def revert_region(paths: ChapterPaths, region_id: str, *, direction: Direction) -> Region | None:
    """Drop every hand edit of a region (a deleted one comes back); returns it as the pipeline read it, or
    None for a hand-added region (which is removed)."""
    with _LOCK:
        _ensure_auto(paths)
        edits = load_edits(paths)
        index = _region_edit_of(paths, edits, region_id)
        if index is None:
            raise EditNotFoundError(f"region {region_id!r} has no edit to revert")
        line_index = _translation_edit_of(paths, edits, region_id)
        del edits.regions[index]
        restored = next((region for region in auto_regions(paths) if region.id == region_id), None)
        if restored is not None:
            _reanchor(edits, line_index, restored.bbox)
        save_edits(paths, edits)
        regions = rebuild(paths, edits, direction=direction)
        return next((region for region in regions if region.id == region_id), None)


def deleted_regions(paths: ChapterPaths) -> list[Region]:
    """The pipeline regions a hand edit deleted, as the pipeline read them (so a tool can offer to restore)."""
    edits = load_edits(paths)
    auto = {region.id: region for region in auto_regions(paths)}
    claims = match_region_edits(list(auto.values()), edits)
    return [auto[region_id] for i, region_id in claims.items() if edits.regions[i].deleted]


def edited_ids(paths: ChapterPaths) -> tuple[list[str], list[str]]:
    """Ids of the current regions (ocr.json) that carry a region edit, and of those with a hand-written line."""
    edits = load_edits(paths)
    claims = match_region_edits(auto_regions(paths), edits)
    edited = {region_id for i, region_id in claims.items() if not edits.regions[i].deleted}
    edited |= {edit.region_id for edit in edits.regions if edit.added and not edit.deleted}
    current = current_regions(paths)
    translated = set(match_translation_edits(current, edits).values())
    return (
        [region.id for region in current if region.id in edited],
        [region.id for region in current if region.id in translated],
    )


def set_translation(
    paths: ChapterPaths,
    region_id: str,
    text: str,
    *,
    direction: Direction,
    suggested_by: str | None = None,
) -> FinalLine:
    """Write a region's English line by hand (or keep a profile's suggestion, `suggested_by`); it replaces
    the judge's line on every later judge run."""
    with _LOCK:
        _ensure_auto(paths)
        region = _find(current_regions(paths), region_id)
        edits = load_edits(paths)
        edit = TranslationEdit(
            region_id=region.id, anchor=region.bbox, text=text, source=region.text, suggested_by=suggested_by
        )
        index = _translation_edit_of(paths, edits, region.id)
        if index is None:
            edits.translations.append(edit)
        else:
            edits.translations[index] = edit
        save_edits(paths, edits)
        rebuild(paths, edits, direction=direction)
        final = FinalArtifact.load(paths.artifact("final.json"))
        return next(line for line in final.lines if line.region_id == region.id)


def revert_translation(paths: ChapterPaths, region_id: str, *, direction: Direction) -> FinalLine | None:
    """Drop a region's hand-written line; returns the judge's line again, or None when it has none."""
    with _LOCK:
        _ensure_auto(paths)
        edits = load_edits(paths)
        index = _translation_edit_of(paths, edits, region_id)
        if index is None:
            raise EditNotFoundError(f"region {region_id!r} has no hand-written line to revert")
        del edits.translations[index]
        save_edits(paths, edits)
        rebuild(paths, edits, direction=direction)
        final_path = paths.artifact("final.json")
        if not final_path.is_file():
            return None
        lines = FinalArtifact.load(final_path).lines
        return next((line for line in lines if line.region_id == region_id), None)

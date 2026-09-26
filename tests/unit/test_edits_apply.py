"""Tests for omniscan.edits.apply — hand edits re-applied to pipeline output (pure functions)."""

from __future__ import annotations

from omniscan.core.schemas import (
    BBox,
    ChapterEdits,
    FinalLine,
    OcrLine,
    Region,
    RegionEdit,
    RegionKind,
    Slice,
    TranslationEdit,
)
from omniscan.edits.apply import (
    MANUAL_ENGINE,
    SOURCE_CHANGED,
    apply_region_edits,
    apply_translation_edits,
    match_region,
    match_region_edits,
)

SLICES = [Slice(index=0, y0=0, y1=500), Slice(index=1, y0=500, y1=1000)]


def box(x0: int, y0: int, x1: int, y1: int) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def region(
    rid: str, bbox: BBox, text: str = "", *, kind: RegionKind = "bubble_text", order: int = 0
) -> Region:
    slice_index = 0 if bbox.y0 < 500 else 1
    return Region(
        id=rid,
        slice_index=slice_index,
        kind=kind,
        bbox=bbox,
        reading_order=order,
        lines=[OcrLine(bbox=bbox, text=text, score=0.8, engine="ocr")],
        text=text,
        confidence=0.8,
        ocr_alt="alt",
    )


A = region("r0001", box(10, 10, 110, 60), "안녕", order=0)
B = region("r0002", box(10, 200, 110, 260), "반가워", order=1)
C = region("r0003", box(10, 600, 110, 660), "잘가", order=0)
AUTO = [A, B, C]


def test_no_edits_returns_regions_unchanged() -> None:
    assert apply_region_edits(AUTO, ChapterEdits(), SLICES) == (AUTO, 0)


def test_text_fix_keeps_lines_and_order_and_trusts_the_text() -> None:
    edits = ChapterEdits(regions=[RegionEdit(region_id="r0002", anchor=B.bbox, text="반가워요")])
    regions, orphans = apply_region_edits(AUTO, edits, SLICES)
    assert orphans == 0
    fixed = regions[1]
    assert (fixed.id, fixed.text, fixed.confidence, fixed.ocr_alt) == ("r0002", "반가워요", 1.0, None)
    assert fixed.lines == B.lines  # the cleaning boxes stay the OCR's
    assert [r.reading_order for r in regions] == [0, 1, 0]


def test_moved_box_replaces_lines_changes_slice_and_reorders() -> None:
    moved = box(10, 700, 110, 760)  # from slice 0 into slice 1, below r0003
    edits = ChapterEdits(regions=[RegionEdit(region_id="r0001", anchor=A.bbox, bbox=moved)])
    regions, _ = apply_region_edits(AUTO, edits, SLICES)
    edited = next(r for r in regions if r.id == "r0001")
    assert edited.bbox == moved and edited.slice_index == 1
    assert edited.lines == [OcrLine(bbox=moved, text="안녕", score=1.0, engine=MANUAL_ENGINE)]
    by_id = {r.id: r.reading_order for r in regions}
    assert by_id["r0003"] == 0 and by_id["r0001"] == 1  # slice 1 re-ordered top to bottom


def test_kind_and_bubble_edits() -> None:
    bubble = box(0, 0, 130, 80)
    edits = ChapterEdits(
        regions=[RegionEdit(region_id="r0001", anchor=A.bbox, kind="watermark", bubble_bbox=bubble)]
    )
    edited = apply_region_edits(AUTO, edits, SLICES)[0][0]
    assert edited.kind == "watermark" and edited.bubble_bbox == bubble and edited.text == "안녕"


def test_deleted_region_is_dropped_without_reordering_the_rest() -> None:
    edits = ChapterEdits(regions=[RegionEdit(region_id="r0001", anchor=A.bbox, deleted=True)])
    regions, _ = apply_region_edits(AUTO, edits, SLICES)
    assert [(r.id, r.reading_order) for r in regions] == [("r0002", 1), ("r0003", 0)]


def test_added_region_is_inserted_in_its_slice_and_ordered() -> None:
    new = box(10, 100, 110, 150)  # between r0001 and r0002
    edits = ChapterEdits(
        regions=[
            RegionEdit(region_id="m0001", anchor=new, added=True, kind="free_text", text="툭", lang="ko")
        ]
    )
    regions, _ = apply_region_edits(AUTO, edits, SLICES)
    added = next(r for r in regions if r.id == "m0001")
    assert (added.kind, added.text, added.slice_index, added.confidence) == ("free_text", "툭", 0, 1.0)
    assert added.lines == [OcrLine(bbox=new, text="툭", score=1.0, engine=MANUAL_ENGINE)]
    by_id = {r.id: r.reading_order for r in regions}
    assert (by_id["r0001"], by_id["m0001"], by_id["r0002"]) == (0, 1, 2)


def test_added_region_replaces_an_unclaimed_pipeline_region_it_covers() -> None:
    edits = ChapterEdits(regions=[RegionEdit(region_id="m0001", anchor=box(12, 12, 112, 62), added=True)])
    ids = [r.id for r in apply_region_edits(AUTO, edits, SLICES)[0]]
    assert ids == ["r0002", "r0003", "m0001"]


def test_added_region_keeps_a_covered_region_another_edit_claims() -> None:
    edits = ChapterEdits(
        regions=[
            RegionEdit(region_id="r0001", anchor=A.bbox, text="안녕하세요"),
            RegionEdit(region_id="m0001", anchor=box(12, 12, 112, 62), added=True),
        ]
    )
    ids = [r.id for r in apply_region_edits(AUTO, edits, SLICES)[0]]
    assert ids == ["r0001", "r0002", "r0003", "m0001"]


def test_edit_follows_its_region_when_ids_shift() -> None:
    """A re-run renumbered the regions: the edit applies by box overlap, not by the stale id."""
    extra = region("r0001", box(300, 10, 400, 60), "새로운")
    shifted = [extra, A.model_copy(update={"id": "r0002"}), B.model_copy(update={"id": "r0003"})]
    edits = ChapterEdits(regions=[RegionEdit(region_id="r0001", anchor=A.bbox, text="고침")])
    regions, orphans = apply_region_edits(shifted, edits, SLICES)
    assert orphans == 0
    assert {r.id: r.text for r in regions} == {"r0001": "새로운", "r0002": "고침", "r0003": "반가워"}


def test_edit_whose_region_is_gone_is_an_orphan() -> None:
    edits = ChapterEdits(regions=[RegionEdit(region_id="r0009", anchor=box(500, 500, 600, 600), text="x")])
    assert apply_region_edits(AUTO, edits, SLICES) == (AUTO, 1)


def test_reapplying_to_an_already_moved_region_is_stable() -> None:
    """Matching also accepts the box the edit moved the region to, so an edit re-applies to its own output."""
    moved = box(300, 300, 400, 350)
    edits = ChapterEdits(regions=[RegionEdit(region_id="r0001", anchor=A.bbox, bbox=moved)])
    once, _ = apply_region_edits(AUTO, edits, SLICES)
    twice, orphans = apply_region_edits(once, edits, SLICES)
    assert orphans == 0 and twice == once


def test_match_region_prefers_the_same_id_then_best_overlap_and_respects_taken() -> None:
    near = region("r0005", box(12, 12, 112, 62))
    assert match_region("r0001", [A.bbox], [near, A]) is A
    assert match_region("r0009", [A.bbox], [near, A]) is A  # identical box beats a near one
    assert match_region("r0009", [A.bbox], [near, A], taken={"r0001"}) is near
    assert match_region("r0009", [box(900, 900, 950, 950)], AUTO) is None


def test_each_region_is_claimed_by_one_edit_only() -> None:
    edits = ChapterEdits(
        regions=[
            RegionEdit(region_id="r0001", anchor=A.bbox, text="first"),
            RegionEdit(region_id="r0001", anchor=A.bbox, text="second"),
        ]
    )
    assert match_region_edits(AUTO, edits) == {0: "r0001"}


def line(rid: str, text: str) -> FinalLine:
    return FinalLine(region_id=rid, text=text, decision="pick", sources=["run1"])


def test_translation_edit_replaces_the_judges_line() -> None:
    edits = ChapterEdits(
        translations=[TranslationEdit(region_id="r0002", anchor=B.bbox, text="Nice!", source="반가워")]
    )
    lines, orphans = apply_translation_edits([line("r0001", "Hi"), line("r0002", "Glad")], edits, AUTO)
    assert orphans == 0
    assert [(x.region_id, x.text, x.decision, x.flags) for x in lines] == [
        ("r0001", "Hi", "pick", []),
        ("r0002", "Nice!", "manual", []),
    ]


def test_translation_edit_for_a_region_without_a_line_is_appended() -> None:
    edits = ChapterEdits(
        translations=[TranslationEdit(region_id="r0003", anchor=C.bbox, text="Bye", source="잘가")]
    )
    lines, _ = apply_translation_edits([line("r0001", "Hi")], edits, AUTO)
    assert [(x.region_id, x.text) for x in lines] == [("r0001", "Hi"), ("r0003", "Bye")]


def test_translation_edit_is_flagged_when_the_source_changed() -> None:
    edits = ChapterEdits(
        translations=[TranslationEdit(region_id="r0002", anchor=B.bbox, text="Nice!", source="다른 글")]
    )
    lines, _ = apply_translation_edits([], edits, AUTO)
    assert lines[0].flags == [SOURCE_CHANGED]


def test_translation_edit_whose_region_is_gone_is_an_orphan() -> None:
    edits = ChapterEdits(
        translations=[TranslationEdit(region_id="r0009", anchor=box(900, 900, 950, 950), text="x", source="")]
    )
    assert apply_translation_edits([line("r0001", "Hi")], edits, AUTO) == ([line("r0001", "Hi")], 1)

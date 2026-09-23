"""Fixed-position watermark reclassification (card F2c): IoA overlap + detect-time region reclassify."""

from __future__ import annotations

from omniscan.core.schemas import BBox, Region, RegionKind
from omniscan.detect.watermark_position import (
    _ioa,
    reclassify_watermark_position_regions,
    region_overlaps_watermark,
)
from omniscan.translate.prompts import translatable

WATERMARK = BBox(x0=0, y0=0, x1=100, y1=100)  # area 10 000 (card F2c's pinned watermark zone)
REGION_ABOVE = BBox(x0=0, y0=0, x1=100, y1=150)  # IoA 10 000/15 000 -> reclassified
REGION_BELOW = BBox(x0=0, y0=50, x1=100, y1=250)  # IoA 5 000/20 000 -> not reclassified
REGION_BOUNDARY = BBox(x0=0, y0=0, x1=100, y1=200)  # IoA 10 000/20 000 = exactly 0.5 -> reclassified


def region(kind: RegionKind, bbox: BBox, region_id: str = "r0001", text: str = "") -> Region:
    return Region(id=region_id, slice_index=0, kind=kind, bbox=bbox, text=text, confidence=0.9)


# ---------------------------------------------------------------- _ioa


def test_ioa_pinned_examples() -> None:
    assert _ioa(REGION_ABOVE, WATERMARK) == 10_000 / 15_000
    assert _ioa(REGION_BELOW, WATERMARK) == 5_000 / 20_000
    assert _ioa(REGION_BOUNDARY, WATERMARK) == 10_000 / 20_000


def test_ioa_is_zero_for_a_zero_area_region_not_an_error() -> None:
    assert _ioa(BBox(x0=40, y0=40, x1=40, y1=90), WATERMARK) == 0.0  # x0 == x1
    assert _ioa(BBox(x0=0, y0=50, x1=100, y1=50), WATERMARK) == 0.0  # y0 == y1
    assert region_overlaps_watermark(BBox(x0=40, y0=40, x1=40, y1=90), [WATERMARK]) is False


def test_ioa_is_zero_for_disjoint_boxes() -> None:
    assert _ioa(BBox(x0=200, y0=200, x1=300, y1=300), WATERMARK) == 0.0


# ---------------------------------------------------------------- region_overlaps_watermark


def test_the_pinned_examples_decide_by_the_threshold() -> None:
    assert region_overlaps_watermark(REGION_ABOVE, [WATERMARK])
    assert region_overlaps_watermark(REGION_BOUNDARY, [WATERMARK])  # exactly 0.5 counts (inclusive)
    assert not region_overlaps_watermark(REGION_BELOW, [WATERMARK])


def test_boxes_are_checked_independently_not_unioned() -> None:
    # each box alone holds only 0.4 of the region, their union 0.8: unioning would wrongly match
    half_each = [BBox(x0=0, y0=0, x1=100, y1=40), BBox(x0=0, y0=60, x1=100, y1=100)]
    assert not region_overlaps_watermark(WATERMARK, half_each)


def test_a_match_in_any_single_box_is_enough() -> None:
    first_clears = [WATERMARK, BBox(x0=500, y0=500, x1=600, y1=600)]
    second_clears = [BBox(x0=500, y0=500, x1=600, y1=600), WATERMARK]
    assert region_overlaps_watermark(REGION_ABOVE, first_clears)
    assert region_overlaps_watermark(REGION_ABOVE, second_clears)


def test_empty_watermark_boxes_never_match() -> None:
    assert not region_overlaps_watermark(REGION_ABOVE, [])


# ---------------------------------------------------------------- reclassify_watermark_position_regions


def test_overlapping_regions_become_watermarks_and_others_pass_through() -> None:
    catch = region("bubble_text", REGION_ABOVE, "r0001")
    free = region("free_text", REGION_BELOW, "r0002")
    sfx = region("sfx", REGION_ABOVE, "r0003")  # an sfx that overlaps stays sfx (F2b's carve-out)
    already = region("watermark", REGION_BELOW, "r0004")  # stays a watermark either way

    out = reclassify_watermark_position_regions([catch, free, sfx, already], [WATERMARK])

    assert [r.kind for r in out] == ["watermark", "free_text", "sfx", "watermark"]
    assert out[0] is not catch  # reclassified regions are new copies
    assert out[1] is free and out[2] is sfx and out[3] is already  # untouched regions keep identity
    assert out[0].model_dump(exclude={"kind"}) == catch.model_dump(exclude={"kind"})


def test_an_already_watermark_region_stays_one_whether_it_overlaps_or_not() -> None:
    overlapping = region("watermark", REGION_ABOVE, "r0001")
    elsewhere = region("watermark", REGION_BELOW, "r0002")
    out = reclassify_watermark_position_regions([overlapping, elsewhere], [WATERMARK])
    assert [r.kind for r in out] == ["watermark", "watermark"]
    assert out[0] is overlapping and out[1] is elsewhere  # never un-marked, never copied needlessly


def test_empty_watermark_boxes_return_every_region_unchanged() -> None:
    regions = [
        region("bubble_text", REGION_ABOVE, "r0001"),
        region("free_text", REGION_BELOW, "r0002"),
        region("sfx", REGION_ABOVE, "r0003"),
    ]
    out = reclassify_watermark_position_regions(regions, [])
    assert len(out) == 3
    assert all(a is b for a, b in zip(out, regions, strict=True))


# ---------------------------------------------------------------- downstream (no new code needed)


def test_a_position_reclassified_region_is_excluded_from_translation() -> None:
    stamp = region("bubble_text", REGION_ABOVE, "r0001", text="dialogue")
    (watermark,) = reclassify_watermark_position_regions([stamp], [WATERMARK])
    assert watermark.kind == "watermark"
    assert translatable([watermark]) == []

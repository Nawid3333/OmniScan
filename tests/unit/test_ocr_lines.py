"""Line logic against exact boxes: polygon_box, merge (NMS + containment), assignment (card C4a)."""

from __future__ import annotations

import pytest

from omniscan.core.schemas import BBox, Region
from omniscan.ocr.lines import LineBox, assign_lines, merge_lines, polygon_box

Box = tuple[float, float, float, float]


def line(box: Box, score: float = 1.0) -> LineBox:
    return LineBox(box=box, score=score)


def region(rid: str, box: tuple[int, int, int, int]) -> Region:
    return Region(
        id=rid, slice_index=0, kind="bubble_text", bbox=BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3])
    )


def test_polygon_box_takes_the_bounds_of_the_points() -> None:
    assert polygon_box([[10, 5], [110, 7], [108, 25], [9, 22]]) == (9, 5, 110, 25)


def test_merge_lines_drops_the_lower_scored_duplicate() -> None:
    merged = merge_lines([line((100, 100, 400, 130), 0.9), line((102, 101, 401, 131), 0.8)], nms_iou=0.5)
    assert merged == [line((100, 100, 400, 130), 0.9)]


def test_merge_lines_containment_rule_prefers_the_whole_line() -> None:
    # IoU exactly nms_iou survives NMS; the truncated box is fully inside the bigger one and loses.
    merged = merge_lines([line((100, 100, 250, 130), 0.95), line((100, 100, 400, 130), 0.7)], nms_iou=0.5)
    assert merged == [line((100, 100, 400, 130), 0.7)]


def test_merge_lines_keeps_different_lines_and_sorts_by_position() -> None:
    merged = merge_lines(
        [
            line((100, 200, 400, 230), 0.8),
            line((100, 100, 400, 130), 0.9),
            line((10, 100, 60, 130), 0.7),
        ],
        nms_iou=0.5,
    )
    assert merged == [
        line((10, 100, 60, 130), 0.7),
        line((100, 100, 400, 130), 0.9),
        line((100, 200, 400, 230), 0.8),
    ]


def test_assign_lines_basic_and_orphans() -> None:
    r1 = region("r0001", (100, 100, 400, 200))
    inside = line((120, 110, 380, 140))
    half_out = line((120, 190, 380, 240))  # only 18 of 50 rows inside the padded box: ioa 0.36
    far = line((500, 500, 600, 540))

    by_region, orphans = assign_lines([r1], [inside, half_out, far], min_ioa=0.5, pad_px=8, direction="ltr")

    assert by_region == {"r0001": [inside]}
    assert orphans == [half_out, far]


def test_assign_lines_tie_goes_to_the_smaller_region() -> None:
    r1 = region("r0001", (100, 100, 400, 200))  # area 30000
    r2 = region("r0002", (110, 140, 390, 190))  # area 14000
    shared = line((120, 150, 380, 180))  # wholly inside both padded boxes

    by_region, orphans = assign_lines([r1, r2], [shared], min_ioa=0.5, pad_px=8, direction="ltr")

    assert by_region == {"r0001": [], "r0002": [shared]}
    assert orphans == []


def test_assign_lines_region_without_lines_keeps_an_entry() -> None:
    r1 = region("r0001", (100, 100, 400, 200))
    by_region, orphans = assign_lines([r1], [], min_ioa=0.5, pad_px=8, direction="ltr")
    assert by_region == {"r0001": []}
    assert orphans == []


def test_assign_lines_orders_a_column_top_to_bottom() -> None:
    r1 = region("r0001", (100, 100, 400, 400))
    bottom = line((120, 300, 380, 330))
    top = line((120, 150, 380, 180))

    by_region, _orphans = assign_lines([r1], [bottom, top], min_ioa=0.5, pad_px=8, direction="ltr")

    assert by_region["r0001"] == [top, bottom]


@pytest.mark.parametrize("direction", ["ltr", "rtl"])
def test_assign_lines_orders_a_row_by_direction(direction: str) -> None:
    r1 = region("r0001", (90, 90, 310, 140))
    left = line((100, 100, 190, 130))
    right = line((210, 102, 300, 132))  # same row: vertical overlap 28 of 30 >= half

    by_region, _orphans = assign_lines([r1], [right, left], min_ioa=0.5, pad_px=8, direction=direction)  # type: ignore[arg-type]

    expected = [left, right] if direction == "ltr" else [right, left]
    assert by_region["r0001"] == expected


def test_assign_lines_orphans_keep_the_input_order() -> None:
    r1 = region("r0001", (100, 100, 400, 200))
    first = line((500, 100, 600, 130))
    second = line((500, 300, 600, 330))

    by_region, orphans = assign_lines([r1], [second, first], min_ioa=0.5, pad_px=8, direction="ltr")

    assert by_region == {"r0001": []}
    assert orphans == [second, first]

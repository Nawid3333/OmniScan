"""Tests for omniscan.export.segments — output images from slices or hand-set cuts (pure)."""

from __future__ import annotations

from omniscan.core.schemas import BBox, Region, Slice
from omniscan.export.segments import Segment, cut_crossings, output_segments

SLICES = [
    Slice(index=0, y0=0, y1=400),
    Slice(index=1, y0=400, y1=600, filtered=True),  # a promo slice
    Slice(index=2, y0=600, y1=1200),
]


def test_without_cuts_one_image_per_kept_slice() -> None:
    assert output_segments(SLICES, 1200, None) == [Segment(0, 400, 0), Segment(600, 1200, 2)]


def test_hand_cuts_split_the_strip_and_filtered_rows_stay_out() -> None:
    # 300: inside slice 0; 900: inside slice 2; the piece 300..900 loses the filtered rows 400..600
    assert output_segments(SLICES, 1200, [900, 300]) == [
        Segment(0, 300, 0),
        Segment(300, 400, 0),
        Segment(600, 900, 2),
        Segment(900, 1200, 2),
    ]


def test_cuts_outside_the_strip_and_duplicates_are_ignored() -> None:
    assert output_segments(SLICES, 1200, [0, 1200, 5000, -3, 300, 300]) == output_segments(
        SLICES, 1200, [300]
    )
    assert output_segments([Slice(index=0, y0=0, y1=100)], 100, []) == [Segment(0, 100, 0)]


def test_a_piece_entirely_inside_a_filtered_slice_disappears() -> None:
    assert output_segments(SLICES, 1200, [450, 550]) == [
        Segment(0, 400, 0),
        Segment(600, 1200, 2),
    ]


def test_cut_crossings_name_the_regions_a_cut_splits() -> None:
    regions = [
        Region(id="r0001", slice_index=0, kind="bubble_text", bbox=BBox(x0=0, y0=100, x1=50, y1=150)),
        Region(
            id="r0002",
            slice_index=0,
            kind="bubble_text",
            bbox=BBox(x0=0, y0=300, x1=50, y1=320),
            bubble_bbox=BBox(x0=0, y0=280, x1=80, y1=360),
        ),
    ]
    assert cut_crossings([120, 350, 150, 500], regions) == [(120, "r0001"), (350, "r0002")]

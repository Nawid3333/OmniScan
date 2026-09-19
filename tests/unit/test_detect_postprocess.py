"""Golden values for detect post-processing: tile->strip, NMS, reading order, region building (card C3)."""

from __future__ import annotations

from omniscan.core.schemas import Slice
from omniscan.detect.postprocess import Det, build_regions, merge_detections, reading_order, tile_det_to_strip
from omniscan.detect.tiles import plan_tiles

TILES = plan_tiles(1000, 3000, 1280, 0.5)
THIRD_TILE = TILES[2]  # (0, 1000, 1000, 2000): top and bottom are internal edges


# ---------------------------------------------------------------- tile -> strip


def test_tile_det_to_strip_moves_and_flags_internal_edges() -> None:
    d = tile_det_to_strip(THIRD_TILE, "bubble", 0.9, (10, 0, 300, 400))
    assert d == Det("bubble", 0.9, (10, 1000, 300, 1400), True)  # top of tile 3 is internal

    d = tile_det_to_strip(THIRD_TILE, "bubble", 0.9, (10, 100, 300, 400))
    assert d == Det("bubble", 0.9, (10, 1100, 300, 1400), False)

    d = tile_det_to_strip(THIRD_TILE, "bubble", 0.9, (10, 100, 300, 998))
    assert d == Det("bubble", 0.9, (10, 1100, 300, 1998), True)  # the tile's bottom is internal


def test_tile_det_to_strip_outer_border_is_never_edge() -> None:
    first = TILES[0]  # top=False: the strip's own border is not a cut
    d = tile_det_to_strip(first, "bubble", 0.9, (10, 0, 300, 400))
    assert d == Det("bubble", 0.9, (10, 0, 300, 400), False)
    d = tile_det_to_strip(first, "text_free", 0.5, (0, 100, 2, 900))
    assert d == Det("text_free", 0.5, (0, 100, 2, 900), False)  # touching x=0 too
    last = TILES[-1]  # bottom=False
    d = tile_det_to_strip(last, "text_bubble", 0.4, (10, 100, 300, 1000))
    assert d == Det("text_bubble", 0.4, (10, 2100, 300, 3000), False)


# ---------------------------------------------------------------- merging


def test_merge_detections_nms_keeps_best_per_class() -> None:
    dets = [
        Det("bubble", 0.9, (0, 0, 100, 100)),
        Det("bubble", 0.8, (5, 5, 100, 100)),
        Det("bubble", 0.7, (500, 500, 600, 600)),
    ]
    merged = merge_detections(dets, nms_iou=0.5, contain_thr=0.85, edge_penalty=0.15)
    assert [(d.cls, d.score) for d in merged] == [("bubble", 0.9), ("bubble", 0.7)]


def test_merge_detections_penalised_edge_box_dropped_as_truncated() -> None:
    dets = [
        Det("text_bubble", 0.95, (0, 0, 50, 50), True),
        Det("text_bubble", 0.7, (0, 0, 100, 100)),
    ]
    merged = merge_detections(dets, nms_iou=0.5, contain_thr=0.85, edge_penalty=0.15)
    assert [(d.score, d.box, d.edge) for d in merged] == [(0.7, (0.0, 0.0, 100.0, 100.0), False)]


def test_merge_detections_never_suppresses_across_classes() -> None:
    dets = [
        Det("bubble", 0.9, (0, 0, 100, 100)),
        Det("text_bubble", 0.8, (0, 0, 100, 100)),
        Det("text_free", 0.7, (0, 0, 100, 100)),
    ]
    merged = merge_detections(dets, nms_iou=0.5, contain_thr=0.85, edge_penalty=0.15)
    assert [d.cls for d in merged] == ["bubble", "text_bubble", "text_free"]


# ---------------------------------------------------------------- reading order


def test_reading_order_ltr_rows_top_to_bottom() -> None:
    boxes = [(100, 0, 200, 50), (0, 5, 90, 55), (0, 100, 90, 150)]
    assert reading_order(boxes, "ltr") == [1, 0, 2]


def test_reading_order_rtl() -> None:
    boxes = [(100, 0, 200, 50), (0, 5, 90, 55), (0, 100, 90, 150)]
    assert reading_order(boxes, "rtl") == [0, 1, 2]


def test_reading_order_empty() -> None:
    assert reading_order([], "ltr") == []


# ---------------------------------------------------------------- regions


def _slices() -> list[Slice]:
    return [
        Slice(index=0, y0=0, y1=1000),
        Slice(index=1, y0=1000, y1=2000, blank=True),
        Slice(index=2, y0=2000, y1=3000),
    ]


def test_build_regions_pairs_text_with_bubble_and_drops_blank_slices() -> None:
    dets = [
        Det("bubble", 0.9, (100, 100, 500, 400)),
        Det("text_bubble", 0.8, (150, 150, 450, 350)),
        Det("text_free", 0.6, (600, 2100, 900, 2200)),
        Det("text_free", 0.6, (600, 1100, 900, 1200)),  # inside the blank slice
    ]
    regions = build_regions(
        dets, _slices(), strip_width=1000, strip_height=3000, merge_bubble_text=True, direction="ltr"
    )
    assert len(regions) == 2
    r1, r2 = regions
    assert r1.id == "r0001" and r1.slice_index == 0 and r1.kind == "bubble_text"
    assert (r1.bbox.x0, r1.bbox.y0, r1.bbox.x1, r1.bbox.y1) == (150, 150, 450, 350)
    assert r1.bubble_bbox is not None  # the detector saw the bubble, so build_regions kept it
    assert (r1.bubble_bbox.x0, r1.bubble_bbox.y0, r1.bubble_bbox.x1, r1.bubble_bbox.y1) == (
        100,
        100,
        500,
        400,
    )
    assert r1.reading_order == 0 and r1.confidence == 0.8
    assert r2.id == "r0002" and r2.slice_index == 2 and r2.kind == "free_text"
    assert (r2.bbox.x0, r2.bbox.y0, r2.bbox.x1, r2.bbox.y1) == (600, 2100, 900, 2200)
    assert r2.bubble_bbox is None and r2.reading_order == 0 and r2.confidence == 0.6


def test_build_regions_merges_two_texts_in_one_bubble() -> None:
    bubble = Det("bubble", 0.9, (0, 0, 200, 200))
    texts = [Det("text_bubble", 0.7, (10, 10, 90, 90)), Det("text_bubble", 0.8, (100, 100, 190, 190))]
    merged = build_regions(
        [bubble, *texts],
        _slices(),
        strip_width=1000,
        strip_height=3000,
        merge_bubble_text=True,
        direction="ltr",
    )
    assert len(merged) == 1
    region = merged[0]
    assert (region.bbox.x0, region.bbox.y0, region.bbox.x1, region.bbox.y1) == (10, 10, 190, 190)
    assert region.confidence == 0.8

    split = build_regions(
        [bubble, *texts],
        _slices(),
        strip_width=1000,
        strip_height=3000,
        merge_bubble_text=False,
        direction="ltr",
    )
    assert len(split) == 2
    assert [r.reading_order for r in split] == [0, 1]


def test_build_regions_text_without_bubble_and_cross_class_duplicate() -> None:
    dets = [
        Det("text_bubble", 0.6, (0, 0, 100, 100)),  # no bubble contains it
        Det("text_free", 0.7, (500, 10, 600, 110)),  # IoU 0 with the text_bubble
        Det("text_bubble", 0.5, (500, 10, 600, 110)),  # same box as the text_free, lower score
    ]
    regions = build_regions(
        dets, _slices(), strip_width=1000, strip_height=3000, merge_bubble_text=True, direction="ltr"
    )
    assert [(r.kind, r.bbox.x0, r.bbox.y0, r.confidence, r.bubble_bbox is None) for r in regions] == [
        ("bubble_text", 0, 0, 0.6, True),
        ("free_text", 500, 10, 0.7, True),
    ]


def test_build_regions_ids_follow_slice_then_reading_order() -> None:
    dets = [
        Det("text_free", 0.6, (0, 2200, 100, 2300)),
        Det("text_free", 0.7, (500, 100, 600, 200)),
        Det("text_free", 0.8, (0, 100, 100, 200)),
    ]
    regions = build_regions(
        dets, _slices(), strip_width=1000, strip_height=3000, merge_bubble_text=True, direction="ltr"
    )
    assert [(r.id, r.slice_index, r.bbox.x0, r.bbox.y0) for r in regions] == [
        ("r0001", 0, 0, 100),
        ("r0002", 0, 500, 100),
        ("r0003", 2, 0, 2200),
    ]


def test_build_regions_drops_filtered_slices_and_clamps_boxes() -> None:
    slices = [Slice(index=0, y0=0, y1=1000, filtered=True), Slice(index=1, y0=1000, y1=2000)]
    dets = [
        Det("text_free", 0.6, (-10, 100, 2000, 250)),
        Det("text_free", 0.9, (10, 1100, 2000, 5000)),  # clamped to the strip on every side
    ]
    regions = build_regions(
        dets, slices, strip_width=1000, strip_height=2000, merge_bubble_text=True, direction="ltr"
    )
    assert [(r.id, r.slice_index, r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1) for r in regions] == [
        ("r0001", 1, 10, 1100, 1000, 2000)
    ]

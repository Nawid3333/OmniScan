"""Unit tests for omniscan.typeset.plan (CPU only; the card's fake font plus real font metrics)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.core.config import TypesetConfig
from omniscan.core.schemas import RGB, BBox, LayoutItem, Region, RegionKind
from omniscan.typeset.fonts import default_font_path
from omniscan.typeset.plan import ellipse_polygon, luminance, plan_layout, target_box

# ---------------------------------------------------------------- fake font (card spec)


class Fake:
    """Advance width = len(text) * size * 0.5."""

    def __init__(self, size: int) -> None:
        self.size = size

    def getlength(self, text: str) -> float:
        return len(text) * self.size * 0.5


def fake_factory(path: Path, size: int) -> Fake:
    return Fake(size)


def make_region(
    region_id: str,
    kind: RegionKind,
    bbox: BBox,
    *,
    bubble_bbox: BBox | None = None,
    text: str = "원문",
    text_color: RGB | None = None,
    stroke_color: RGB | None = None,
    slice_index: int = 0,
    reading_order: int = 0,
) -> Region:
    return Region(
        id=region_id,
        slice_index=slice_index,
        kind=kind,
        bbox=bbox,
        bubble_bbox=bubble_bbox,
        text=text,
        text_color=text_color,
        stroke_color=stroke_color,
        reading_order=reading_order,
    )


# ---------------------------------------------------------------- luminance


def test_luminance_greys() -> None:
    assert luminance((128, 128, 128)) == pytest.approx(128.0, abs=1e-9)
    assert luminance((127, 127, 127)) == pytest.approx(127.0, abs=1e-9)


def test_luminance_endpoints() -> None:
    assert luminance((0, 0, 0)) == 0.0
    assert luminance((255, 255, 255)) == pytest.approx(255.0, abs=1e-9)


# ---------------------------------------------------------------- ellipse polygon


def test_ellipse_polygon_48_points() -> None:
    polygon = ellipse_polygon(BBox(x0=0, y0=0, x1=400, y1=200))
    assert len(polygon) == 48
    assert polygon[0] == (400, 100)  # t = 0: the right edge's midpoint
    assert polygon[12] == (200, 200)  # t = pi/2: the bottom edge's midpoint (y points down)
    assert all(0 <= x <= 400 and 0 <= y <= 200 for x, y in polygon)


def test_ellipse_polygon_no_duplicate_consecutive_points() -> None:
    polygon = ellipse_polygon(BBox(x0=0, y0=0, x1=400, y1=200))
    assert len(set(polygon)) == len(polygon)  # every vertex is distinct
    assert polygon[0] != polygon[-1]  # the polygon closes without a repeated vertex


def test_ellipse_polygon_point_count_is_configurable() -> None:
    assert len(ellipse_polygon(BBox(x0=0, y0=0, x1=100, y1=100), points=8)) == 8
    assert len(ellipse_polygon(BBox(x0=0, y0=0, x1=100, y1=100), points=64)) == 64


# ---------------------------------------------------------------- target box

BUBBLE = BBox(x0=0, y0=0, x1=400, y1=200)
ELLIPSE_BOX = BBox(x0=66, y0=36, x1=334, y1=164)  # inscribed ellipse box of BUBBLE, margin 6


def test_target_box_bubble_uses_inscribed_ellipse_box() -> None:
    region = make_region("r1", "bubble_text", BBox(x0=150, y0=80, x1=250, y1=120), bubble_bbox=BUBBLE)
    assert target_box(region, TypesetConfig(margin_px=6)) == ELLIPSE_BOX


def test_target_box_bubble_prefers_larger_original_text_box() -> None:
    region = make_region("r1", "bubble_text", BBox(x0=20, y0=20, x1=380, y1=180), bubble_bbox=BUBBLE)
    assert target_box(region, TypesetConfig(margin_px=6)) == BBox(x0=20, y0=20, x1=380, y1=180)


def test_target_box_bubble_equal_area_returns_ellipse_box() -> None:
    # 268x128 = the ellipse box's area exactly: only a strictly larger text box wins
    region = make_region("r1", "bubble_text", BBox(x0=0, y0=0, x1=268, y1=128), bubble_bbox=BUBBLE)
    assert target_box(region, TypesetConfig(margin_px=6)) == ELLIPSE_BOX


def test_target_box_bubble_without_bubble_bbox_grows_by_free_grow() -> None:
    region = make_region("r1", "bubble_text", BBox(x0=90, y0=70, x1=110, y1=90))
    assert target_box(region, TypesetConfig(free_grow=0.10)) == BBox(x0=88, y0=68, x1=112, y1=92)


def test_target_box_free_text_clamps_at_origin() -> None:
    region = make_region("r1", "free_text", BBox(x0=2, y0=2, x1=52, y1=32))
    assert target_box(region, TypesetConfig(free_grow=0.10)) == BBox(x0=0, y0=0, x1=57, y1=35)


def test_target_box_sfx_behaves_like_free_text() -> None:
    region = make_region("r1", "sfx", BBox(x0=50, y0=50, x1=150, y1=100))
    assert target_box(region, TypesetConfig(free_grow=0.10)) == BBox(x0=40, y0=45, x1=160, y1=105)


# ---------------------------------------------------------------- role / colour / stroke


def run_plan_layout(region: Region, line: str = "Hello.", fill: RGB | None = None) -> list[LayoutItem]:
    fills = {} if fill is None else {region.id: fill}
    return plan_layout([region], {region.id: line}, fills, TypesetConfig(), font_factory=fake_factory)


def test_plan_layout_bubble_on_white_fill() -> None:
    region = make_region("r1", "bubble_text", BBox(x0=150, y0=80, x1=250, y1=120), bubble_bbox=BUBBLE)
    (item,) = run_plan_layout(region, fill=(255, 255, 255))
    assert item.font_role == "dialogue"
    assert item.color == (0, 0, 0)
    assert item.stroke_px == 0
    assert item.stroke_color == (0, 0, 0)


def test_plan_layout_bubble_on_dark_fill() -> None:
    region = make_region("r1", "bubble_text", BBox(x0=150, y0=80, x1=250, y1=120), bubble_bbox=BUBBLE)
    (item,) = run_plan_layout(region, fill=(24, 24, 32))
    assert item.color == (255, 255, 255)


def test_plan_layout_bubble_without_fill_is_black() -> None:
    region = make_region("r1", "bubble_text", BBox(x0=150, y0=80, x1=250, y1=120), bubble_bbox=BUBBLE)
    (item,) = run_plan_layout(region)
    assert item.color == (0, 0, 0)


def test_plan_layout_free_text() -> None:
    region = make_region("r1", "free_text", BBox(x0=0, y0=0, x1=200, y1=60))
    (item,) = run_plan_layout(region)
    assert item.font_role == "free"
    assert item.color == (255, 255, 255)
    assert item.stroke_px == TypesetConfig().stroke_free_px
    assert item.stroke_color == (0, 0, 0)


def test_plan_layout_sfx() -> None:
    region = make_region("r1", "sfx", BBox(x0=0, y0=0, x1=200, y1=60))
    (item,) = run_plan_layout(region, line="KRAKOOM")
    assert item.font_role == "sfx"
    assert item.color == (255, 255, 255)
    assert item.stroke_px == TypesetConfig().stroke_sfx_px


def test_plan_layout_region_colours_override_the_defaults() -> None:
    region = make_region(
        "r1",
        "free_text",
        BBox(x0=0, y0=0, x1=200, y1=60),
        text_color=(10, 20, 30),
        stroke_color=(1, 2, 3),
    )
    (item,) = run_plan_layout(region)
    assert item.color == (10, 20, 30)
    assert item.stroke_color == (1, 2, 3)


# ---------------------------------------------------------------- region selection and order


def test_plan_layout_skips_empty_whitespace_and_missing_lines() -> None:
    regions = [
        make_region("r1", "bubble_text", BBox(x0=0, y0=0, x1=200, y1=100)),
        make_region("r2", "bubble_text", BBox(x0=0, y0=100, x1=200, y1=200)),
        make_region("r3", "bubble_text", BBox(x0=0, y0=200, x1=200, y1=300)),
    ]
    lines = {"r1": "   ", "r2": "Hello."}  # r3 has no final line at all
    items = plan_layout(regions, lines, {}, TypesetConfig(), font_factory=fake_factory)
    assert [item.region_id for item in items] == ["r2"]


def test_plan_layout_skips_watermark_regions() -> None:
    region = make_region("r1", "watermark", BBox(x0=0, y0=0, x1=200, y1=100), text="omniscan")
    assert plan_layout([region], {"r1": "omniscan"}, {}, TypesetConfig(), font_factory=fake_factory) == []


def test_plan_layout_keeps_translatable_order() -> None:
    r2 = make_region("r2", "bubble_text", BBox(x0=0, y0=0, x1=200, y1=100))
    r10 = make_region("r10", "bubble_text", BBox(x0=0, y0=100, x1=200, y1=200))
    lines = {r.id: "Hello." for r in (r2, r10)}
    items = plan_layout([r10, r2], lines, {}, TypesetConfig(), font_factory=fake_factory)
    assert [item.region_id for item in items] == ["r2", "r10"]  # natural order, not list order


def test_plan_layout_item_fields_come_from_layout_region() -> None:
    dialogue = make_region("r1", "bubble_text", BBox(x0=0, y0=0, x1=200, y1=100), bubble_bbox=BUBBLE)
    sfx = make_region("r2", "sfx", BBox(x0=0, y0=100, x1=200, y1=200))
    targets = [target_box(r, TypesetConfig()) for r in (dialogue, sfx)]
    items = plan_layout(
        [dialogue, sfx],
        {"r1": "Hello.", "r2": "KRAKOOM"},
        {},
        TypesetConfig(),
        font_factory=fake_factory,
    )
    assert [item.font for item in items] == ["ComicNeue-Bold.ttf", "Bangers-Regular.ttf"]
    assert [item.align for item in items] == ["center", "center"]
    for item, target in zip(items, targets, strict=True):
        assert target.x0 <= item.box.x0 <= item.box.x1 <= target.x1
        assert target.y0 <= item.box.y0 <= item.box.y1 <= target.y1


def test_plan_layout_overflow_at_min_px() -> None:
    # a small bubble: the long text cannot fit at any size, so min_px wins with overflow
    small_bubble = BBox(x0=0, y0=0, x1=100, y1=60)
    region = make_region("r1", "bubble_text", BBox(x0=30, y0=15, x1=70, y1=45), bubble_bbox=small_bubble)
    too_long = "This sentence is far too long to ever fit inside such a tiny target box, no matter what."
    (item,) = run_plan_layout(region, line=too_long)
    assert item.overflow is True
    assert item.size_px == TypesetConfig().min_px


# ---------------------------------------------------------------- real fonts


def test_plan_layout_real_font_fits_inside_the_bubble() -> None:
    region = make_region("r1", "bubble_text", BBox(x0=150, y0=80, x1=250, y1=120), bubble_bbox=BUBBLE)
    (item,) = plan_layout(
        [region], {"r1": "Are you okay? The dungeon just opened!"}, {"r1": (255, 255, 255)}, TypesetConfig()
    )
    assert item.color == (0, 0, 0)
    assert item.overflow is False
    assert TypesetConfig().min_px <= item.size_px <= TypesetConfig().max_px
    ell = ELLIPSE_BOX
    assert ell.x0 <= item.box.x0 and item.box.x1 <= ell.x1
    assert ell.y0 <= item.box.y0 and item.box.y1 <= ell.y1
    assert item.font == default_font_path("dialogue").name

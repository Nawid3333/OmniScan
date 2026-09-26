"""Tests for omniscan.typeset.overrides — hand-set lettering over the typesetter's items (real fonts)."""

from __future__ import annotations

import pytest

from omniscan.core.config import TypesetConfig
from omniscan.core.schemas import BBox, ChapterEdits, LayoutEdit, LayoutItem, Region
from omniscan.typeset.overrides import apply_layout_edits, overridden
from omniscan.typeset.plan import plan_layout

CFG = TypesetConfig(style="webtoon")
TEXT = "Where do you think you are going at this hour?"


def region(rid: str = "r0001", y0: int = 100) -> Region:
    return Region(
        id=rid,
        slice_index=0,
        kind="bubble_text",
        bbox=BBox(x0=120, y0=y0 + 40, x1=280, y1=y0 + 120),
        bubble_bbox=BBox(x0=80, y0=y0, x1=320, y1=y0 + 160),
        text="이 시간에 어디 가?",
        reading_order=0,
    )


def auto_item(r: Region, text: str = TEXT) -> LayoutItem:
    items = plan_layout([r], {r.id: text}, {}, CFG)
    assert len(items) == 1
    return items[0]


def edit(r: Region, **fields: object) -> LayoutEdit:
    return LayoutEdit.model_validate({"region_id": r.id, "anchor": r.bbox.model_dump(), **fields})


def test_style_only_edits_keep_the_geometry() -> None:
    r = region()
    item = auto_item(r)
    styled = overridden(
        item,
        edit(r, color=(200, 0, 0), stroke_px=3, stroke_color=(255, 255, 0), align="left", angle=-8.0),
        r,
        TEXT,
        CFG,
    )
    assert (styled.color, styled.stroke_px, styled.stroke_color, styled.align, styled.angle) == (
        (200, 0, 0),
        3,
        (255, 255, 0),
        "left",
        -8.0,
    )
    assert (styled.box, styled.lines, styled.size_px, styled.font) == (
        item.box,
        item.lines,
        item.size_px,
        item.font,
    )


def letters(lines: list[str]) -> str:
    """The letters and digits of `lines` (spaces and hyphenation dropped)."""
    return "".join(ch for ch in " ".join(lines) if ch.isalnum())


def test_a_fixed_size_refits_the_line_in_the_balloon() -> None:
    r = region()
    item = auto_item(r)
    assert item.size_px > 14
    smaller = overridden(item, edit(r, size_px=14), r, TEXT, CFG)
    assert smaller.size_px == 14 and not smaller.overflow
    assert letters(smaller.lines) == letters([TEXT])  # every word kept, only re-broken
    assert smaller.box.height < item.box.height
    # still centred where the typesetter centred its own block
    assert abs((smaller.box.y0 + smaller.box.y1) - (item.box.y0 + item.box.y1)) <= 2
    assert abs((smaller.box.x0 + smaller.box.x1) - (item.box.x0 + item.box.x1)) <= 2


def test_a_new_box_moves_and_refits_the_lettering() -> None:
    r = region()
    item = auto_item(r)
    box = BBox(x0=400, y0=500, x1=700, y1=560)
    moved = overridden(item, edit(r, box=box), r, TEXT, CFG)
    cx, cy = (moved.box.x0 + moved.box.x1) / 2, (moved.box.y0 + moved.box.y1) / 2
    assert abs(cx - 550) <= 1 and abs(cy - 530) <= 1
    assert moved.box.width <= box.width and moved.box.height <= box.height and not moved.overflow
    assert len(moved.lines) < len(item.lines)  # a wide, low box takes longer lines


def test_explicit_line_breaks_are_kept_and_overflow_is_flagged() -> None:
    r = region()
    item = auto_item(r)
    lines = ["WHERE DO YOU THINK", "YOU'RE GOING?!"]
    set_by_hand = overridden(item, edit(r, lines=lines, size_px=30), r, TEXT, CFG)
    assert set_by_hand.lines == lines and set_by_hand.size_px == 30
    assert set_by_hand.overflow  # two 30 px lines this long do not fit the balloon's ellipse box
    roomy = overridden(item, edit(r, lines=["Hi!"], box=BBox(x0=0, y0=0, x1=400, y1=400)), r, TEXT, CFG)
    assert roomy.lines == ["Hi!"] and not roomy.overflow


def test_font_edit_names_the_font_and_an_unknown_font_fails() -> None:
    r = region()
    item = auto_item(r)
    kalam = overridden(item, edit(r, font="Kalam-Bold.ttf"), r, TEXT, CFG)
    assert kalam.font == "Kalam-Bold.ttf" and kalam.lines
    with pytest.raises(FileNotFoundError):
        overridden(item, edit(r, font="NoSuchFont.ttf"), r, TEXT, CFG)


def test_capitals_stay_capitals_when_refitted() -> None:
    r = region()
    item = auto_item(r, text="STOP RIGHT THERE!")
    refit = overridden(item, edit(r, size_px=20), r, "Stop right there!", CFG)
    assert all(line == line.upper() for line in refit.lines)


def test_apply_hides_matches_by_box_and_counts_orphans() -> None:
    first, second = region("r0001", 100), region("r0002", 400)
    items = plan_layout([first, second], {"r0001": TEXT, "r0002": "Wait!"}, {}, CFG)
    renumbered = second.model_copy(update={"id": "r0007"})  # a re-run renumbered r0002
    moved_items = [items[0], items[1].model_copy(update={"region_id": "r0007"})]
    edits = ChapterEdits(
        layout=[
            edit(first, hidden=True),
            edit(second, color=(1, 2, 3)),  # made on r0002, follows the box to r0007
            LayoutEdit(region_id="r0009", anchor=BBox(x0=900, y0=900, x1=950, y1=950), color=(9, 9, 9)),
        ]
    )
    result, orphans = apply_layout_edits(
        moved_items, edits, [first, renumbered], {"r0001": TEXT, "r0007": "Wait!"}, CFG
    )
    assert [item.region_id for item in result] == ["r0007"] and result[0].color == (1, 2, 3)
    assert orphans == 1
    assert apply_layout_edits(items, ChapterEdits(), [first, second], {}, CFG) == (items, 0)

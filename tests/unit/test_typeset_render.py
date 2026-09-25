"""Unit tests for omniscan.typeset.render (CPU only; real OFL fonts from fonts/)."""

from __future__ import annotations

import numpy as np
import pytest

from omniscan.core.schemas import BBox, LayoutItem
from omniscan.typeset.fonts import fonts_dir
from omniscan.typeset.render import render_item

BOX = BBox(x0=100, y0=200, x1=220, y1=272)  # 120 x 72


def item(**overrides: object) -> LayoutItem:
    """A dialogue LayoutItem with the card's base values, overridden per test."""
    fields: dict[str, object] = {
        "region_id": "r1",
        "font_role": "dialogue",
        "font": "ComicNeue-Bold.ttf",
        "size_px": 32,
        "lines": ["Hello", "world"],
        "box": BOX,
        "align": "center",
        "color": (0, 0, 0),
    }
    fields.update(overrides)
    return LayoutItem(**fields)  # type: ignore[arg-type]


def ink_span(patch_rgba: np.ndarray) -> tuple[int, int, int, int]:
    """(row0, row1, col0, col1) bounds of the patch's alpha > 0 pixels, inclusive."""
    alpha = patch_rgba[..., 3] > 0
    rows, cols = np.where(alpha.any(axis=1))[0], np.where(alpha.any(axis=0))[0]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def test_render_item_geometry_bands_and_no_fringes() -> None:
    patch = render_item(item())
    assert patch is not None
    assert patch.x == BOX.x0 - 2
    assert patch.y == BOX.y0 - 2
    assert patch.rgba.shape == (76, 124, 4)  # box grown by 2 px on every side
    assert patch.rgba.dtype == np.uint8
    alpha = patch.rgba[..., 3]
    assert alpha[2:38].any() and alpha[38:74].any()  # ink in both line bands (pitch 36)
    border = np.ones_like(alpha, dtype=bool)
    border[2:74, 2:122] = False
    assert not alpha[border].any()  # the outer 2-px border is untouched
    ink = alpha > 0
    assert (patch.rgba[ink][:, :3] == 0).all()  # pure fill colour wherever alpha > 0 (no fringes)


def test_render_item_stroke_grows_and_uses_only_two_colours() -> None:
    plain = render_item(item(color=(255, 255, 255)))
    stroked = render_item(item(color=(255, 255, 255), stroke_px=3, stroke_color=(0, 0, 0)))
    assert plain is not None and stroked is not None
    assert stroked.rgba.shape[:2] == (BOX.height + 10, BOX.width + 10)  # box grown by 2*(3+2)
    r0 = ink_span(plain.rgba)
    r3 = ink_span(stroked.rgba)
    assert r3[1] - r3[0] > r0[1] - r0[0] and r3[3] - r3[2] > r0[3] - r0[2]  # ink box is larger
    opaque = stroked.rgba[stroked.rgba[..., 3] == 255][:, :3]
    assert ((opaque == 255).all(axis=1)).any() and ((opaque == 0).all(axis=1)).any()  # white and black
    # anti-aliased fill edges over the opaque stroke are white/black mixes at full alpha: whatever the
    # alpha, the RGB is a mix of only the two colours (gray here) — never a third colour or a dark fringe
    ink = stroked.rgba[..., 3] > 0
    rgb = stroked.rgba[ink][:, :3].astype(int)
    assert (rgb[:, 0] == rgb[:, 1]).all() and (rgb[:, 1] == rgb[:, 2]).all()


@pytest.mark.parametrize("align", ["center", "left", "right"])
def test_render_item_alignment(align: str) -> None:
    wide = BBox(x0=100, y0=200, x1=500, y1=260)  # wider than the text
    patch = render_item(item(lines=["Hello"], box=wide, align=align))  # type: ignore[arg-type]
    assert patch is not None
    cols = np.where((patch.rgba[..., 3] > 0).any(axis=0))[0]
    x0 = patch.x + int(cols.min())
    x1 = patch.x + int(cols.max()) + 1
    if align == "center":
        assert abs((x0 + x1) / 2 - (wide.x0 + wide.x1) / 2) <= 0.1 * 32
    elif align == "left":
        assert abs(x0 - wide.x0) <= 0.15 * 32
    else:
        assert abs(x1 - wide.x1) <= 0.15 * 32


def test_render_item_without_lines_is_none() -> None:
    assert render_item(item(lines=[])) is None


def test_render_item_missing_font_raises() -> None:
    with pytest.raises(FileNotFoundError, match="font not found"):
        render_item(item(font="NoSuchFont.ttf"))


def test_render_item_font_path_override() -> None:
    patch = render_item(item(font="NoSuchFont.ttf"), font_path=fonts_dir() / "ComicNeue-Regular.ttf")
    assert patch is not None
    assert patch.rgba.shape == (76, 124, 4)
    assert (patch.rgba[..., 3] > 0).any()


def test_render_item_rotation_keeps_the_centre_and_the_two_colours() -> None:
    flat = render_item(item(color=(255, 0, 0), stroke_px=3, stroke_color=(0, 0, 255)))
    tilted = render_item(item(color=(255, 0, 0), stroke_px=3, stroke_color=(0, 0, 255), angle=30.0))
    assert flat is not None and tilted is not None
    h, w = tilted.rgba.shape[:2]
    assert h > flat.rgba.shape[0]  # the rotated block needs more height
    centre = (BOX.x0 + BOX.x1) / 2, (BOX.y0 + BOX.y1) / 2
    assert abs(tilted.x + w / 2 - centre[0]) <= 1 and abs(tilted.y + h / 2 - centre[1]) <= 1
    ink = tilted.rgba[..., 3] > 0
    rgb = tilted.rgba[ink][:, :3].astype(int)
    assert (rgb[:, 1] == 0).all()  # only red/blue mixes: no fringe pulled in from transparent pixels
    rows, _, cols, _ = ink_span(tilted.rgba)
    assert rows >= 0 and cols >= 0


def test_render_item_font_may_be_an_absolute_path() -> None:
    patch = render_item(item(font=str(fonts_dir() / "Kalam-Bold.ttf")))
    assert patch is not None and patch.rgba[..., 3].any()

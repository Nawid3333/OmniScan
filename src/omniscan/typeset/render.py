"""Glyph rasteriser: draw one LayoutItem into a small straight-alpha RGBA patch (CPU, PIL)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from omniscan.core.schemas import RGB, LayoutItem
from omniscan.typeset.fonts import font_file, load_font


@dataclass(frozen=True, slots=True)
class GlyphPatch:
    """One rendered layout item: top-left position in strip pixels plus its RGBA pixels."""

    x: int  # top-left in strip pixels (may be negative)
    y: int
    rgba: np.ndarray  # uint8 [h, w, 4], STRAIGHT (non-premultiplied) alpha


def render_item(item: LayoutItem, *, font_path: Path | None = None) -> GlyphPatch | None:
    """Draw `item` into a straight-alpha RGBA patch (box grown by stroke_px + 2 on every side, then
    rotated by `item.angle` around the box centre when set); None without lines."""
    if not item.lines:
        return None
    font = load_font(font_path if font_path is not None else font_file(item.font), item.size_px)
    margin = item.stroke_px + 2
    size = (item.box.width + 2 * margin, item.box.height + 2 * margin)
    fill_mask = Image.new("L", size)
    stroke_mask = Image.new("L", size) if item.stroke_px > 0 else None
    fill_draw = ImageDraw.Draw(fill_mask)
    stroke_draw = ImageDraw.Draw(stroke_mask) if stroke_mask is not None else None
    pitch = item.box.height // len(item.lines)
    ascent, descent = font.getmetrics()
    for i, line in enumerate(item.lines):
        x = _pen_x(item, font.getlength(line), margin)
        y = margin + i * pitch + (pitch - (ascent + descent)) // 2
        fill_draw.text((x, y), line, font=font, fill=255, anchor="la")
        if stroke_draw is not None:
            stroke_draw.text(
                (x, y),
                line,
                font=font,
                fill=255,
                anchor="la",
                stroke_width=item.stroke_px,
                stroke_fill=255,
            )
    x, y = item.box.x0 - margin, item.box.y0 - margin
    if item.angle:
        # rotate the coverage masks, not the coloured layers: interpolating straight-alpha colours would
        # pull the transparent pixels' RGB into the anti-aliased edge
        centre_x, centre_y = x + size[0] / 2, y + size[1] / 2
        fill_mask = fill_mask.rotate(item.angle, resample=Image.Resampling.BICUBIC, expand=True)
        if stroke_mask is not None:
            stroke_mask = stroke_mask.rotate(item.angle, resample=Image.Resampling.BICUBIC, expand=True)
        size = fill_mask.size
        x, y = round(centre_x - size[0] / 2), round(centre_y - size[1] / 2)
    result = _layer(size, item.color, fill_mask)
    if stroke_mask is not None:
        result = Image.alpha_composite(_layer(size, item.stroke_color, stroke_mask), result)
    return GlyphPatch(x=x, y=y, rgba=np.array(result))


def _pen_x(item: LayoutItem, width: float, margin: int) -> float:
    """Pen x for one line of advance `width`, by the item's alignment."""
    if item.align == "center":
        return margin + (item.box.width - width) / 2
    if item.align == "right":
        return margin + item.box.width - width
    return margin


def _layer(size: tuple[int, int], color: RGB, mask: Image.Image) -> Image.Image:
    """A straight-alpha RGBA layer of `color` with alpha `mask` (RGB never bleeds toward black)."""
    layer = Image.new("RGBA", size, (*color, 0))
    layer.putalpha(mask)
    return layer

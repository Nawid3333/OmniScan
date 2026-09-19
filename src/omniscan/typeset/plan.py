"""Typeset planning: target boxes, font roles and colours per region, then the C7a fitting engine.

Pure CPU planning over the layout engine in `typeset/fit.py` — nothing is drawn and no image is written.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from omniscan.core.config import TypesetConfig
from omniscan.core.schemas import RGB, BBox, FontRole, LayoutItem, Region, RegionKind
from omniscan.translate.prompts import translatable
from omniscan.typeset.fit import FontFactory, inscribed_box, layout_region
from omniscan.typeset.fonts import load_font

_ROLE_BY_KIND: dict[RegionKind, FontRole] = {
    "bubble_text": "dialogue",
    "free_text": "free",
    "sfx": "sfx",
}


def luminance(rgb: RGB) -> float:
    """Perceived brightness of an RGB triple (BT.601 luma), 0..255."""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def ellipse_polygon(box: BBox, points: int = 48) -> list[tuple[int, int]]:
    """The ellipse inscribed in `box` sampled at `points` vertices, starting at the right edge
    (clockwise on screen because y points down)."""
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    a, b = box.width / 2, box.height / 2
    return [
        (
            round(cx + a * math.cos(2 * math.pi * i / points)),
            round(cy + b * math.sin(2 * math.pi * i / points)),
        )
        for i in range(points)
    ]


def target_box(region: Region, cfg: TypesetConfig) -> BBox:
    """Where the region's English text is laid out: for bubble text the bubble's inscribed ellipse
    box (unless the original text box is strictly larger), else the text box grown by `free_grow`."""
    if region.kind == "bubble_text" and region.bubble_bbox is not None:
        ell = inscribed_box(region.bubble_bbox, ellipse_polygon(region.bubble_bbox), margin_px=cfg.margin_px)
        if region.bbox.width * region.bbox.height > ell.width * ell.height:
            return region.bbox  # the space the original text used lies inside the bubble
        return ell
    grow_x = round(region.bbox.width * cfg.free_grow)
    grow_y = round(region.bbox.height * cfg.free_grow)
    return BBox(
        x0=max(0, region.bbox.x0 - grow_x),
        y0=max(0, region.bbox.y0 - grow_y),
        x1=region.bbox.x1 + grow_x,
        y1=region.bbox.y1 + grow_y,
    )


def _text_color(region: Region, fill: RGB | None) -> RGB:
    """Text colour: the region's own colour when known, else black on light bubble fills and white
    on dark ones; free text and SFX are always white."""
    if region.text_color is not None:
        return region.text_color
    if region.kind == "bubble_text":
        if fill is None or luminance(fill) >= 128:
            return (0, 0, 0)
        return (255, 255, 255)
    return (255, 255, 255)


def _stroke_px(region: Region, cfg: TypesetConfig) -> int:
    """Outline width for the region kind: none inside bubbles, configured for free text and SFX."""
    if region.kind == "bubble_text":
        return 0
    if region.kind == "sfx":
        return cfg.stroke_sfx_px
    return cfg.stroke_free_px


def plan_layout(
    regions: Sequence[Region],
    lines: Mapping[str, str],
    fills: Mapping[str, RGB],
    cfg: TypesetConfig,
    *,
    font_factory: FontFactory = load_font,
) -> list[LayoutItem]:
    """Fit every translatable region's final English line into its target box (reading order);
    regions with an empty line are skipped."""
    items: list[LayoutItem] = []
    for region in translatable(regions):
        text = lines.get(region.id, "")
        if not text.strip():
            continue
        items.append(
            layout_region(
                region.id,
                text,
                target_box(region, cfg),
                role=_ROLE_BY_KIND[region.kind],
                min_px=cfg.min_px,
                max_px=cfg.max_px,
                line_spacing=cfg.line_spacing,
                color=_text_color(region, fills.get(region.id)),
                stroke_px=_stroke_px(region, cfg),
                stroke_color=region.stroke_color if region.stroke_color is not None else (0, 0, 0),
                font_factory=font_factory,
            )
        )
    return items

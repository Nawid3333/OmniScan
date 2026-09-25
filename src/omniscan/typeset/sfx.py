"""Sound-effect lettering: the English effect redrawn in the original's style, or a small translation.

Official releases replace a sound effect with an English one drawn to match it: the same size and
placement, the same colours and outline, the same tilt, a typeface of similar weight (big brushy
letters for a crash, thin ones for a rustle). `layout_sfx` does that from what the OCR stage measured
on the original (`Region.text_color`, `stroke_color`, `angle`, `weight`): the effect fills its box,
rotated like the original, stacked letter by letter when the original ran down a tall narrow box.
When the original stays on the page (sfx.mode = "subtitle", or it could not be erased)
`layout_sfx_subtitle` sets a small outlined translation just below it instead, as many manga
releases do for effects woven into the art.
"""

from __future__ import annotations

import math
from pathlib import Path

from omniscan.core.config import TypesetConfig
from omniscan.core.schemas import RGB, BBox, LayoutItem, Region
from omniscan.typeset.fit import FontFactory, Shape, fit_shape, line_height, measurer
from omniscan.typeset.fonts import SFX_FONTS, SfxWeight, font_file, layout_font_name, load_font

_HEAVY_WEIGHT = 0.16  # stroke width / letter height of the original at or above which it is "heavy" ...
_BOLD_WEIGHT = 0.085  # ... "bold" at or above this, "light" below (calibrated on Hangul lettering)
_STROKE_SHARE = 0.08  # outline width as a share of the letter size (at least stroke_sfx_px)
_WORD_SPACING = 0.95  # line pitch of multi-line effects
_STACK_SPACING = 0.9  # letter pitch of stacked (vertical) effects
_STACK_ASPECT = 1.6  # a box this much taller than wide is lettered top to bottom ...
_STACK_MAX_LETTERS = 8  # ... when the effect has at most this many letters ...
_STACK_MAX_ANGLE = 20.0  # ... and is not tilted
_SUBTITLE_SHARE = 0.75  # subtitle size as a share of the chapter's dialogue size
_SUBTITLE_MAX_PX = 24
_SUBTITLE_GAP_PX = 4  # gap between the original effect and its subtitle
_SUBTITLE_STROKE_PX = 3
_MIN_OUTLINE_CONTRAST = 96  # an outline must differ from its fill by this much (largest channel)


def weight_class(weight: float | None) -> SfxWeight:
    """The typeface weight matching the original's measured stroke weight ("bold" when unknown)."""
    if weight is None:
        return "bold"
    if weight >= _HEAVY_WEIGHT:
        return "heavy"
    return "bold" if weight >= _BOLD_WEIGHT else "light"


def sfx_font(region: Region, cfg: TypesetConfig) -> Path:
    """`cfg.font_sfx` when set, else the preset typeface for the original's stroke weight."""
    return font_file(cfg.font_sfx or SFX_FONTS[weight_class(region.weight)])


def contrast(rgb: RGB) -> RGB:
    """Black for a light colour, white for a dark one (BT.601 luma split at 128)."""
    return (0, 0, 0) if 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] >= 128 else (255, 255, 255)


def outline_for(fill: RGB, measured: RGB | None) -> RGB:
    """The measured outline colour when it stands out from the fill, else black or white by contrast
    (an outline too close to its fill would blur the letters instead of separating them from the art)."""
    if (
        measured is not None
        and max(abs(a - b) for a, b in zip(fill, measured, strict=True)) >= _MIN_OUTLINE_CONTRAST
    ):
        return measured
    return contrast(fill)


def sfx_colors(region: Region) -> tuple[RGB, RGB]:
    """(fill, outline): the original's measured colours; white fill and a contrasting outline otherwise."""
    fill = region.text_color if region.text_color is not None else (255, 255, 255)
    return fill, outline_for(fill, region.stroke_color)


def _grown(box: BBox, share: float) -> BBox:
    """`box` grown by `share` of its size on every side (never below 0)."""
    dx, dy = round(box.width * share), round(box.height * share)
    return BBox(x0=max(0, box.x0 - dx), y0=max(0, box.y0 - dy), x1=box.x1 + dx, y1=box.y1 + dy)


def _arrangements(text: str, target: BBox, angle: float) -> list[tuple[list[str], float]]:
    """Candidate (lines, line spacing): one line; the words on separate lines; letters stacked."""
    words = text.split()
    options: list[tuple[list[str], float]] = [([" ".join(words)], _WORD_SPACING)]
    if len(words) > 1:
        options.append((words, _WORD_SPACING))
    letters = [c for c in "".join(words) if not c.isspace()]
    tall = target.height >= _STACK_ASPECT * target.width
    if tall and len(letters) <= _STACK_MAX_LETTERS and abs(angle) <= _STACK_MAX_ANGLE:
        options.append((letters, _STACK_SPACING))
    return options


def stroke_for(size_px: int, cfg: TypesetConfig) -> int:
    """Outline width of an effect lettered at `size_px`."""
    return max(cfg.stroke_sfx_px, round(_STROKE_SHARE * size_px))


def _block(
    lines: list[str], spacing: float, size: int, font_path: Path, factory: FontFactory
) -> tuple[float, int]:
    """(width, height) of the lettered block at `size`, without its outline."""
    font = factory(font_path, size)
    return max(font.getlength(line) for line in lines), len(lines) * line_height(size, spacing)


def _fits(width: float, height: float, target: BBox, angle: float) -> bool:
    """Whether a width x height block rotated by `angle` degrees fits inside `target`."""
    c, s = abs(math.cos(math.radians(angle))), abs(math.sin(math.radians(angle)))
    return width * c + height * s <= target.width and width * s + height * c <= target.height


def layout_sfx(
    region: Region, text: str, cfg: TypesetConfig, *, font_factory: FontFactory = load_font
) -> LayoutItem:
    """The English effect filling the original's box (grown by `cfg.free_grow`) in the original's style."""
    path = sfx_font(region, cfg)
    factory = measurer(font_factory)
    target = _grown(region.bbox, cfg.free_grow)
    angle = region.angle
    best: tuple[int, list[str], float] | None = None
    for lines, spacing in _arrangements(text.upper(), target, angle):

        def fits(size: int, lines: list[str] = lines, spacing: float = spacing) -> bool:
            width, height = _block(lines, spacing, size, path, factory)
            outline = 2 * stroke_for(size, cfg)
            return _fits(width + outline, height + outline, target, angle)

        lo, hi = cfg.min_px, max(cfg.min_px, cfg.sfx_max_px)
        if not fits(lo):
            continue
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid - 1
        if best is None or lo > best[0]:
            best = (lo, lines, spacing)
    overflow = best is None
    size, lines, spacing = best if best is not None else (cfg.min_px, [text.upper()], _WORD_SPACING)
    width, height = _block(lines, spacing, size, path, factory)
    cx, cy = (target.x0 + target.x1) / 2, (target.y0 + target.y1) / 2
    x0, y0 = round(cx - width / 2), round(cy - height / 2)
    fill, outline_color = sfx_colors(region)
    return LayoutItem(
        region_id=region.id,
        font_role="sfx",
        font=layout_font_name(path),
        size_px=size,
        lines=lines,
        box=BBox(x0=x0, y0=y0, x1=x0 + math.ceil(width), y1=y0 + height),
        color=fill,
        stroke_px=stroke_for(size, cfg),
        stroke_color=outline_color,
        overflow=overflow,
        angle=angle,
    )


def layout_sfx_subtitle(
    region: Region,
    text: str,
    cfg: TypesetConfig,
    font_path: Path,
    *,
    body_px: int | None,
    font_factory: FontFactory = load_font,
) -> LayoutItem:
    """A small outlined translation centred just below the original effect (which stays on the page)."""
    size_cap = min(_SUBTITLE_MAX_PX, round(body_px * _SUBTITLE_SHARE)) if body_px else _SUBTITLE_MAX_PX
    size_cap = max(cfg.min_px, size_cap)
    width = max(region.bbox.width, 8 * size_cap)
    cx = (region.bbox.x0 + region.bbox.x1) / 2
    top = region.bbox.y1 + _SUBTITLE_GAP_PX
    area = BBox(
        x0=max(0, round(cx - width / 2)),
        y0=top,
        x1=round(cx + width / 2),
        y1=top + 3 * line_height(size_cap, cfg.line_spacing),
    )
    fit = fit_shape(
        text.upper(),
        Shape("rect", area),
        font_path,
        min_px=cfg.min_px,
        max_px=size_cap,
        line_spacing=cfg.line_spacing,
        hyphenate=cfg.hyphenate,
        font_factory=font_factory,
    )
    x0 = round(cx - fit.width / 2)
    return LayoutItem(
        region_id=region.id,
        font_role="free",
        font=layout_font_name(font_path),
        size_px=fit.size_px,
        lines=fit.lines,
        box=BBox(x0=max(0, x0), y0=top, x1=max(0, x0) + fit.width, y1=top + fit.height),
        color=(255, 255, 255),
        stroke_px=_SUBTITLE_STROKE_PX,
        stroke_color=(0, 0, 0),
        overflow=fit.overflow,
    )

"""Typeset planning: lettering shape, style, font, size and colours per region, then the layout engine.

Pure CPU planning over `typeset/fit.py` (balloon-shaped lines) and `typeset/sfx.py` (sound effects) —
nothing is drawn and no image is written. Like an official release, sizes are consistent across the
chapter: dialogue is set no larger than `size_spread` x the chapter's typical balloon size, so a short
"Huh?" in a roomy balloon does not shout at 48 px next to 20 px dialogue.
"""

from __future__ import annotations

import statistics
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from omniscan.core.config import SfxConfig, TypesetConfig
from omniscan.core.schemas import RGB, BBox, FontRole, LayoutItem, Region, RegionKind
from omniscan.translate.prompts import translatable
from omniscan.typeset.fit import Fit, FontFactory, Shape, fit_shape
from omniscan.typeset.fonts import (
    STYLE_FONTS,
    STYLE_UPPERCASE,
    LetteringStyle,
    font_file,
    layout_font_name,
    load_font,
)
from omniscan.typeset.sfx import layout_sfx, layout_sfx_subtitle, outline_for

_ROLE_BY_KIND: dict[RegionKind, FontRole] = {
    "bubble_text": "dialogue",
    "free_text": "free",
    "sfx": "sfx",
}
_MIN_TYPICAL_SAMPLES = 3  # fewer fitted dialogue balloons than this: no chapter-wide size limit
_SHOUT_MIN_LETTERS = 4  # an all-capitals line with at least this many letters is shouted


def luminance(rgb: RGB) -> float:
    """Perceived brightness of an RGB triple (BT.601 luma), 0..255."""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def lettering_style(region: Region, cfg: TypesetConfig) -> LetteringStyle:
    """The configured style; for "auto" manga lettering on Japanese sources, webtoon lettering otherwise."""
    if cfg.style != "auto":
        return cfg.style
    return "manga" if region.lang == "ja" else "webtoon"


def use_uppercase(style: LetteringStyle, cfg: TypesetConfig) -> bool:
    """Whether dialogue is lettered in capitals: forced by `cfg.uppercase`, else the style's convention."""
    if cfg.uppercase == "auto":
        return STYLE_UPPERCASE[style]
    return cfg.uppercase == "always"


def is_shouted(text: str) -> bool:
    """A shouted line: written in capitals (at least `_SHOUT_MIN_LETTERS` letters) or ending in "!!"."""
    letters = [c for c in text if c.isalpha()]
    capitals = len(letters) >= _SHOUT_MIN_LETTERS and all(c.isupper() for c in letters)
    return capitals or text.rstrip().rstrip("\"'”’").endswith("!!")


def role_for(region: Region, text: str) -> FontRole:
    """Font role of a region's English line: shouted dialogue gets the shout role."""
    role = _ROLE_BY_KIND[region.kind]
    return "shout" if role == "dialogue" and is_shouted(text) else role


def role_font(role: FontRole, style: LetteringStyle, cfg: TypesetConfig) -> Path:
    """The user's font for the role when configured (`cfg.font_<role>`), else the style's preset."""
    override: str = getattr(cfg, f"font_{role}")
    return font_file(override or STYLE_FONTS[style][role])


def _centre(box: BBox) -> tuple[float, float]:
    """Centre point of a box."""
    return (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2


def lettering_shape(region: Region, cfg: TypesetConfig) -> Shape:
    """Where the English lettering goes.

    Bubble text: an ellipse inside the bubble, centred halfway between the bubble's and the original
    text's centres (a tail stretches the bubble box, the artist's text sits in the balloon's body),
    kept `bubble_padding` of the bubble size (at least `margin_px`) from its edge — unless the original
    text box is wider and taller than that ellipse's box (a rectangular balloon or caption), then the
    text box itself. Everything else: the text box grown by `free_grow` on every side.
    """
    if region.kind == "bubble_text" and region.bubble_bbox is not None:
        bubble = region.bubble_bbox
        (bx, by), (tx, ty) = _centre(bubble), _centre(region.bbox)
        cx, cy = (bx + tx) / 2, (by + ty) / 2
        pad_x = max(cfg.margin_px, cfg.bubble_padding * bubble.width)
        pad_y = max(cfg.margin_px, cfg.bubble_padding * bubble.height)
        a = min(cx - bubble.x0, bubble.x1 - cx) - pad_x
        b = min(cy - bubble.y0, bubble.y1 - cy) - pad_y
        if a > 0 and b > 0:
            ellipse = BBox(x0=round(cx - a), y0=round(cy - b), x1=round(cx + a), y1=round(cy + b))
            if not (region.bbox.width >= ellipse.width and region.bbox.height >= ellipse.height):
                return Shape("ellipse", ellipse)
        return Shape("rect", region.bbox)
    grow_x = round(region.bbox.width * cfg.free_grow)
    grow_y = round(region.bbox.height * cfg.free_grow)
    return Shape(
        "rect",
        BBox(
            x0=max(0, region.bbox.x0 - grow_x),
            y0=max(0, region.bbox.y0 - grow_y),
            x1=region.bbox.x1 + grow_x,
            y1=region.bbox.y1 + grow_y,
        ),
    )


def _text_color(region: Region, fill: RGB | None) -> RGB:
    """Text colour: the region's own colour when known, else black on light bubble fills and white
    on dark ones; free text is white."""
    if region.text_color is not None:
        return region.text_color
    if region.kind == "bubble_text":
        if fill is None or luminance(fill) >= 128:
            return (0, 0, 0)
        return (255, 255, 255)
    return (255, 255, 255)


def _stroke_px(region: Region, cfg: TypesetConfig) -> int:
    """Outline width for the region kind: none inside bubbles, `stroke_free_px` for free text."""
    return 0 if region.kind == "bubble_text" else cfg.stroke_free_px


@dataclass(slots=True)
class _Job:
    """One dialogue/free-text region on its way to a LayoutItem."""

    region: Region
    text: str
    role: FontRole
    font: Path
    shape: Shape
    fit: Fit


def typical_size(fits: Sequence[Fit]) -> int | None:
    """The chapter's typical dialogue size: the median largest size of the balloons that fit; None with
    fewer than `_MIN_TYPICAL_SAMPLES` of them."""
    sizes = [fit.best_px for fit in fits if not fit.overflow]
    if len(sizes) < _MIN_TYPICAL_SAMPLES:
        return None
    return round(statistics.median(sizes))


def plan_layout(
    regions: Sequence[Region],
    lines: Mapping[str, str],
    fills: Mapping[str, RGB],
    cfg: TypesetConfig,
    *,
    sfx: SfxConfig | None = None,
    erased: Collection[str] | None = None,
    font_factory: FontFactory = load_font,
) -> list[LayoutItem]:
    """Letter every translatable region's final English line (reading order); empty lines are skipped.

    Dialogue and free text are shaped to their balloon or box, then capped chapter-wide by
    `size_spread` (`shout_spread` for shouted lines) x `typical_size`. Sound effects follow `sfx.mode`:
    "replace" redraws the effect in the original's style where it was erased (`erased` holds the ids of
    the cleaned regions; None means all were), "subtitle" — and "replace" where the original stayed —
    set a small translation below it, "keep" leaves them untranslated.
    """
    sfx_cfg = sfx if sfx is not None else SfxConfig()
    jobs: list[_Job] = []
    effects: list[tuple[Region, str]] = []
    for region in translatable(regions):
        text = " ".join(lines.get(region.id, "").split())
        if not text:
            continue
        if region.kind == "sfx":
            effects.append((region, text))
            continue
        style = lettering_style(region, cfg)
        role = role_for(region, text)
        lettered = text.upper() if use_uppercase(style, cfg) else text
        font = role_font(role, style, cfg)
        shape = lettering_shape(region, cfg)
        fit = _fit(lettered, shape, font, cfg, None, font_factory)
        jobs.append(_Job(region, lettered, role, font, shape, fit))
    typical = typical_size([job.fit for job in jobs if job.role == "dialogue"])
    items: dict[str, LayoutItem] = {}
    for job in jobs:
        cap = _size_cap(job.role, typical, cfg)
        if cap is not None and job.fit.size_px > cap:
            job.fit = _fit(job.text, job.shape, job.font, cfg, cap, font_factory)
        items[job.region.id] = _item(job, fills.get(job.region.id), cfg)
    for region, text in effects:
        if sfx_cfg.mode == "keep":
            continue
        if sfx_cfg.mode == "replace" and (erased is None or region.id in erased):
            items[region.id] = layout_sfx(region, text, cfg, font_factory=font_factory)
        else:
            font = role_font("free", lettering_style(region, cfg), cfg)
            items[region.id] = layout_sfx_subtitle(
                region, text, cfg, font, body_px=typical, font_factory=font_factory
            )
    return [items[region.id] for region in translatable(regions) if region.id in items]


def _size_cap(role: FontRole, typical: int | None, cfg: TypesetConfig) -> int | None:
    """Largest size a line of `role` may take given the chapter's typical size (None: no limit)."""
    if typical is None or cfg.size_spread <= 0:
        return None
    spread = cfg.shout_spread if role == "shout" else cfg.size_spread
    return max(cfg.min_px, round(typical * spread))


def _fit(
    text: str, shape: Shape, font: Path, cfg: TypesetConfig, cap: int | None, font_factory: FontFactory
) -> Fit:
    """`fit_shape` with the configured sizes, spacing and hyphenation."""
    return fit_shape(
        text,
        shape,
        font,
        min_px=cfg.min_px,
        max_px=cfg.max_px,
        size_cap=cap,
        line_spacing=cfg.line_spacing,
        hyphenate=cfg.hyphenate,
        font_factory=font_factory,
    )


def _item(job: _Job, fill: RGB | None, cfg: TypesetConfig) -> LayoutItem:
    """The render-ready item: the fitted block centred in its shape."""
    cx, cy = _centre(job.shape.box)
    x0 = round(cx - job.fit.width / 2)
    y0 = round(cy - job.fit.height / 2)
    region = job.region
    color = _text_color(region, fill)
    return LayoutItem(
        region_id=region.id,
        font_role=job.role,
        font=layout_font_name(job.font),
        size_px=job.fit.size_px,
        lines=job.fit.lines,
        box=BBox(x0=x0, y0=y0, x1=x0 + job.fit.width, y1=y0 + job.fit.height),
        color=color,
        stroke_px=_stroke_px(region, cfg),
        stroke_color=outline_for(color, region.stroke_color),
        overflow=job.fit.overflow,
    )

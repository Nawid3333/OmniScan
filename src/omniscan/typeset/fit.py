"""Typeset layout engine: pick the largest font size whose greedily wrapped text fits a target box.

Pure metric math over PIL font advances — nothing is drawn, no image is written.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from omniscan.core.schemas import RGB, BBox, FontRole, LayoutItem
from omniscan.typeset.fonts import default_font_path, load_font


class Measurable(Protocol):
    """Anything that can measure a single line's advance width in pixels (PIL fonts, test fakes)."""

    def getlength(self, text: str) -> float: ...


FontFactory = Callable[[Path, int], Measurable]


def wrap_words(text: str, font: Measurable, max_width: float) -> tuple[list[str], bool]:
    """Greedily word-wrap `text` to lines of advance width <= max_width; returns (lines, too_wide)
    where too_wide marks that some single word alone is wider than max_width."""
    words = text.split()
    if not words:
        return [], False
    lines: list[str] = []
    line = ""
    too_wide = False
    for word in words:
        if not line:
            if font.getlength(word) <= max_width:
                line = word
            else:  # a single word wider than the box: its own line, no hyphenation
                lines.append(word)
                too_wide = True
            continue
        candidate = f"{line} {word}"
        if font.getlength(candidate) <= max_width:
            line = candidate
            continue
        lines.append(line)
        if font.getlength(word) <= max_width:
            line = word
        else:
            lines.append(word)
            too_wide = True
            line = ""
    if line:
        lines.append(line)
    return lines, too_wide


def line_height(size_px: int, line_spacing: float) -> int:
    """Line pitch in pixels: size times spacing, rounded, never below 1."""
    return max(1, round(size_px * line_spacing))


@dataclass(frozen=True, slots=True)
class Fit:
    """One layout decision: the chosen size, its wrapped lines and whether they fit the box."""

    size_px: int
    lines: list[str]
    width: int  # ceil of the widest line's advance; 0 for no lines
    height: int  # len(lines) * line_height
    overflow: bool


def fit_text(
    text: str,
    max_w: int,
    max_h: int,
    font_path: Path,
    *,
    min_px: int = 14,
    max_px: int = 48,
    line_spacing: float = 1.15,
    font_factory: FontFactory = load_font,
) -> Fit:
    """Largest font size in [min_px, max_px] whose word-wrapped text fits (max_w, max_h); the min
    size with overflow=True when nothing fits."""
    if min_px < 1 or max_px < min_px:
        raise ValueError(f"need 1 <= min_px <= max_px, got {min_px}..{max_px}")
    if max_w < 1 or max_h < 1:
        raise ValueError(f"need max_w >= 1 and max_h >= 1, got {max_w}x{max_h}")
    if line_spacing <= 0:
        raise ValueError(f"line_spacing must be > 0, got {line_spacing}")
    normalized = " ".join(text.split())
    if not normalized:
        return Fit(size_px=max_px, lines=[], width=0, height=0, overflow=False)
    fallback: Fit | None = None
    for size in range(max_px, min_px - 1, -1):
        font = font_factory(font_path, size)
        lines, too_wide = wrap_words(normalized, font, max_w)
        height = len(lines) * line_height(size, line_spacing)
        width = math.ceil(max((font.getlength(line) for line in lines), default=0.0))
        result = Fit(size_px=size, lines=lines, width=width, height=height, overflow=False)
        if not too_wide and height <= max_h:
            return result
        if size == min_px:
            fallback = result
    assert fallback is not None  # the loop always reaches min_px
    return dataclasses.replace(fallback, overflow=True)


def _on_segment(px: float, py: float, xi: float, yi: float, xj: float, yj: float) -> bool:
    """Whether point (px, py) lies on the closed segment (xi, yi)-(xj, yj)."""
    cross = (xj - xi) * (py - yi) - (yj - yi) * (px - xi)
    if abs(cross) > 1e-9:
        return False
    return min(xi, xj) <= px <= max(xi, xj) and min(yi, yj) <= py <= max(yi, yj)


def _point_inside(px: float, py: float, polygon: Sequence[tuple[int, int]]) -> bool:
    """Even-odd point-in-polygon; points exactly on the outline count as outside."""
    n = len(polygon)
    j = n - 1
    for i in range(n):  # boundary first: on the outline is not inside
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if _on_segment(px, py, xi, yi, xj, yj):
            return False
        j = i
    inside = False
    j = n - 1
    for i in range(n):  # ray cast to +x, half-open in y so vertices are handled consistently
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > py) != (yj > py):
            x_at = xi + (py - yi) * (xj - xi) / (yj - yi)
            if x_at > px:
                inside = not inside
        j = i
    return inside


def _boundary_samples(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    """24 sample points on a rectangle's boundary: the four corners plus 5 evenly spaced interior
    points per side."""
    points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for ax, ay, bx, by in (
        (x0, y0, x1, y0),
        (x1, y0, x1, y1),
        (x1, y1, x0, y1),
        (x0, y1, x0, y0),
    ):
        for k in range(1, 6):
            t = k / 6
            points.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    return points


def _degenerate_centre(x0: int, x1: int, y0: int, y1: int) -> tuple[int, int, int, int]:
    """Collapse each axis that rounded to a negative span onto its midpoint (a zero-size box)."""
    if x1 < x0:
        x0 = x1 = (x0 + x1) // 2
    if y1 < y0:
        y0 = y1 = (y0 + y1) // 2
    return x0, y0, x1, y1


def inscribed_box(bubble: BBox, polygon: Sequence[tuple[int, int]] | None, *, margin_px: int = 6) -> BBox:
    """Largest bubble-centred rectangle (scaled 1.00 down to 0.20) whose 24 boundary samples lie
    inside the bubble outline, shrunk by margin_px; the whole box minus margin when polygon is None."""
    if polygon is None:
        x0, y0, x1, y1 = _degenerate_centre(
            bubble.x0 + margin_px,
            bubble.x1 - margin_px,
            bubble.y0 + margin_px,
            bubble.y1 - margin_px,
        )
        return BBox(x0=x0, y0=y0, x1=x1, y1=y1)
    w, h = bubble.width, bubble.height
    cx, cy = (bubble.x0 + bubble.x1) / 2, (bubble.y0 + bubble.y1) / 2
    accepted: tuple[float, float, float, float] | None = None
    for step in range(100, 19, -1):
        s = step / 100
        rect = (cx - s * w / 2, cy - s * h / 2, cx + s * w / 2, cy + s * h / 2)
        if all(_point_inside(px, py, polygon) for px, py in _boundary_samples(*rect)):
            accepted = rect
            break
    if accepted is None:
        accepted = (cx - 0.10 * w, cy - 0.10 * h, cx + 0.10 * w, cy + 0.10 * h)
    x0, y0, x1, y1 = _degenerate_centre(
        round(accepted[0] + margin_px),
        round(accepted[2] - margin_px),
        round(accepted[1] + margin_px),
        round(accepted[3] - margin_px),
    )
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def layout_region(
    region_id: str,
    text: str,
    target: BBox,
    *,
    role: FontRole = "dialogue",
    font_path: Path | None = None,
    min_px: int = 14,
    max_px: int = 48,
    line_spacing: float = 1.15,
    align: Literal["center", "left", "right"] = "center",
    color: RGB = (0, 0, 0),
    stroke_px: int = 0,
    stroke_color: RGB = (255, 255, 255),
    font_factory: FontFactory = load_font,
) -> LayoutItem:
    """Fit `text` into `target` with the role's font and return the render-ready LayoutItem (the
    text box is centred in target; it may exceed target when overflow is true)."""
    path = font_path if font_path is not None else default_font_path(role)
    fit = fit_text(
        text,
        target.width,
        target.height,
        path,
        min_px=min_px,
        max_px=max_px,
        line_spacing=line_spacing,
        font_factory=font_factory,
    )
    x0 = target.x0 + (target.width - fit.width) // 2
    y0 = target.y0 + (target.height - fit.height) // 2
    box = BBox(x0=x0, y0=y0, x1=x0 + fit.width, y1=y0 + fit.height)
    return LayoutItem(
        region_id=region_id,
        font_role=role,
        font=path.name,
        size_px=fit.size_px,
        lines=fit.lines,
        box=box,
        align=align,
        color=color,
        stroke_px=stroke_px,
        stroke_color=stroke_color,
        overflow=fit.overflow,
    )

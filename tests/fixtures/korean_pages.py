"""Synthetic Korean webtoon pages with exact ground truth (card X1).

Every page is generated deterministically from one seed: a background, speech bubbles and text
(free-standing and SFX), plus the full truth about what was drawn — per-line boxes and strings, bubble
outlines, a pixel-exact text mask and the same page without any text (the ideal inpainting result).
Fonts come from `fonts/` (SIL OFL, committed) so pages render identically on every OS. CPU only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from omniscan.core.schemas import RGB, BBox, OcrLine, Region, RegionsArtifact

FONTS_DIR: Path = Path(__file__).resolve().parents[2] / "fonts"

KOREAN_LINES: tuple[str, ...] = (
    "괜찮아요? 던전이 열렸어!",
    "민준 형, 빨리 도망가요!",
    "그럴 리가 없어...",
    "저 녀석은 S급 헌터야.",
    "내가 널 지켜줄게.",
    "뭐라고?! 다시 말해 봐!",
    "게이트가 닫히기 전에 나가야 해.",
    "오랜만이야, 성진아.",
    "지금 무슨 일이 일어난 거지?",
    "조용히 해. 누가 오고 있어.",
    "이건 시작에 불과해.",
    "약속할게. 반드시 돌아올 거야.",
)

SFX_LINES: tuple[str, ...] = ("쾅!", "두근두근", "쿵!", "슈웅")

RegionKind = Literal["bubble_text", "free_text", "sfx"]
Shape = Literal["ellipse", "rounded"]
Background = Literal["gradient", "noise", "flat"]

_MARGIN = 40  # min distance from every region box to the page edges
_GAP = 40  # min distance between neighbouring region boxes
_INSET = _GAP // 2  # kept inside each slot on top of _MARGIN, so both margins above hold
_POLYGON_POINTS = 48  # ellipse outline samples
_MIN_FONT, _MAX_FONT = 26, 34
_BUBBLE_PAD = 14  # min ink-to-bubble-edge slack to aim for (test requires >= 8 on every side)
_TRUTH = "truth"


def font_path(name: str) -> Path:
    """Path of a font file shipped in `fonts/`."""
    path = FONTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"font not found: {name} (looked in {FONTS_DIR})")
    return path


@dataclass(frozen=True, slots=True)
class LineTruth:
    """One rendered text line: tight ink box in page pixels plus the string."""

    bbox: BBox
    text: str


@dataclass(frozen=True, slots=True)
class RegionTruth:
    """One text region (bubble text, free text or SFX) and everything drawn for it."""

    kind: RegionKind
    lines: tuple[LineTruth, ...]  # top to bottom
    bbox: BBox  # union of the line boxes
    bubble_bbox: BBox | None  # bounding box of the bubble shape (None for free_text / sfx)
    bubble_polygon: tuple[tuple[int, int], ...] | None  # bubble outline, >= 12 points, clockwise
    fill: RGB  # bubble fill colour (free_text / sfx: mean colour of the background under bbox)
    text_color: RGB

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass(frozen=True, slots=True)
class KoreanPage:
    """A generated page with text, the same page without text, the text mask and the truth."""

    image: Image.Image  # RGB, with text
    clean: Image.Image  # RGB, identical but without ANY text
    text_mask: np.ndarray  # bool [H, W]: True exactly where image differs from clean
    regions: tuple[RegionTruth, ...]  # reading order: rows top to bottom, left to right in a row


# ---------------------------------------------------------------- low-level drawing helpers


def _break_lines(text: str, font: ImageFont.FreeTypeFont, limit: float) -> tuple[str, ...]:
    """Greedy break at spaces so every line's advance width fits `limit`; long words stay alone."""
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else f"{current} {word}"
        if not current or font.getlength(candidate) <= limit:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return tuple(lines)


def _ink_bbox(layer: Image.Image) -> tuple[int, int, int, int] | None:
    """Bounding box of all pixels with any alpha (the ink) on `layer`."""
    alpha = layer.getchannel("A")
    return alpha.point(lambda v: 255 if v > 0 else 0).getbbox()  # type: ignore[operator]


def _ink_box(text: str, font: ImageFont.FreeTypeFont, stroke: int) -> tuple[int, int, int, int] | None:
    """Ink box of `text` relative to an anchor="mm" point at (0, 0); None for blank text."""
    pad = int(font.getlength(text)) + 4 * stroke + 24  # covers half the advance plus stroke on all sides
    layer = Image.new("RGBA", (2 * pad, 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(
        (pad, pad), text, font=font, anchor="mm", fill=(0, 0, 0), stroke_width=stroke, stroke_fill=(0, 0, 0)
    )
    box = _ink_bbox(layer)
    if box is None:
        return None
    return box[0] - pad, box[1] - pad, box[2] - pad, box[3] - pad


def _draw_block(
    work: Image.Image,
    lines: tuple[str, ...],
    font: ImageFont.FreeTypeFont,
    centre: tuple[float, float],
    fill: RGB,
    stroke: int,
    stroke_fill: RGB,
) -> tuple[LineTruth, ...]:
    """Draw centred text lines onto `work` (in place); return their truth boxes."""
    cx, cy = centre
    line_h = 1.25 * font.size
    truths: list[LineTruth] = []
    for i, text in enumerate(lines):
        y = cy + (i - (len(lines) - 1) / 2) * line_h
        layer = Image.new("RGBA", work.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).text(
            (cx, y),
            text,
            font=font,
            anchor="mm",
            fill=fill,
            stroke_width=stroke,
            stroke_fill=stroke_fill,
        )
        box = _ink_bbox(layer)
        if box is not None:
            truths.append(LineTruth(bbox=BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]), text=text))
        work.alpha_composite(layer)
    return tuple(truths)


def _background(rng: np.random.Generator, width: int, height: int, kind: Background) -> Image.Image:
    """Background layer covering the whole page (drawn before bubbles and text)."""
    if kind == "flat":
        arr = np.broadcast_to(rng.integers(185, 245, 3), (height, width, 3)).astype(np.uint8)
    elif kind == "gradient":
        top = rng.integers(110, 245, 3).astype(np.float64)
        bottom = rng.integers(110, 245, 3).astype(np.float64)
        t = np.linspace(0.0, 1.0, height)[:, None, None]
        base = top[None, None, :] * (1.0 - t) + bottom[None, None, :] * t
        base = np.repeat(base, width, axis=1)
        arr = np.clip(base + rng.normal(0.0, 6.0, (height, width, 3)), 0.0, 255.0).astype(np.uint8)
    else:  # noise: blocks of 8x8 pixels, each one random colour
        blocks = rng.integers(0, 256, ((height + 7) // 8, (width + 7) // 8, 3), dtype=np.uint8)
        arr = np.repeat(np.repeat(blocks, 8, axis=0), 8, axis=1)[:height, :width]
    return Image.fromarray(arr, "RGB")


def _ellipse_polygon(bbox: BBox) -> tuple[tuple[int, int], ...]:
    """48 outline points of the ellipse inscribed in `bbox`, clockwise from the top."""
    cx = (bbox.x0 + bbox.x1) / 2
    cy = (bbox.y0 + bbox.y1) / 2
    rx = (bbox.x1 - bbox.x0) / 2
    ry = (bbox.y1 - bbox.y0) / 2
    return tuple(
        (round(cx + rx * np.cos(a)), round(cy + ry * np.sin(a)))
        for a in np.linspace(-np.pi / 2, 3 * np.pi / 2, _POLYGON_POINTS, endpoint=False)
    )


def _rounded_polygon(bbox: BBox, radius: int) -> tuple[tuple[int, int], ...]:
    """Outline of a rounded rectangle: the 4 corner arcs (4 points each), clockwise."""
    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    points: list[tuple[int, int]] = []
    for cx, cy, start in (
        (x1 - radius, y0 + radius, -90.0),  # top-right corner arc
        (x1 - radius, y1 - radius, 0.0),  # bottom-right
        (x0 + radius, y1 - radius, 90.0),  # bottom-left
        (x0 + radius, y0 + radius, 180.0),  # top-left
    ):
        for a in np.linspace(np.radians(start), np.radians(start + 90.0), 4):
            points.append((round(cx + radius * np.cos(a)), round(cy + radius * np.sin(a))))
    return tuple(points)


def _saturated(rng: np.random.Generator) -> RGB:
    """Random fully saturated colour (one channel pinned high, one low)."""
    channels = [int(rng.integers(210, 256)), int(rng.integers(0, 60)), int(rng.integers(40, 200))]
    rng.shuffle(channels)
    return channels[0], channels[1], channels[2]


# ---------------------------------------------------------------- layout


@dataclass(frozen=True, slots=True)
class _Slot:
    """Where a region may be drawn: one column of a row slot (the region box stays inside)."""

    x0: int
    x1: int
    y0: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass(frozen=True, slots=True)
class _Plan:
    """Everything decided for one region before any ink is drawn."""

    kind: RegionKind
    text: str
    slot: _Slot
    font_size: int
    lines: tuple[str, ...]
    centre: tuple[float, float]
    bubble: tuple[int, int, Shape, bool, int] | None  # (w, h, shape, dark, corner radius)


def _slots(rng: np.random.Generator, width: int, height: int, kinds: list[RegionKind]) -> list[_Slot]:
    """One slot per region, in reading order: equal row slots; pairs share a row side by side."""
    n = len(kinds)
    pairable = sum(1 for k in kinds if k != "sfx")  # only bubbles / free text may share a row
    sfx = n - pairable
    # tallest region any slot must hold: a paired 2-line bubble at the smallest font, or an SFX
    needed = 105 if sfx else 91
    rows_cap = (height - 2 * _MARGIN) // (needed + _GAP)
    rows_min = sfx + -(-pairable // 2)  # sfx rows are lone; the rest pair up
    rows_max = min(n, rows_cap)
    if rows_min > rows_max:
        raise ValueError(
            f"cannot place {n} regions on a {width}x{height} page: at least {rows_min} rows are"
            f" needed but at most {rows_max} fit (each region needs >= {needed} px of height)"
        )
    rows = int(rng.integers(rows_min, rows_max + 1))

    slots: list[_Slot] = []
    slot_h = (height - 2 * _MARGIN) / rows
    usable = width - 2 * _MARGIN
    i = 0
    for r in range(rows):
        y0 = _MARGIN + r * slot_h + _INSET
        y1 = _MARGIN + (r + 1) * slot_h - _INSET
        if i >= pairable:  # SFX regions sit in the bottom rows, one per row
            slots.append(_Slot(_MARGIN, width - _MARGIN, y0, y1))
            i += 1
            continue
        remaining_regions, remaining_rows = pairable - i, rows - sfx - r
        if remaining_regions == remaining_rows:
            take = 1
        elif remaining_regions == 2 * remaining_rows:
            take = 2
        else:
            take = 2 if rng.random() < 0.5 else 1
        if take == 2:
            col = (usable - _GAP) // 2
            slots.append(_Slot(_MARGIN, _MARGIN + col, y0, y1))
            slots.append(_Slot(width - _MARGIN - col, width - _MARGIN, y0, y1))
            i += 2
        else:
            slots.append(_Slot(_MARGIN, width - _MARGIN, y0, y1))
            i += 1
    return slots


@lru_cache(maxsize=64)
def _load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path(name)), size)


def _block_size(
    lines: tuple[str, ...], inks: tuple[tuple[int, int, int, int], ...], size: int
) -> tuple[int, int]:
    """Ink width and height of a centred line block (anchors share x, spaced 1.25 x size)."""
    offs = [(i - (len(lines) - 1) / 2) * 1.25 * size for i in range(len(lines))]
    x0 = min(ink[0] for ink in inks)
    x1 = max(ink[2] for ink in inks)
    y0 = min(off + ink[1] for off, ink in zip(offs, inks, strict=True))
    y1 = max(off + ink[3] for off, ink in zip(offs, inks, strict=True))
    return x1 - x0, int(np.ceil(y1 - y0))


def _ellipse_need(
    lines: tuple[str, ...],
    inks: tuple[tuple[int, int, int, int], ...],
    size: int,
    block_h: float,
    margin: int,
) -> int:
    """Smallest ellipse width whose interior keeps every line's ink `margin` px away."""
    need = 0
    for i, (_line, ink) in enumerate(zip(lines, inks, strict=True)):
        off = (i - (len(lines) - 1) / 2) * 1.25 * size
        for d in (off + ink[1], off + ink[3]):  # top and bottom ink rows of this line
            ratio = min(0.99, 2.0 * abs(d) / block_h)
            span = max(0.05, np.sqrt(1.0 - ratio * ratio))
            need = max(need, int(np.ceil((ink[2] - ink[0] + 2 * margin) / span)))
    return need


def _fit_bubble(
    rng: np.random.Generator,
    text: str,
    font_name: str,
    shape: Shape,
    col_w: int,
    avail_h: float,
) -> tuple[int, tuple[str, ...], int, int, Shape] | None:
    """Pick font size, line breaks and a bubble size whose interior holds the text with margin.

    Returns (font size, lines, bubble width, bubble height, shape), or None when nothing fits.
    """
    factor, pad_base = (0.72, 1.0) if shape == "ellipse" else (0.85, 1.2)
    pad_scale = rng.uniform(1.0, 1.3)
    for size in range(int(rng.integers(_MIN_FONT, _MAX_FONT + 1)), _MIN_FONT - 1, -1):
        font = _load_font(font_name, size)
        width = max(140, int(rng.uniform(0.6, 1.0) * col_w))
        for _ in range(24):
            lines = _break_lines(text, font, factor * width - 2)
            inks = tuple(ink for line in lines if (ink := _ink_box(line, font, 0)) is not None)
            if len(inks) != len(lines):
                return None
            block_w, block_h = _block_size(lines, inks, size)
            pad = max(16.0, min(size * pad_base * pad_scale, (avail_h - block_h) / 2 - 2))
            height = block_h + 2 * pad
            if height > avail_h:
                break  # too tall even here: only a smaller font can help
            need = (
                _ellipse_need(lines, inks, size, height, _BUBBLE_PAD)
                if shape == "ellipse"
                else block_w + 2 * _BUBBLE_PAD
            )
            if width >= need:
                return size, lines, width, int(np.ceil(height)), shape
            wider = min(col_w, max(need, width + 20))
            if wider <= width:
                break  # cannot widen any further inside this column
            width = wider
    return None


def _fit_free(
    rng: np.random.Generator, text: str, font_name: str, col_w: int, avail_h: float, stroke: int
) -> tuple[int, tuple[str, ...], int, int]:
    """Pick a font size and line breaks for text drawn straight on the background."""
    low = 90 if stroke > 3 else _MIN_FONT  # SFX starts at 90 px
    for size in range(int(rng.integers(low, 4 * low // 3 + 1)), low - 1, -1):
        font = _load_font(font_name, size)
        limit = 0.9 * col_w if stroke <= 3 else col_w
        lines = _break_lines(text, font, limit - 6)
        inks = tuple(ink for line in lines if (ink := _ink_box(line, font, stroke)) is not None)
        if len(inks) != len(lines):
            continue
        block_w, block_h = _block_size(lines, inks, size)
        if block_w <= col_w and block_h <= avail_h:
            return size, lines, block_w, block_h
    raise ValueError(f"text {text!r} does not fit its {col_w}x{avail_h:.0f} px slot even at font size {low}")


def _plans(
    rng: np.random.Generator, font_name: str, kinds: list[RegionKind], slots: list[_Slot]
) -> list[_Plan]:
    """Decide style, line breaks, size and centre for every region (no ink drawn yet)."""
    texts: list[str] = []
    pairable = sum(1 for k in kinds if k != "sfx")
    picks = rng.choice(len(KOREAN_LINES), size=pairable, replace=pairable > len(KOREAN_LINES))
    texts.extend(KOREAN_LINES[int(i)] for i in picks)
    texts.extend(SFX_LINES[int(j)] for j in rng.integers(0, len(SFX_LINES), len(kinds) - pairable))

    fits: list[tuple[int, tuple[str, ...], int, int, Shape | None]] = []
    for slot, kind, text in zip(slots, kinds, texts, strict=True):
        col_w, avail_h = slot.x1 - slot.x0, slot.y1 - slot.y0
        if kind == "sfx":
            fits.append((*_fit_free(rng, text, font_name, col_w, avail_h, stroke=5), None))
            continue
        if kind == "free_text":
            fits.append((*_fit_free(rng, text, font_name, col_w, avail_h, stroke=3), None))
            continue
        shape: Shape = "ellipse" if rng.random() < 0.5 else "rounded"
        fit = _fit_bubble(rng, text, font_name, shape, col_w, avail_h)
        if fit is None and shape == "ellipse":  # an ellipse is geometrically impossible here
            fit = _fit_bubble(rng, text, font_name, "rounded", col_w, avail_h)
        if fit is None:
            raise ValueError(
                f"bubble text {text!r} does not fit its {col_w}x{avail_h:.0f} px slot"
                f" (font sizes {_MIN_FONT}..{_MAX_FONT})"
            )
        fits.append(fit)

    plans: list[_Plan] = []
    row_start = 0
    while row_start < len(slots):  # group slots into rows (same y range) for shared jitter bounds
        row_end = row_start + 1
        while row_end < len(slots) and (slots[row_end].y0, slots[row_end].y1) == (
            slots[row_start].y0,
            slots[row_start].y1,
        ):
            row_end += 1
        row = range(row_start, row_end)
        min_h = min(fits[i][3] for i in row)
        j_v = min(
            15.0,
            0.2 * min_h,  # keeps same-row boxes vertically overlapping by >= 60%
            min((slots[i].y1 - slots[i].y0 - fits[i][3]) / 2 - 1 for i in row),
        )
        for i in row:
            slot, fit, kind, text = slots[i], fits[i], kinds[i], texts[i]
            size, lines, width, height, fit_shape = fit
            j_h = min(40.0, max(0.0, (slot.x1 - slot.x0 - width) / 2 - 1))
            cx = slot.cx + float(rng.uniform(-j_h, j_h))
            cy = slot.cy + float(rng.uniform(-j_v, j_v))
            bubble: tuple[int, int, Shape, bool, int] | None = None
            if kind == "bubble_text":
                assert fit_shape is not None
                radius = min(28, max(10, int(0.18 * min(width, height))))
                bubble = (width, height, fit_shape, bool(rng.random() < 0.2), radius)
            plans.append(_Plan(kind, text, slot, size, lines, (cx, cy), bubble))
        row_start = row_end
    return plans


def _bubble_box(plan: _Plan, width: int, height: int) -> tuple[int, int]:
    """Top-left corner of a bubble of (width, height) around the plan centre, clamped into its slot."""
    slot = plan.slot
    x0 = max(slot.x0, min(round(plan.centre[0] - width / 2), slot.x1 - width))
    y0 = max(
        int(np.ceil(slot.y0)),
        min(round(plan.centre[1] - height / 2), int(np.floor(slot.y1)) - height),
    )
    return x0, y0


def _draw_bubble(page: Image.Image, plan: _Plan) -> None:
    """Draw the bubble shape for `plan` onto the page."""
    assert plan.bubble is not None
    width, height, shape, dark, radius = plan.bubble
    x0, y0 = _bubble_box(plan, width, height)
    fill: RGB = (24, 24, 32) if dark else (255, 255, 255)
    outline: RGB = (255, 255, 255) if dark else (0, 0, 0)
    draw = ImageDraw.Draw(page)
    if shape == "ellipse":
        draw.ellipse([x0, y0, x0 + width, y0 + height], fill=fill, outline=outline, width=3)
    else:
        draw.rounded_rectangle(
            [x0, y0, x0 + width, y0 + height], radius=radius, fill=fill, outline=outline, width=3
        )


def _union(lines: tuple[LineTruth, ...]) -> BBox:
    """Smallest box containing every line box."""
    return BBox(
        x0=min(line.bbox.x0 for line in lines),
        y0=min(line.bbox.y0 for line in lines),
        x1=max(line.bbox.x1 for line in lines),
        y1=max(line.bbox.y1 for line in lines),
    )


def make_korean_page(
    seed: int,
    *,
    width: int = 800,
    height: int = 1400,
    n_bubbles: int = 4,
    n_free: int = 1,
    n_sfx: int = 0,
    font: str = "NanumGothic-Regular.ttf",
    background: Literal["gradient", "noise", "flat"] = "gradient",
) -> KoreanPage:
    """Generate one deterministic synthetic Korean webtoon page with exact ground truth."""
    rng = np.random.Generator(np.random.PCG64(seed))
    wanted: tuple[tuple[RegionKind, int], ...] = (
        ("bubble_text", n_bubbles),
        ("free_text", n_free),
        ("sfx", n_sfx),
    )
    kinds: list[RegionKind] = [kind for kind, count in wanted for _ in range(count)]
    page = _background(rng, width, height, background)
    if not kinds:
        return KoreanPage(page, page.copy(), np.zeros((height, width), dtype=bool), ())

    slots = _slots(rng, width, height, kinds)
    plans = _plans(rng, font, kinds, slots)

    for plan in plans:  # bubbles first, so `clean` (the ideal inpainting result) can be taken now
        if plan.bubble is not None:
            _draw_bubble(page, plan)
    clean = page.copy()

    work = page.convert("RGBA")
    regions: list[RegionTruth] = []
    for plan in plans:
        bubble_w = bubble_h = radius = 0
        shape: Shape = "rounded"
        dark = False
        if plan.bubble is not None:
            bubble_w, bubble_h, shape, dark, radius = plan.bubble
        if plan.kind == "bubble_text":
            text_color: RGB = (255, 255, 255) if dark else (0, 0, 0)
            fill: RGB = (24, 24, 32) if dark else (255, 255, 255)
            truths = _draw_block(
                work, plan.lines, _load_font(font, plan.font_size), plan.centre, text_color, 0, (0, 0, 0)
            )
        else:
            text_color = _saturated(rng) if plan.kind == "sfx" else (255, 255, 255)
            truths = _draw_block(
                work,
                plan.lines,
                _load_font(font, plan.font_size),
                plan.centre,
                text_color,
                5 if plan.kind == "sfx" else 3,
                (255, 255, 255) if plan.kind == "sfx" else (0, 0, 0),
            )
            arr = np.asarray(clean, dtype=np.float64)
            box = _union(truths)
            mean = arr[box.y0 : box.y1, box.x0 : box.x1].reshape(-1, 3).mean(axis=0)
            fill = (round(float(mean[0])), round(float(mean[1])), round(float(mean[2])))
        bubble_bbox = None
        polygon = None
        if plan.bubble is not None:
            x0, y0 = _bubble_box(plan, bubble_w, bubble_h)
            bubble_bbox = BBox(x0=x0, y0=y0, x1=x0 + bubble_w, y1=y0 + bubble_h)
            polygon = (
                _ellipse_polygon(bubble_bbox) if shape == "ellipse" else _rounded_polygon(bubble_bbox, radius)
            )
        regions.append(
            RegionTruth(
                kind=plan.kind,
                lines=truths,
                bbox=_union(truths),
                bubble_bbox=bubble_bbox,
                bubble_polygon=polygon,
                fill=fill,
                text_color=text_color,
            )
        )

    image = work.convert("RGB")
    mask = np.any(np.asarray(image) != np.asarray(clean), axis=-1)
    return KoreanPage(image, clean, mask, tuple(regions))


def to_regions_artifact(page: KoreanPage) -> RegionsArtifact:
    """Turn a page's ground truth into a `RegionsArtifact` exactly as OCR would produce it."""
    regions = [
        Region(
            id=f"r{i + 1:04d}",
            slice_index=0,
            kind=truth.kind,
            bbox=truth.bbox,
            bubble_bbox=truth.bubble_bbox,
            polygon=list(truth.bubble_polygon) if truth.bubble_polygon is not None else None,
            reading_order=i,
            lang="ko",
            orientation="h",
            lines=[OcrLine(bbox=line.bbox, text=line.text, score=1.0, engine=_TRUTH) for line in truth.lines],
            text=truth.text,
            confidence=1.0,
            text_color=truth.text_color,
        )
        for i, truth in enumerate(page.regions)
    ]
    return RegionsArtifact(regions=regions)

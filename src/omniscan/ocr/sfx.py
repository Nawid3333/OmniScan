"""Sound effects after OCR: which free-text regions are onomatopoeia, and what the original looks like.

The comic text detector's classes never produce a sound effect on real webtoons (M10: zero sfx
regions on two real Solo Leveling chapters) — effects come back as ordinary free text and were
translated and lettered like dialogue. After OCR a free-text region becomes kind "sfx" when its text
reads as a sound effect (`is_sfx`):

- its letters (punctuation, spaces and long-vowel marks ignored, stretched letters collapsed) are at
  most `max_chars` long, and
- they are made of lexicon words (config/sfx_text.toml; 쾅쾅쾅 and 두근두근 are repeats of 쾅 and 두근)
  lettered at least `lexicon_size_ratio` x as large as the chapter's dialogue, or
- with no lexicon match, the text is very short (`size_max_chars`) and lettered at least `size_ratio`
  x as large as the dialogue — big standalone lettering in the art is an effect.

`measure_lettering_style` then reads the look of every effect and every free-text caption off the
strip — fill and outline colour (a white caption with a black outline is not "black text"), and for
effects the baseline angle and stroke weight — which the typesetter uses to letter the English to match.
"""

from __future__ import annotations

import math
import statistics
import tomllib
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 — torch's standard alias

from omniscan.core.config import DEFAULT_TOML, USER_TOML, SfxConfig
from omniscan.core.schemas import RGB, Region
from omniscan.gpu.morph import dilate, erode
from omniscan.inpaint.glyph_mask import fill_holes, find_glyphs, half_stroke, letters_only, outline_width

_IGNORED = frozenset("ー〜～ッっ")  # long-vowel and glottal-stop marks: stretching, not new sounds
_MIN_REFERENCE_REGIONS = 3  # fewer dialogue regions than this: no size reference
_STYLE_PAD_PX = 24  # crop margin around a region when measuring its lettering (room to see the background)
_MIN_THINNING = 1.04  # a tilt must make the lettering this much thinner across than level lettering
_EDGE_ANGLE = 5.0  # a best tilt this close to the searched range's edge is vertical lettering
_MIN_RUN = 1.6  # the lettering must run this much longer along its baseline than across it
_MAX_SAMPLES = 20_000  # ink pixels sampled for tilt and height
_MIN_ANGLE = 8.0  # smaller tilts are the letters' own shapes (or invisible): lettered level
_MAX_ANGLE = 45.0  # steeper tilts are vertical lettering (stacked by the typesetter), not a tilt
_FILL_CONTRAST = 60.0  # an enclosed area this far (max channel) from the background is a fill colour
_MIN_OUTLINE_PX = 2  # thinner rings around the letters are anti-aliasing, not an outline


def default_sfx_text_paths() -> list[Path]:
    """Shipped repo file, then the per-user one (both contribute)."""
    return [DEFAULT_TOML.parent / "sfx_text.toml", USER_TOML.parent / "sfx_text.toml"]


def load_sfx_lexicon(paths: Sequence[Path]) -> dict[str, frozenset[str]]:
    """Per-language sets of normalised sound-effect words from every existing file in `paths`."""
    words: dict[str, set[str]] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{path}: invalid TOML: {exc}") from exc
        table = data.get("sfx_text", {})
        if not isinstance(table, dict):
            raise ValueError(f"{path}: expected an [sfx_text] table")
        for lang, entries in table.items():
            if not isinstance(entries, list) or not all(isinstance(e, str) and e.strip() for e in entries):
                raise ValueError(f"{path}: sfx_text.{lang} must be a list of non-empty strings")
            words.setdefault(lang, set()).update(n for e in entries if (n := normalise(e)))
    return {lang: frozenset(entries) for lang, entries in words.items()}


def normalise(text: str) -> str:
    """The letters of `text` only (no punctuation, spaces or long-vowel marks), runs of one letter
    collapsed to a single letter: "쿠구구구궁!!" -> "쿠구궁", "ドドドド" -> "ド"."""
    letters = [c for c in text if unicodedata.category(c).startswith("L") and c not in _IGNORED]
    out: list[str] = []
    for c in letters:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def letter_count(text: str) -> int:
    """How many letters `text` has (punctuation, spaces and long-vowel marks ignored)."""
    return sum(1 for c in text if unicodedata.category(c).startswith("L") and c not in _IGNORED)


def made_of_words(core: str, words: frozenset[str]) -> bool:
    """Whether the normalised `core` splits into a sequence of lexicon `words` (repeats allowed)."""
    if not core or not words:
        return False
    longest = max(len(word) for word in words)
    reachable = [True] + [False] * len(core)
    for end in range(1, len(core) + 1):
        for start in range(max(0, end - longest), end):
            if reachable[start] and core[start:end] in words:
                reachable[end] = True
                break
    return reachable[-1]


def glyph_size(region: Region) -> float | None:
    """Typical letter size of a region: the side of its box's area shared out per letter."""
    count = letter_count(region.text)
    if count == 0:
        return None
    return math.sqrt(region.bbox.width * region.bbox.height / count)


def dialogue_glyph_size(regions: Sequence[Region]) -> float | None:
    """Median letter size of the chapter's bubble text; None with too few bubbles to tell."""
    sizes = [s for r in regions if r.kind == "bubble_text" and (s := glyph_size(r)) is not None]
    return statistics.median(sizes) if len(sizes) >= _MIN_REFERENCE_REGIONS else None


def is_sfx(region: Region, words: frozenset[str], reference: float | None, cfg: SfxConfig) -> bool:
    """Whether a free-text region reads as a sound effect (see the module docstring)."""
    if region.kind != "free_text" or any(c.isdigit() or (c.isascii() and c.isalpha()) for c in region.text):
        return False
    count = letter_count(region.text)
    if count == 0 or count > cfg.max_chars:
        return False
    size = glyph_size(region)
    ratio = size / reference if size is not None and reference else None
    if made_of_words(normalise(region.text), words):
        return ratio is None or ratio >= cfg.lexicon_size_ratio
    return ratio is not None and count <= cfg.size_max_chars and ratio >= cfg.size_ratio


def reclassify_sfx_regions(
    regions: Sequence[Region], lexicon: Mapping[str, frozenset[str]], cfg: SfxConfig
) -> list[Region]:
    """Free-text regions that read as sound effects become kind "sfx"; every other region is unchanged
    (the same object)."""
    reference = dialogue_glyph_size(regions)
    return [
        region.model_copy(update={"kind": "sfx"})
        if is_sfx(region, lexicon.get(region.lang, frozenset()), reference, cfg)
        else region
        for region in regions
    ]


def _median_rgb(crop: torch.Tensor, mask: torch.Tensor) -> RGB:
    """Median colour of the masked pixels of a uint8 [3, h, w] crop."""
    values = crop[:, mask].float().median(dim=1).values.round().tolist()
    return (int(values[0]), int(values[1]), int(values[2]))


def _far(rgb: RGB, other: RGB, limit: float) -> bool:
    return max(abs(a - b) for a, b in zip(rgb, other, strict=True)) > limit


def _extents(points: torch.Tensor, degrees: torch.Tensor, *, along: bool = False) -> torch.Tensor:
    """Robust span (2nd-98th percentile) of (y, x) `points` across — or with `along`, along — a baseline
    tilted by each of `degrees` counter-clockwise."""
    theta = torch.deg2rad(degrees)[:, None]
    if along:
        coords = points[None, :, 1] * torch.cos(theta) - points[None, :, 0] * torch.sin(theta)
    else:
        coords = points[None, :, 0] * torch.cos(theta) + points[None, :, 1] * torch.sin(theta)
    spans = torch.quantile(coords, torch.tensor([0.02, 0.98], device=points.device), dim=1)
    return spans[1] - spans[0] + 1.0


def _sample(ink: torch.Tensor) -> torch.Tensor:
    """Float (y, x) positions of the ink pixels, strided down to at most `_MAX_SAMPLES`."""
    points = torch.nonzero(ink).float()
    if points.shape[0] > _MAX_SAMPLES:
        points = points[:: math.ceil(points.shape[0] / _MAX_SAMPLES)]
    return points


def ink_angle(ink: torch.Tensor) -> float:
    """Baseline tilt of the lettering in degrees counter-clockwise: the tilt at which a run of letters is
    thinnest across. 0.0 when the tilt is below `_MIN_ANGLE`, at the edge of the searched range (a
    vertical column), makes the lettering thinner than level by less than `_MIN_THINNING`, or when the
    lettering is not at least `_MIN_RUN` x longer along that baseline than across it (a lone syllable
    has no baseline to measure)."""
    points = _sample(ink)
    if points.shape[0] < 3:
        return 0.0
    degrees = torch.arange(-_MAX_ANGLE, _MAX_ANGLE + 1.0, 1.0, device=points.device)
    extents = _extents(points, degrees)
    best = int(torch.argmin(extents))
    angle = float(degrees[best])
    thinnest = float(extents[best])
    level = float(extents[len(degrees) // 2])
    run = float(_extents(points, degrees[best : best + 1], along=True)[0])
    if not _MIN_ANGLE <= abs(angle) <= _MAX_ANGLE - _EDGE_ANGLE:
        return 0.0
    if level < _MIN_THINNING * thinnest or run < _MIN_RUN * thinnest:
        return 0.0
    return angle


def thinnest_tilt(ink: torch.Tensor) -> float:
    """The baseline tilt (degrees counter-clockwise, within the searched range) at which the ink is
    thinnest across — its tilt even when too short to report as a tilt; 0.0 without ink."""
    points = _sample(ink)
    if points.shape[0] < 3:
        return 0.0
    degrees = torch.arange(-_MAX_ANGLE, _MAX_ANGLE + 1.0, 1.0, device=points.device)
    return float(degrees[int(torch.argmin(_extents(points, degrees)))])


def levelled(mask: torch.Tensor, degrees: float) -> torch.Tensor:
    """`mask` rotated clockwise by `degrees` (undoing a counter-clockwise tilt) on a square canvas large
    enough to hold it, so strokes are measured upright (a square kernel erodes diagonal strokes faster)."""
    if abs(degrees) < 1.0:
        return mask
    h, w = mask.shape
    side = math.ceil(math.hypot(h, w)) + 2
    top, left = (side - h) // 2, (side - w) // 2
    canvas = F.pad(mask.float()[None, None], (left, side - w - left, top, side - h - top))
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    theta = torch.tensor([[[c, s, 0.0], [-s, c, 0.0]]], dtype=canvas.dtype, device=mask.device)
    grid = F.affine_grid(theta, list(canvas.shape), align_corners=False)
    return F.grid_sample(canvas, grid, mode="bilinear", align_corners=False)[0, 0] > 0.5


def letter_height(ink: torch.Tensor, lines: int) -> float:
    """Size of one letter: the lettering's thinnest extent across any baseline tilt in the searched range
    (a tilted lone syllable is no taller than a level one) per line, or its level width when that is
    smaller (a vertical column of letters)."""
    points = _sample(ink)
    if points.shape[0] < 3:
        return 0.0
    degrees = torch.arange(-_MAX_ANGLE, _MAX_ANGLE + 1.0, 1.0, device=points.device)
    across = float(_extents(points, degrees).min()) / max(1, lines)
    level = torch.zeros(1, device=points.device)
    return min(across, float(_extents(points, level, along=True)[0]))


def measure_lettering_style(strip: torch.Tensor, region: Region) -> Region:
    """`region` with its lettering's `text_color` and `stroke_color` measured on the uint8 [3, H, W]
    strip (an outline told apart from the fill), plus `angle` and `weight` for a sound effect;
    unchanged when the lettering cannot be separated from the art.

    Art lines as dark as the letters join the ink cluster; strokes much thinner than the lettering's
    own are opened away before the shape is measured, so they neither tilt nor lighten the effect.
    """
    height, width = int(strip.shape[1]), int(strip.shape[2])
    x0, y0 = max(0, region.bbox.x0 - _STYLE_PAD_PX), max(0, region.bbox.y0 - _STYLE_PAD_PX)
    x1, y1 = min(width, region.bbox.x1 + _STYLE_PAD_PX), min(height, region.bbox.y1 + _STYLE_PAD_PX)
    crop = strip[:, y0:y1, x0:x1]
    boxes = [(ln.bbox.x0 - x0, ln.bbox.y0 - y0, ln.bbox.x1 - x0, ln.bbox.y1 - y0) for ln in region.lines]
    if not boxes:
        boxes = [(region.bbox.x0 - x0, region.bbox.y0 - y0, region.bbox.x1 - x0, region.bbox.y1 - y0)]
    glyphs = find_glyphs(crop, boxes, grow=2.0, max_grow_px=16)
    if glyphs is None:
        return region
    ink, half = letters_only(crop, glyphs.ink)  # half: about a quarter of the letters' stroke width
    body = fill_holes(ink)
    reach = max(2, 2 * half)  # about half the stroke width
    rim = outline_width(crop, body, reach + 8)
    # the background is sampled beyond any outline: a thick white rim is not the page behind the letters
    far = dilate(body, rim + 2 * reach + 3) & ~dilate(body, rim + 2 * reach)
    background = _median_rgb(crop, far) if bool(far.any()) else None
    outline = None
    if rim >= _MIN_OUTLINE_PX and background is not None:
        colour = _median_rgb(crop, dilate(body, rim) & ~body)
        outline = colour if _far(colour, background, _FILL_CONTRAST / 2) else None
    enclosed = body & ~ink
    if background is not None and int(enclosed.sum()) > 0.2 * int(ink.sum()):
        inner = _median_rgb(crop, enclosed)
        # the ink is an outline around a coloured fill — unless the "fill" is the letters' own outline
        # showing inside their counters (an outlined "ㅇ" is lined with the outline's colour)
        if _far(inner, background, _FILL_CONTRAST) and (
            outline is None or _far(inner, outline, _FILL_CONTRAST)
        ):
            # the strokes are the enclosed pixels of the fill's colour (not the counters inside them)
            near = (crop.float() - torch.tensor(inner, device=crop.device)[:, None, None]).abs().amax(dim=0)
            strokes = enclosed & (near <= _FILL_CONTRAST)
            return _styled(
                region, inner, _median_rgb(crop, ink), strokes if bool(strokes.any()) else enclosed
            )
    core = erode(ink, 1)
    return _styled(region, _median_rgb(crop, core if bool(core.any()) else ink), outline, ink)


def _styled(region: Region, fill: RGB, outline: RGB | None, letters: torch.Tensor) -> Region:
    """`region` with the measured colours and, for a sound effect, the tilt of its `letters` (the fill
    strokes, without any outline) and their stroke weight: stroke width ~ 4 x half-stroke over the
    letter height, both measured on the letters rotated level."""
    if region.kind != "sfx":
        return region.model_copy(update={"text_color": fill, "stroke_color": outline})
    angle = ink_angle(letters)
    upright = levelled(letters, thinnest_tilt(letters))
    half = half_stroke(upright)
    tall = letter_height(upright, max(1, region.text.count("\n") + 1))
    return region.model_copy(
        update={
            "text_color": fill,
            "stroke_color": outline,
            "angle": angle,
            "weight": round(4 * half / tall, 3) if tall > 0 else None,
        }
    )

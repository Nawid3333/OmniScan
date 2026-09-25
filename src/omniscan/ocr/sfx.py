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

from omniscan.core.config import DEFAULT_TOML, USER_TOML, SfxConfig
from omniscan.core.schemas import RGB, Region
from omniscan.gpu.morph import dilate, erode
from omniscan.inpaint.glyph_mask import find_glyphs, outline_width, stroke_width

_IGNORED = frozenset("ー〜～ッっ")  # long-vowel and glottal-stop marks: stretching, not new sounds
_MIN_REFERENCE_REGIONS = 3  # fewer dialogue regions than this: no size reference
_STYLE_PAD_PX = 6  # crop padding around a region when measuring its lettering
_MIN_ANISOTROPY = 3.0  # the ink must be this much longer than tall before its tilt counts
_MIN_ANGLE = 4.0  # smaller tilts are the letters' own shapes: lettered level
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


def ink_angle(ink: torch.Tensor) -> float:
    """Baseline tilt of the ink in degrees counter-clockwise (principal axis of its pixel positions);
    0.0 when the ink is not clearly elongated, tilts less than `_MIN_ANGLE` or runs steeper than
    `_MAX_ANGLE` (vertical lettering)."""
    points = torch.nonzero(ink).float()  # (y, x) rows
    if points.shape[0] < 3:
        return 0.0
    centred = points - points.mean(dim=0)
    cov = centred.T @ centred / (points.shape[0] - 1)
    var_y, var_x, cov_xy = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
    spread = math.sqrt(((var_x - var_y) / 2) ** 2 + cov_xy**2)
    major, minor = (var_x + var_y) / 2 + spread, (var_x + var_y) / 2 - spread
    if minor <= 0 or major / minor < _MIN_ANISOTROPY**2:
        return 0.0
    angle = -math.degrees(0.5 * math.atan2(2 * cov_xy, var_x - var_y))  # y points down on the page
    return round(angle, 1) if _MIN_ANGLE <= abs(angle) <= _MAX_ANGLE else 0.0


def letter_height(ink: torch.Tensor, lines: int, angle: float) -> float:
    """Height of one line of the ink, measured across its baseline (tilted by `angle` degrees)."""
    points = torch.nonzero(ink).float()  # (y, x) rows
    if points.shape[0] == 0:
        return 0.0
    theta = math.radians(angle)
    across = points[:, 0] * math.cos(theta) + points[:, 1] * math.sin(theta)  # along the baseline's normal
    return (float(across.max() - across.min()) + 1.0) / max(1, lines)


def measure_lettering_style(strip: torch.Tensor, region: Region) -> Region:
    """`region` with its lettering's `text_color` and `stroke_color` measured on the uint8 [3, H, W]
    strip (an outline told apart from the fill), plus `angle` and `weight` for a sound effect;
    unchanged when the lettering cannot be separated from the art."""
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
    ink, body = glyphs.ink, glyphs.body
    reach = max(2, math.ceil(glyphs.stroke_px))
    far = dilate(body, 3 * reach + 3) & ~dilate(body, 2 * reach + 2)
    background = _median_rgb(crop, far) if bool(far.any()) else None
    enclosed = body & ~ink
    letters = ink
    if background is not None and int(enclosed.sum()) > 0.2 * int(ink.sum()):
        inner = _median_rgb(crop, enclosed)
        if _far(inner, background, _FILL_CONTRAST):  # the ink is an outline around a coloured fill
            fill, outline, letters = inner, _median_rgb(crop, ink), body
            return _styled(region, fill, outline, letters)
    core = erode(ink, 1)
    fill = _median_rgb(crop, core if bool(core.any()) else ink)
    rim = outline_width(crop, body, reach + 8)
    outline = None
    if rim >= _MIN_OUTLINE_PX and background is not None:
        ring = dilate(body, rim) & ~body
        colour = _median_rgb(crop, ring)
        outline = colour if _far(colour, background, _FILL_CONTRAST / 2) else None
        letters = dilate(body, rim) if outline is not None else ink
    return _styled(region, fill, outline, letters)


def _styled(region: Region, fill: RGB, outline: RGB | None, letters: torch.Tensor) -> Region:
    """`region` with the measured colours and, for a sound effect, the tilt and stroke weight of its
    `letters` mask."""
    if region.kind != "sfx":
        return region.model_copy(update={"text_color": fill, "stroke_color": outline})
    angle = ink_angle(letters)
    tall = letter_height(letters, max(1, region.text.count("\n") + 1), angle)
    return region.model_copy(
        update={
            "text_color": fill,
            "stroke_color": outline,
            "angle": angle,
            "weight": round(stroke_width(letters) / tall, 3) if tall > 0 else None,
        }
    )

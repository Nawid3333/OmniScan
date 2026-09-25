"""Whole-page lettering sweep: sound effects (and stamped watermarks) the comic detector missed.

The comic detector finds speech and captions but few of the effects drawn into the art (M10). With
`sfx.sweep` the OCR stage also runs CRAFT (ocr/craft.py) over every active tile of the strip:

1. word boxes mostly inside a detected region (or its balloon) belong to it and are dropped;
2. the rest are candidates — one word each (CRAFT's own links join the letters of a word; words side
   by side are usually different effects), except syllables of one size stacked in a column, which
   are one effect — whose letters are at least `sfx.sweep_min_px` big;
3. each candidate is read twice by the stage's own OCR engine: its crop, and its letters alone (black
   on white, without the art, outline or colour around them), both turned level when the lettering is
   tilted and, for a column, with the syllables laid out as a row (line recognisers read left to right);
4. a confident reading that matches a watermark pattern becomes a watermark (erased); one that is a
   known sound effect within a few misread strokes (`closest_sfx`, one letter edit per two syllables)
   becomes kind "sfx", spelled as the lexicon spells it when that spelling is the only close one (the
   raw reading kept in `ocr_alt`); the closest of the two readings wins. Anything else stays
   untouched: an unknown word may be art.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F  # noqa: N812 — torch's standard alias

from omniscan.core.schemas import BBox, Lang, OcrLine, Region, RegionKind, Slice
from omniscan.detect.postprocess import Box, ioa
from omniscan.detect.tiles import Tile
from omniscan.inpaint.glyph_mask import find_glyphs, letters_only
from omniscan.ocr.crop_readers import TextReader
from omniscan.ocr.lines import LineBox, merge_lines
from omniscan.ocr.sfx import closest_sfx, letter_count, normalise, thinnest_tilt
from omniscan.ocr.watermark_text import matches_watermark_text

_INSIDE_IOA = 0.5  # a word with this share of its box inside a region's (padded) box belongs to it
_REGION_PAD_PX = 8  # regions' boxes are grown by this before the test above
_NMS_IOU = 0.5  # the same word seen by two overlapping tiles
_OVERLAP = 0.7  # syllables in one column overlap sideways by this share of the narrower one
_GAP = 0.3  # ... and are at most this x their width apart
_SIZE_RATIO = 1.3  # neighbours this much wider or narrower are separate effects
_SYLLABLE = (0.6, 1.6)  # width / height of one stacked syllable
_TALL = 1.6  # a box this many times taller than wide is a column of letters
_CROP_PAD = 0.06  # crop margin around a candidate, x its letter size (+ 4 px): art around it misleads
_MIN_TILT = 3.0  # smaller tilts are not undone before reading
_MAX_TILT = 40.0  # a steeper "tilt" is a lone glyph's own shape (the search ends at 45)
_EDITS_PER_LETTER = 0.5  # misread strokes allowed per syllable when matching the lexicon (rounded up)


class WordDetector(Protocol):
    """CRAFT (ocr/craft.py): tiles -> (word box in tile pixels, peak score) each."""

    def detect(self, tiles: Sequence[torch.Tensor]) -> list[list[tuple[Box, float]]]: ...


@dataclass(frozen=True, slots=True)
class Candidate:
    """Lettering the sweep found outside every region: its word boxes and their union (strip pixels)."""

    box: Box
    words: tuple[Box, ...]  # one, or a column's syllables top to bottom
    vertical: bool

    @property
    def letter_px(self) -> float:
        """The size of one letter: the height of a word, the width of a column."""
        return self.box[2] - self.box[0] if self.vertical else self.box[3] - self.box[1]


def find_words(strip: torch.Tensor, tiles: Sequence[Tile], detector: WordDetector) -> list[LineBox]:
    """CRAFT word boxes over `tiles` of the uint8 [3, H, W] strip, in strip pixels, merged across tiles."""
    found: list[LineBox] = []
    for tile, words in zip(
        tiles, detector.detect([strip[:, t.y0 : t.y1, t.x0 : t.x1] for t in tiles]), strict=True
    ):
        found.extend(
            LineBox((b[0] + tile.x0, b[1] + tile.y0, b[2] + tile.x0, b[3] + tile.y0), score)
            for b, score in words
        )
    return merge_lines(found, nms_iou=_NMS_IOU)


def outside_regions(words: Sequence[LineBox], regions: Sequence[Region]) -> list[LineBox]:
    """The words not mostly inside any region's box or balloon (grown by `_REGION_PAD_PX`)."""
    taken: list[Box] = []
    for region in regions:
        for box in (region.bbox, region.bubble_bbox):
            if box is not None:
                taken.append(
                    (
                        box.x0 - _REGION_PAD_PX,
                        box.y0 - _REGION_PAD_PX,
                        box.x1 + _REGION_PAD_PX,
                        box.y1 + _REGION_PAD_PX,
                    )
                )
    return [word for word in words if all(ioa(word.box, box) < _INSIDE_IOA for box in taken)]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def _syllable(box: Box) -> bool:
    return _SYLLABLE[0] <= (box[2] - box[0]) / max(1e-6, box[3] - box[1]) <= _SYLLABLE[1]


def _same_column(a: Box, b: Box) -> bool:
    """Whether two syllable-shaped boxes of one width sit one above the other, close together."""
    wa, wb = a[2] - a[0], b[2] - b[0]
    if not (_syllable(a) and _syllable(b)) or max(wa, wb) > _SIZE_RATIO * min(wa, wb):
        return False
    return _overlap(a[0], a[2], b[0], b[2]) >= _OVERLAP * min(wa, wb) and -_overlap(
        a[1], a[3], b[1], b[3]
    ) <= (_GAP * max(wa, wb))


def group_words(words: Sequence[LineBox], *, min_px: float) -> list[Candidate]:
    """One candidate per word, stacked syllables joined into a column; the ones whose letters are
    smaller than `min_px` dropped; sorted top to bottom."""
    boxes = [word.box for word in words]
    parent = list(range(len(boxes)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    stacked: set[int] = set()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _same_column(boxes[i], boxes[j]):
                parent[root(i)] = root(j)
                stacked.update((i, j))
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(root(i), []).append(i)
    candidates: list[Candidate] = []
    for members in groups.values():
        member_boxes = [boxes[i] for i in members]
        union = (
            min(b[0] for b in member_boxes),
            min(b[1] for b in member_boxes),
            max(b[2] for b in member_boxes),
            max(b[3] for b in member_boxes),
        )
        vertical = any(i in stacked for i in members) or (union[3] - union[1]) >= _TALL * (
            union[2] - union[0]
        )
        candidate = Candidate(
            box=union, words=tuple(sorted(member_boxes, key=lambda b: (b[1], b[0]))), vertical=vertical
        )
        if candidate.letter_px >= min_px:
            candidates.append(candidate)
    return sorted(candidates, key=lambda c: (c.box[1], c.box[0]))


def _crop(strip: torch.Tensor, box: Box, pad: float) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """A view of `box` grown by `pad` (clamped to the strip) and its integer bounds."""
    height, width = int(strip.shape[1]), int(strip.shape[2])
    x0, y0 = max(0, math.floor(box[0] - pad)), max(0, math.floor(box[1] - pad))
    x1, y1 = min(width, math.ceil(box[2] + pad)), min(height, math.ceil(box[3] + pad))
    return strip[:, y0:y1, x0:x1], (x0, y0, x1, y1)


def _rotate(crop: torch.Tensor, degrees: float) -> torch.Tensor:
    """A uint8 [3, h, w] crop turned clockwise by `degrees` (undoing a counter-clockwise tilt) on a
    square canvas that holds all of it, the new corners filled from the crop's edges."""
    h, w = int(crop.shape[1]), int(crop.shape[2])
    side = math.ceil(math.hypot(h, w)) + 2
    top, left = (side - h) // 2, (side - w) // 2
    canvas = F.pad(crop.float()[None], (left, side - w - left, top, side - h - top), mode="replicate")
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    theta = torch.tensor([[[c, s, 0.0], [-s, c, 0.0]]], dtype=canvas.dtype, device=crop.device)
    grid = F.affine_grid(theta, list(canvas.shape), align_corners=False)
    turned = F.grid_sample(canvas, grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
    return turned.round().clamp(0, 255).to(torch.uint8)


def _bounds(mask: torch.Tensor, pad: int) -> tuple[int, int, int, int] | None:
    """The (x0, y0, x1, y1) bounding box of a bool [h, w] mask grown by `pad` (clamped); None when empty."""
    rows, cols = torch.nonzero(mask.any(dim=1)).flatten(), torch.nonzero(mask.any(dim=0)).flatten()
    if rows.numel() == 0:
        return None
    return (
        max(0, int(cols[0]) - pad),
        max(0, int(rows[0]) - pad),
        min(int(mask.shape[1]), int(cols[-1]) + 1 + pad),
        min(int(mask.shape[0]), int(rows[-1]) + 1 + pad),
    )


def _views(crop: torch.Tensor, box: tuple[int, int, int, int], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(the crop, its letters in black on white) turned level when the letters in `box` (crop pixels)
    are tilted and cut to the letters grown by `pad`; the crop twice when no letters are found."""
    glyphs = find_glyphs(crop, [box], grow=2.0, max_grow_px=16)
    if glyphs is None:
        return crop, crop
    letters, _ = letters_only(crop, glyphs.ink)
    tilt = thinnest_tilt(letters)
    if _MIN_TILT <= abs(tilt) <= _MAX_TILT:
        crop = _rotate(crop, tilt)
        letters = _rotate(letters.to(torch.uint8)[None].expand(3, -1, -1) * 255, tilt)[0] > 127
    bounds = _bounds(letters, pad)
    if bounds is None:
        return crop, crop
    x0, y0, x1, y1 = bounds
    black = torch.where(letters[y0:y1, x0:x1], 0, 255).to(torch.uint8)[None].expand(3, -1, -1)
    return crop[:, y0:y1, x0:x1], black.contiguous()


def _as_row(pieces: Sequence[torch.Tensor]) -> torch.Tensor:
    """Crops scaled to one height and laid side by side (a column of syllables as a line)."""
    height = min(int(piece.shape[1]) for piece in pieces)
    scaled = [
        F.interpolate(
            piece.float()[None],
            size=(height, max(1, round(int(piece.shape[2]) * height / int(piece.shape[1])))),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        for piece in pieces
    ]
    return torch.cat(scaled, dim=2).round().clamp(0, 255).to(torch.uint8)


def reading_crops(strip: torch.Tensor, candidate: Candidate) -> tuple[torch.Tensor, torch.Tensor]:
    """What the recogniser reads for `candidate`: (its crop, its letters alone in black on white), turned
    level when the lettering is tilted, a column's syllables laid out left to right."""
    pad = round(_CROP_PAD * candidate.letter_px) + 4
    if candidate.vertical and len(candidate.words) > 1:
        pieces = candidate.words
    elif candidate.vertical:
        x0, y0, x1, y1 = candidate.box
        cells = max(2, round((y1 - y0) / max(1.0, x1 - x0)))
        step = (y1 - y0) / cells
        pieces = tuple((x0, y0 + k * step, x1, y0 + (k + 1) * step) for k in range(cells))
    else:
        pieces = (candidate.box,)
    views: list[tuple[torch.Tensor, torch.Tensor]] = []
    for piece in pieces:
        crop, (cx0, cy0, _, _) = _crop(strip, piece, 2 * pad)
        box = (
            math.floor(piece[0]) - cx0,
            math.floor(piece[1]) - cy0,
            math.ceil(piece[2]) - cx0,
            math.ceil(piece[3]) - cy0,
        )
        views.append(_views(crop, box, pad))
    if len(views) == 1:
        return views[0]
    return _as_row([colour for colour, _ in views]), _as_row([black for _, black in views])


def _respell(raw: str, spelling: str) -> str:
    """`spelling` wrapped in the punctuation around the letters of `raw` ("광!" + 쾅 -> "쾅!")."""
    letters = [i for i, c in enumerate(raw) if normalise(c)]
    if not letters:
        return spelling
    return raw[: letters[0]] + spelling + raw[letters[-1] + 1 :]


def _slice_of(slices: Sequence[Slice], box: Box) -> Slice | None:
    centre = (box[1] + box[3]) / 2
    return next((s for s in slices if s.y0 <= centre < s.y1), None)


@dataclass(frozen=True, slots=True)
class SweepRules:
    """What a candidate's reading must be to become a region, and how the region is labelled."""

    words: frozenset[str]  # the source language's sound-effect lexicon (normalised)
    watermark_patterns: tuple[str, ...]
    min_score: float  # readings below this are not trusted (ocr.drop_conf)
    max_chars: int  # longer readings are never an effect (sfx.max_chars)
    engine: str  # the reader's name, recorded on the region's line
    lang: Lang


@dataclass(frozen=True, slots=True)
class _Reading:
    """What one reading of a candidate makes it: its kind, text (respelled) and the raw reading."""

    kind: RegionKind
    text: str
    raw: str
    score: float
    edits: int  # letter edits from the lexicon's spelling (0 for a watermark)


def classify(raw: str, score: float, rules: SweepRules) -> _Reading | None:
    """A trusted reading as a watermark or a known sound effect (see the module docstring); None when
    it is neither."""
    if matches_watermark_text(raw, rules.watermark_patterns):
        return _Reading("watermark", raw, raw, score, 0)
    if any(c.isdigit() or (c.isascii() and c.isalpha()) for c in raw):
        return None
    core = normalise(raw)
    if not 0 < letter_count(raw) <= rules.max_chars:
        return None
    match = closest_sfx(core, rules.words, math.ceil(_EDITS_PER_LETTER * len(core)))
    if match is None:
        return None
    spelling, edits, unique = match
    return _Reading("sfx", _respell(raw, spelling) if edits and unique else raw, raw, score, edits)


def sweep_regions(
    regions: Sequence[Region],
    candidates: Sequence[Candidate],
    readings: Sequence[Sequence[tuple[str, float]]],
    slices: Sequence[Slice],
    rules: SweepRules,
    strip_size: tuple[int, int],
) -> tuple[list[Region], dict[str, float]]:
    """`regions` plus a region for every candidate one of whose readings is a watermark or a known sound
    effect — the closest such reading, then the surest — ids continuing after the last region's, each
    after its slice's other regions; plus the sweep's counts. `strip_size` is (width, height)."""
    out = list(regions)
    width, height = strip_size
    counts = {"sweep_sfx": 0.0, "sweep_watermarks": 0.0, "sweep_unknown": 0.0}
    for candidate, options in zip(candidates, readings, strict=True):
        owner = _slice_of(slices, candidate.box)
        trusted = [(raw, score) for raw, score in options if score >= rules.min_score and raw.strip()]
        if owner is None or owner.blank or owner.filtered or not trusted:
            continue
        found = [r for raw, score in trusted if (r := classify(raw, score, rules)) is not None]
        if not found:
            counts["sweep_unknown"] += 1
            continue
        best = min(found, key=lambda r: (r.edits, -r.score))
        box = BBox(
            x0=max(0, math.floor(candidate.box[0])),
            y0=max(0, math.floor(candidate.box[1])),
            x1=min(width, math.ceil(candidate.box[2])),
            y1=min(height, math.ceil(candidate.box[3])),
        )
        counts["sweep_sfx" if best.kind == "sfx" else "sweep_watermarks"] += 1
        out.append(
            Region(
                id=f"r{len(out) + 1:04d}",
                slice_index=owner.index,
                kind=best.kind,
                bbox=box,
                reading_order=sum(1 for r in out if r.slice_index == owner.index),
                lang=rules.lang,
                orientation="v" if candidate.vertical else "h",
                lines=[OcrLine(bbox=box, text=best.text, score=round(best.score, 4), engine=rules.engine)],
                text=best.text,
                confidence=round(best.score, 4),
                ocr_alt=best.raw if best.raw != best.text else None,
            )
        )
    return out, counts


def sweep(
    strip: torch.Tensor,
    regions: Sequence[Region],
    slices: Sequence[Slice],
    tiles: Sequence[Tile],
    detector: WordDetector,
    reader: TextReader,
    rules: SweepRules,
    *,
    min_px: float,
) -> tuple[list[Region], dict[str, float]]:
    """Find, read and classify the lettering outside `regions` on `tiles` of the uint8 [3, H, W] strip
    (see the module docstring): the regions with the new ones appended, and the sweep's counts."""
    words = outside_regions(find_words(strip, tiles, detector), regions)
    candidates = group_words(words, min_px=min_px)
    crops = [crop for candidate in candidates for crop in reading_crops(strip, candidate)]
    flat = reader.read(crops) if crops else []
    readings = [flat[2 * i : 2 * i + 2] for i in range(len(candidates))]
    size = (int(strip.shape[2]), int(strip.shape[1]))
    out, counts = sweep_regions(regions, candidates, readings, slices, rules, size)
    return out, {"sweep_words": float(len(words)), "sweep_candidates": float(len(candidates)), **counts}

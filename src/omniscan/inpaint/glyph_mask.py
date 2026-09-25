"""Glyph-precise text masks: remove only the lettering's own ink, its outline and its anti-aliasing.

A line box around lettering is mostly background — the gaps between letters, the space between lines
and, for text drawn on artwork, the art itself. Masking the whole box makes LaMa repaint all of it,
which is where blurred rectangles and smears on text over art come from (and with a crop-reading OCR
engine the "line" is the whole region box). Professional cleaning removes just the lettering; this
module finds it from the pixels, on the crop's device:

1. inside the text boxes, split the pixels into two clusters (Otsu along their principal colour axis,
   which is plain brightness for black-and-white lettering);
2. the ink is the cluster that is rarer on the crop's border than inside the boxes (the border is
   the padding around the text, i.e. background); when the border does not tell, the minority;
3. whatever the ink encloses is part of the lettering too (the fill inside an outline, counters);
4. the lettering is grown over its outline — one-pixel rings are peeled outwards until their colour
   matches the background — plus two pixels of anti-aliasing and JPEG ringing.

An implausible split (too little contrast, almost no ink, mostly ink) returns None and the caller
keeps its rectangle mask — never worse than before.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from omniscan.gpu.morph import dilate, erode, luminance, otsu_threshold
from omniscan.inpaint.flat import line_mask

_MIN_INSIDE_PX = 64  # fewer text-box pixels than this: too small to split reliably
_BORDER_PX = 2  # width of the crop frame sampled as background
_MIN_BORDER_PX = 16  # fewer background samples than this: fall back to the minority rule
_BORDER_MARGIN = 0.05  # the border must differ from the boxes by this share to decide the ink side
_AA_PX = 2  # growth beyond the lettering itself: anti-aliasing and JPEG ringing
_OUTLINE_TOL = (
    16.0  # a ring whose median colour is this close (max channel) to the background ends the lettering
)
_FAR_BAND_PX = 3  # width of the band sampled as background beyond the largest growth
_FLOOD_CHECK_EVERY = 16  # flood-fill steps between convergence checks (each check is a device sync)


@dataclass(frozen=True, slots=True)
class Glyphs:
    """The lettering found in one crop: its ink, its whole body, the removal mask and the stroke width."""

    ink: torch.Tensor  # bool [h, w]: the ink cluster's pixels (the letters' own colour)
    body: torch.Tensor  # bool [h, w]: the ink plus everything it encloses (outlined fills, counters)
    mask: torch.Tensor  # bool [h, w]: the body grown over outline and anti-aliasing (the pixels to remove)
    stroke_px: float  # estimated stroke width of the ink, in pixels
    ink_is_dark: bool  # the ink is darker than the rest of the text boxes


def find_glyphs(
    crop: torch.Tensor,
    boxes: Sequence[tuple[int, int, int, int]],
    *,
    grow: float = 1.0,
    min_grow_px: int = 2,
    max_grow_px: int = 12,
    min_ink: float = 0.01,
    max_ink: float = 0.7,
    min_contrast: float = 48.0,
) -> Glyphs | None:
    """The lettering inside `boxes` (crop-relative) of the uint8 [3, h, w] `crop`, or None.

    The removal mask is the body grown by its outline width plus 2 px, at most
    `ceil(grow * stroke width) + 2` and always within [`min_grow_px`, `max_grow_px`]. None when the
    boxes hold too few pixels, the two colour clusters' means are less than `min_contrast` apart
    (RGB distance along the principal axis), or the ink share of the boxes is outside
    [`min_ink`, `max_ink`].
    """
    height, width = int(crop.shape[1]), int(crop.shape[2])
    inside = line_mask(height, width, boxes, dilate_px=0, device=crop.device)
    n_inside = int(inside.sum())
    if n_inside < _MIN_INSIDE_PX:
        return None
    pixels = crop.float()
    axis = principal_axis(pixels[:, inside])
    if axis is None:
        return None
    projected = torch.einsum("c,chw->hw", axis, pixels)
    lo, hi = projected[inside].aminmax()
    scaled = (projected - lo) * (255.0 / (hi - lo).clamp(min=1e-6))
    low = scaled < otsu_threshold(scaled[inside].clamp(0, 255))
    low_inside = low & inside
    n_low = int(low_inside.sum())
    if n_low == 0 or n_low == n_inside:
        return None
    if float(projected[inside & ~low].mean()) - float(projected[low_inside].mean()) < min_contrast:
        return None
    ink_is_low = _ink_is_low(low, inside, n_low / n_inside)
    ink = inside & (low if ink_is_low else ~low)
    if not min_ink <= int(ink.sum()) / n_inside <= max_ink:
        return None
    luma = luminance(crop)
    ink_is_dark = float(luma[ink].mean()) < float(luma[inside & ~ink].mean())
    body = fill_holes(ink)
    if int((body & inside).sum()) / n_inside > max_ink:
        body = ink  # the ink encloses most of the boxes (a frame, not letters): keep just the ink
    stroke = stroke_width(ink)
    limit = min(max_grow_px, max(min_grow_px, math.ceil(grow * stroke) + _AA_PX))
    grow_px = min(limit, max(min_grow_px, outline_width(crop, body, limit) + _AA_PX))
    return Glyphs(ink=ink, body=body, mask=dilate(body, grow_px), stroke_px=stroke, ink_is_dark=ink_is_dark)


def principal_axis(pixels: torch.Tensor) -> torch.Tensor | None:
    """Unit RGB direction of largest colour variance of float [3, n] `pixels` (for black-and-white
    lettering this is the grey axis; for coloured text on a background of equal brightness it still
    separates the two); None for fewer than two pixels."""
    if pixels.shape[1] < 2:
        return None
    centred = pixels - pixels.mean(dim=1, keepdim=True)
    _, vectors = torch.linalg.eigh(centred @ centred.T / (pixels.shape[1] - 1))
    return vectors[:, -1]


def _ink_is_low(low: torch.Tensor, inside: torch.Tensor, low_share_inside: float) -> bool:
    """Whether the low cluster is the ink: it is when the boxes hold more low pixels than the crop's
    border (the background around the text); the minority cluster when the border cannot tell."""
    frame = torch.ones_like(inside)
    frame[_BORDER_PX:-_BORDER_PX, _BORDER_PX:-_BORDER_PX] = False
    border = frame & ~inside
    n_border = int(border.sum())
    if n_border >= _MIN_BORDER_PX:
        low_share_border = int((low & border).sum()) / n_border
        if abs(low_share_inside - low_share_border) >= _BORDER_MARGIN:
            return low_share_inside > low_share_border
    return low_share_inside < 0.5


def fill_holes(mask: torch.Tensor) -> torch.Tensor:
    """`mask` plus every region it encloses: the complement of what a flood from the image border
    reaches without crossing `mask` (4-connected steps, run on the mask's device)."""
    open_px = ~mask
    outside = torch.zeros_like(mask)
    outside[0, :] = open_px[0, :]
    outside[-1, :] = open_px[-1, :]
    outside[:, 0] = open_px[:, 0]
    outside[:, -1] = open_px[:, -1]
    limit = mask.shape[0] * mask.shape[1]
    steps = 0
    while steps < limit:
        before = outside
        for _ in range(_FLOOD_CHECK_EVERY):
            grown = outside.clone()
            grown[1:, :] |= outside[:-1, :]
            grown[:-1, :] |= outside[1:, :]
            grown[:, 1:] |= outside[:, :-1]
            grown[:, :-1] |= outside[:, 1:]
            outside = grown & open_px
        steps += _FLOOD_CHECK_EVERY
        if torch.equal(outside, before):
            break
    return ~outside


def outline_width(crop: torch.Tensor, body: torch.Tensor, max_px: int) -> int:
    """How many pixels beyond `body` the lettering still continues (an outline, heavy anti-aliasing).

    One-pixel rings are peeled off the body outwards; the lettering ends at the first ring whose median
    colour is within `_OUTLINE_TOL` (largest channel difference) of the background's, sampled in a band
    just beyond `max_px`. `max_px` when the background cannot be sampled or no ring matches it.
    """
    far = dilate(body, max_px + _FAR_BAND_PX) & ~dilate(body, max_px)
    if int(far.sum()) < _MIN_BORDER_PX:
        return max_px
    pixels = crop.float()
    background = pixels[:, far].median(dim=1).values
    previous = body
    for px in range(1, max_px + 1):
        grown = dilate(body, px)
        ring = grown & ~previous
        previous = grown
        if not bool(ring.any()):
            return px - 1
        if float((pixels[:, ring].median(dim=1).values - background).abs().max()) < _OUTLINE_TOL:
            return px - 1
    return max_px


def stroke_width(ink: torch.Tensor) -> float:
    """Mean stroke width of the ink: twice its area over its boundary length (a stroke of width w
    and length L has area wL and about 2L boundary pixels); 0.0 without ink."""
    area = int(ink.sum())
    if area == 0:
        return 0.0
    boundary = int((ink & ~erode(ink, 1)).sum())
    return 2.0 * area / max(boundary, 1)


def local_ring(mask: torch.Tensor, px: int) -> torch.Tensor:
    """The band of pixels within `px` of `mask` but outside it (what surrounds the removed ink)."""
    return dilate(mask, px) & ~mask

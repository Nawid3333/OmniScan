"""Flat-fill inpainting primitives: text masks from OCR line boxes and uniform-colour fills."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

# torch.quantile needs float input and < 16 M elements; above this the dev array is strided-sampled.
_QUANTILE_SAMPLE_CAP = 1_000_000


@dataclass(frozen=True, slots=True)
class FlatResult:
    """Outcome of one flat fill: the cleaned crop, the fill colour (None when not flat) and success."""

    pixels: torch.Tensor  # uint8 [3, h, w]: the input crop, unchanged where the mask is False
    fill: tuple[int, int, int] | None
    ok: bool  # True when the ring was uniform enough for a flat fill


def line_mask(
    height: int,
    width: int,
    boxes: Sequence[tuple[int, int, int, int]],
    *,
    dilate_px: int,
    device: torch.device,
) -> torch.Tensor:
    """Bool [height, width] mask: every (x0, y0, x1, y1) box grown by `dilate_px` on all sides, clamped."""
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    for x0, y0, x1, y1 in boxes:
        mask[
            max(0, y0 - dilate_px) : min(height, y1 + dilate_px),
            max(0, x0 - dilate_px) : min(width, x1 + dilate_px),
        ] = True
    return mask


def flat_fill(
    crop: torch.Tensor,
    mask: torch.Tensor,
    *,
    flat_tol: float,
    min_ring_px: int,
    ring_mask: torch.Tensor | None = None,
) -> FlatResult:
    """Replace masked pixels with the ring's median colour when the ring is uniform enough; the ring is
    `ring_mask` when given (e.g. the band around glyphs), else every unmasked pixel of the crop."""
    ring = crop[:, ~mask if ring_mask is None else ring_mask]
    if not bool(mask.any()) or ring.shape[1] < min_ring_px:
        return FlatResult(crop.clone(), None, False)
    ring_f = ring.float()
    median = ring_f.median(dim=1).values
    dev = (ring_f - median[:, None]).abs().amax(dim=0)  # per ring pixel: worst-channel deviation
    if dev.numel() > _QUANTILE_SAMPLE_CAP:
        dev = dev[:: math.ceil(dev.numel() / _QUANTILE_SAMPLE_CAP)]
    if float(torch.quantile(dev, 0.9)) > flat_tol:
        return FlatResult(crop.clone(), None, False)
    values = median.round().tolist()  # one sync for all three channels
    fill = (int(values[0]), int(values[1]), int(values[2]))
    pixels = crop.clone()
    pixels[:, mask] = torch.tensor(fill, dtype=torch.uint8, device=crop.device)[:, None]
    return FlatResult(pixels, fill, True)

"""Binary morphology and thresholding on GPU tensors (masks stay on their device, no Python pixel loops)."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 — torch's standard alias


def dilate(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Bool [h, w] mask grown by `px` on all sides (square max pool), same shape and device."""
    if px <= 0:
        return mask.clone()
    grown = F.max_pool2d(mask.float()[None, None], kernel_size=2 * px + 1, stride=1, padding=px)
    return grown[0, 0] > 0


def erode(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Bool [h, w] mask shrunk by `px` on all sides; pixels outside the image count as unset."""
    if px <= 0:
        return mask.clone()
    padded = F.pad(mask.float()[None, None], (px, px, px, px), value=0.0)
    shrunk = -F.max_pool2d(-padded, kernel_size=2 * px + 1, stride=1)
    return shrunk[0, 0] > 0


def otsu_threshold(luminance: torch.Tensor) -> float:
    """The luminance cutoff maximising between-class variance of a two-way split (Otsu's method).

    `luminance < cutoff` is the "low" side, `luminance >= cutoff` the "high" side: `torch.histc`'s
    256 bins each span exactly one luminance unit ([i, i+1) for bin i), so the bin index maximising
    variance must be offset by +1 to become a correct exclusive cutoff — a value sitting exactly on
    an integer luminance (e.g. 40.0, in bin 40) must land on the same side as the rest of its bin.
    """
    hist = torch.histc(luminance.float(), bins=256, min=0, max=256)
    weights = torch.arange(256, dtype=torch.float32, device=hist.device)
    cum_weight = torch.cumsum(hist, dim=0)
    cum_moment = torch.cumsum(hist * weights, dim=0)
    total_weight = cum_weight[-1]
    total_moment = cum_moment[-1]
    weight_lo = cum_weight.clamp(min=1e-6)
    weight_hi = (total_weight - cum_weight).clamp(min=1e-6)
    mean_lo = cum_moment / weight_lo
    mean_hi = (total_moment - cum_moment) / weight_hi
    variance = cum_weight * (total_weight - cum_weight) * (mean_lo - mean_hi) ** 2
    return float(torch.argmax(variance)) + 1.0


def luminance(pixels: torch.Tensor) -> torch.Tensor:
    """BT.601 luma of uint8/float RGB `pixels` shaped [3, ...] as float32 [...]."""
    rgb = pixels.float()
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

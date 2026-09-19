"""LaMa region pipeline: the `needs_lama` items of inpaint.json -> LaMa-cleaned patches + the new artifact."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 — torch's standard alias

from omniscan.core.config import InpaintConfig
from omniscan.core.schemas import BBox, InpaintArtifact, InpaintItem


def dilate_mask(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Bool [h, w] mask grown by `px` on all sides (max pool), same shape and device."""
    if px <= 0:
        return mask.clone()
    grown = F.max_pool2d(mask.float()[None, None], kernel_size=2 * px + 1, stride=1, padding=px)
    return grown[0, 0] > 0


def window_origin(box: BBox, strip_w: int, strip_h: int, window: int) -> tuple[int, int, int, int]:
    """(x0, y0, w, h) of the window crop centred on `box` and clamped into the strip."""
    cx = (box.x0 + box.x1) // 2
    cy = (box.y0 + box.y1) // 2
    w = min(window, strip_w)
    h = min(window, strip_h)
    x0 = min(max(cx - window // 2, 0), strip_w - w)
    y0 = min(max(cy - window // 2, 0), strip_h - h)
    return x0, y0, w, h


def lama_regions(
    strip: torch.Tensor,
    inpaint: InpaintArtifact,
    patches: Mapping[str, tuple[np.ndarray, np.ndarray]],
    inpainter: Any,
    cfg: InpaintConfig,
) -> tuple[InpaintArtifact, dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, float]]:
    """LaMa-clean every `needs_lama` region of the uint8 [3, H, W] strip: artifact + patches + metrics.

    The flat-fill mask of each such region (from `patches`) is grown by `cfg.lama_dilate_px`, a window of
    `cfg.lama_window` pixels around the region is inpainted (padded by replication when the strip is
    smaller), and the region's own patch is cut back out. Regions wider or taller than
    `cfg.lama_window - 2 * cfg.lama_context_px` are skipped (`skipped_too_large`): the window must keep
    that much context around them on every side.
    """
    items: list[InpaintItem] = []
    patches_out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    skipped = 0
    strip_h, strip_w = int(strip.shape[1]), int(strip.shape[2])
    limit = cfg.lama_window - 2 * cfg.lama_context_px
    for item in inpaint.items:
        if not item.needs_lama:
            continue
        if item.box.width > limit or item.box.height > limit:
            skipped += 1
            continue
        mask = torch.from_numpy(np.ascontiguousarray(patches[item.region_id][1])).to(strip.device)
        mask = dilate_mask(mask, cfg.lama_dilate_px)
        wx, wy, w, h = window_origin(item.box, strip_w, strip_h, cfg.lama_window)
        window = strip[:, wy : wy + h, wx : wx + w]
        if w < cfg.lama_window or h < cfg.lama_window:  # strip smaller than the window: replicate-pad
            pad_r, pad_b = cfg.lama_window - w, cfg.lama_window - h
            window = F.pad(window.float()[None], (0, pad_r, 0, pad_b), mode="replicate")
            window = window.round().to(torch.uint8)[0]
        full_mask = torch.zeros((cfg.lama_window, cfg.lama_window), dtype=torch.bool, device=strip.device)
        by0, bx0 = item.box.y0 - wy, item.box.x0 - wx
        full_mask[by0 : by0 + item.box.height, bx0 : bx0 + item.box.width] = mask
        result = inpainter.inpaint(window, full_mask)
        pixels = result[:, by0 : by0 + item.box.height, bx0 : bx0 + item.box.width]
        items.append(
            InpaintItem(
                region_id=item.region_id,
                box=item.box,
                method="lama",
                fill=None,
                needs_lama=False,
                mask_px=int(mask.sum()),
            )
        )
        patches_out[item.region_id] = (pixels, mask)
    metrics = {
        "regions": float(len(items)),
        "skipped_too_large": float(skipped),
        "mask_px": float(sum(item.mask_px for item in items)),
    }
    return InpaintArtifact(items=items), patches_out, metrics

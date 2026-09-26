"""Compositing: apply inpaint patches and blend glyph patches into the strip tensor, in place."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from omniscan.core.schemas import BBox, CleanupPatch, InpaintItem
from omniscan.typeset.render import GlyphPatch


def apply_patch(strip: torch.Tensor, box: BBox, pixels: np.ndarray, mask: np.ndarray) -> None:
    """Replace `strip`'s pixels inside `box` with `pixels` exactly where `mask` is True (in place, clipped)."""
    overlap = _overlap(box, strip.shape[1], strip.shape[2])
    if overlap is None:
        return
    x0, y0, x1, y1 = overlap
    h = min(pixels.shape[0], mask.shape[0])
    w = min(pixels.shape[1], mask.shape[1])
    py0, px0 = max(y0 - box.y0, 0), max(x0 - box.x0, 0)
    py1, px1 = min(y1 - box.y0, h), min(x1 - box.x0, w)
    if py0 >= py1 or px0 >= px1:
        return
    sel = torch.from_numpy(np.ascontiguousarray(mask[py0:py1, px0:px1])).to(strip.device)
    rgb = torch.from_numpy(np.ascontiguousarray(pixels[py0:py1, px0:px1])).permute(2, 0, 1).to(strip.device)
    target = strip[:, box.y0 + py0 : box.y0 + py1, box.x0 + px0 : box.x0 + px1]
    target.copy_(torch.where(sel, rgb, target))


def blend_rgba(strip: torch.Tensor, patch: GlyphPatch) -> None:
    """Alpha-blend a rendered glyph patch into `strip` in place (clipped; all arithmetic on the strip's device)."""
    height, width = int(strip.shape[1]), int(strip.shape[2])
    x0, y0 = max(patch.x, 0), max(patch.y, 0)
    x1 = min(patch.x + patch.rgba.shape[1], width)
    y1 = min(patch.y + patch.rgba.shape[0], height)
    if x0 >= x1 or y0 >= y1:
        return
    rgba = np.ascontiguousarray(patch.rgba[y0 - patch.y : y1 - patch.y, x0 - patch.x : x1 - patch.x])
    source = torch.from_numpy(rgba).to(strip.device)  # one host->device upload per patch
    alpha = source[..., 3].float() / 255.0
    src = source[..., :3].permute(2, 0, 1).float()
    target = strip[:, y0:y1, x0:x1]
    blended = torch.round(src * alpha + target.float() * (1.0 - alpha)).clamp_(0, 255).to(torch.uint8)
    target.copy_(blended)


def apply_patches(
    strip: torch.Tensor,
    items: Sequence[InpaintItem],
    patches: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> int:
    """Apply every item's patch that has an entry in `patches`, in item order; returns how many were applied."""
    applied = 0
    for item in items:
        entry = patches.get(item.region_id)
        if entry is None:
            continue
        apply_patch(strip, item.box, entry[0], entry[1])
        applied += 1
    return applied


def snapshot(strip: torch.Tensor, box: BBox) -> np.ndarray:
    """A copy of the strip's pixels inside `box` as uint8 [h, w, 3] (`box` must lie inside the strip). Always
    a copy: on a CPU strip `.cpu().numpy()` alone would share memory and follow later edits of the strip."""
    return strip[:, box.y0 : box.y1, box.x0 : box.x1].permute(1, 2, 0).cpu().numpy().copy()


def apply_cleanup(
    strip: torch.Tensor,
    patches: Sequence[CleanupPatch],
    arrays: Mapping[str, tuple[np.ndarray | None, np.ndarray]],
    originals: Mapping[str, np.ndarray],
) -> int:
    """Apply the hand cleanup in painting order: stored pixels, or for "restore" the raw pixels `originals`
    holds (snapshots taken before any automatic patch); returns how many were applied."""
    applied = 0
    for patch in patches:
        entry = arrays.get(patch.id)
        if entry is None:
            continue
        pixels, mask = entry
        if pixels is None:
            pixels = originals.get(patch.id)
            if pixels is None:
                continue
        apply_patch(strip, patch.box, pixels, mask)
        applied += 1
    return applied


def _overlap(box: BBox, height: int, width: int) -> tuple[int, int, int, int] | None:
    """Intersection of `box` with a strip of [height, width] as (x0, y0, x1, y1); None when fully outside."""
    x0, y0 = max(box.x0, 0), max(box.y0, 0)
    x1, y1 = min(box.x1, width), min(box.y1, height)
    return None if x0 >= x1 or y0 >= y1 else (x0, y0, x1, y1)

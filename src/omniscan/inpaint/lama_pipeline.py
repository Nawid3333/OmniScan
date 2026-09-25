"""LaMa region pipeline: the `needs_lama` items of inpaint.json -> LaMa-cleaned patches + the new artifact."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 — torch's standard alias

from omniscan.core.config import InpaintConfig
from omniscan.core.schemas import BBox, InpaintArtifact, InpaintItem
from omniscan.gpu.morph import dilate as dilate_mask


def window_origin(box: BBox, strip_w: int, strip_h: int, window: int) -> tuple[int, int, int, int]:
    """(x0, y0, w, h) of the window crop centred on `box` and clamped into the strip."""
    cx = (box.x0 + box.x1) // 2
    cy = (box.y0 + box.y1) // 2
    w = min(window, strip_w)
    h = min(window, strip_h)
    x0 = min(max(cx - window // 2, 0), strip_w - w)
    y0 = min(max(cy - window // 2, 0), strip_h - h)
    return x0, y0, w, h


_TILE_OVERLAP_PX = (
    32  # adjacent tiles of an over-size region share this many pixels of mask on the split axis
)


def _tile_starts(length: int, limit: int, overlap_px: int) -> list[int]:
    """Start offsets covering `length` in steps of at most `limit`, consecutive steps overlapping by `overlap_px`."""
    if length <= limit:
        return [0]
    step = max(1, limit - overlap_px)
    starts = list(range(0, length - limit, step))
    starts.append(length - limit)  # the last tile is flush with the far edge, not necessarily `step`-spaced
    return starts


def _tile_boxes(box: BBox, limit: int, overlap_px: int) -> list[BBox]:
    """Split `box` into sub-boxes at most `limit` px on a side; a box already inside `limit` is returned whole.

    Real full-width narration captions (measured on Solo Leveling: a 513 px wide box against a 448 px
    limit) are wider than one `lama_window` can hold with its required context margin. Tiling instead of
    skipping keeps the same fixed `lama_window` (no extra model warm-up) and covers the whole mask.
    """
    return [
        BBox(
            x0=box.x0 + x,
            y0=box.y0 + y,
            x1=box.x0 + x + min(limit, box.width - x),
            y1=box.y0 + y + min(limit, box.height - y),
        )
        for y in _tile_starts(box.height, limit, overlap_px)
        for x in _tile_starts(box.width, limit, overlap_px)
    ]


def _lama_crop(
    strip: torch.Tensor,
    box: BBox,
    mask: torch.Tensor,
    inpainter: Any,
    cfg: InpaintConfig,
    strip_w: int,
    strip_h: int,
) -> torch.Tensor:
    """Inpaint `box` (already `<= cfg.lama_window` on a side) through one window; returns its cleaned pixels."""
    wx, wy, w, h = window_origin(box, strip_w, strip_h, cfg.lama_window)
    window = strip[:, wy : wy + h, wx : wx + w]
    if w < cfg.lama_window or h < cfg.lama_window:  # strip smaller than the window: replicate-pad
        pad_r, pad_b = cfg.lama_window - w, cfg.lama_window - h
        window = F.pad(window.float()[None], (0, pad_r, 0, pad_b), mode="replicate")
        window = window.round().to(torch.uint8)[0]
    full_mask = torch.zeros((cfg.lama_window, cfg.lama_window), dtype=torch.bool, device=strip.device)
    by0, bx0 = box.y0 - wy, box.x0 - wx
    full_mask[by0 : by0 + box.height, bx0 : bx0 + box.width] = mask
    result = inpainter.inpaint(window, full_mask)
    return result[:, by0 : by0 + box.height, bx0 : bx0 + box.width]


def lama_regions(
    strip: torch.Tensor,
    inpaint: InpaintArtifact,
    patches: Mapping[str, tuple[np.ndarray, np.ndarray]],
    inpainter: Any,
    cfg: InpaintConfig,
) -> tuple[InpaintArtifact, dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, float]]:
    """LaMa-clean every `needs_lama` region of the uint8 [3, H, W] strip: artifact + patches + metrics.

    The flat-fill mask of each such region (from `patches`) is grown by `cfg.lama_dilate_px`, then a window
    of `cfg.lama_window` pixels around the region is inpainted (padded by replication when the strip is
    smaller) and the region's own patch is cut back out. A region wider or taller than
    `cfg.lama_window - 2 * cfg.lama_context_px` (the window must keep that much context around it on every
    side) is split into overlapping tiles of at most that size, each inpainted through its own
    `lama_window` crop, and stitched back into one patch — no region is skipped (`skipped_too_large` in the
    metrics is kept, always 0.0, for older callers of this function's return shape).
    """
    items: list[InpaintItem] = []
    patches_out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    strip_h, strip_w = int(strip.shape[1]), int(strip.shape[2])
    limit = cfg.lama_window - 2 * cfg.lama_context_px
    for item in inpaint.items:
        if not item.needs_lama:
            continue
        box = item.box
        mask = torch.from_numpy(np.ascontiguousarray(patches[item.region_id][1])).to(strip.device)
        mask = dilate_mask(mask, cfg.lama_dilate_px)
        if box.width <= limit and box.height <= limit:
            pixels = _lama_crop(strip, box, mask, inpainter, cfg, strip_w, strip_h)
        else:
            pixels = torch.zeros((3, box.height, box.width), dtype=strip.dtype, device=strip.device)
            for tile in _tile_boxes(box, limit, _TILE_OVERLAP_PX):
                tx0, ty0 = tile.x0 - box.x0, tile.y0 - box.y0
                tile_mask = mask[ty0 : ty0 + tile.height, tx0 : tx0 + tile.width]
                pixels[:, ty0 : ty0 + tile.height, tx0 : tx0 + tile.width] = _lama_crop(
                    strip, tile, tile_mask, inpainter, cfg, strip_w, strip_h
                )
        items.append(
            InpaintItem(
                region_id=item.region_id,
                box=box,
                method="lama",
                fill=None,
                needs_lama=False,
                mask_px=int(mask.sum()),
            )
        )
        patches_out[item.region_id] = (pixels, mask)
    metrics = {
        "regions": float(len(items)),
        "skipped_too_large": 0.0,
        "mask_px": float(sum(item.mask_px for item in items)),
    }
    return InpaintArtifact(items=items), patches_out, metrics

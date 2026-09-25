"""LaMa region pipeline: the `needs_lama` items of inpaint.json -> LaMa-cleaned patches + the new artifact."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def _window_mask(
    wx: int, wy: int, masks: Sequence[tuple[BBox, torch.Tensor]], size: int, device: torch.device
) -> torch.Tensor:
    """Bool [size, size] mask of the window at (wx, wy): every region mask of `masks` that reaches into it."""
    full = torch.zeros((size, size), dtype=torch.bool, device=device)
    for box, mask in masks:
        x0, y0 = max(box.x0, wx), max(box.y0, wy)
        x1, y1 = min(box.x1, wx + size), min(box.y1, wy + size)
        if x0 < x1 and y0 < y1:
            full[y0 - wy : y1 - wy, x0 - wx : x1 - wx] |= mask[
                y0 - box.y0 : y1 - box.y0, x0 - box.x0 : x1 - box.x0
            ]
    return full


def _lama_crop(
    strip: torch.Tensor,
    tile: BBox,
    masks: Sequence[tuple[BBox, torch.Tensor]],
    inpainter: Any,
    cfg: InpaintConfig,
) -> torch.Tensor:
    """Inpaint `tile` (at most `cfg.lama_window` on a side) through one window whose mask holds every
    region mask of `masks` inside it; returns the tile's pixels."""
    strip_h, strip_w = int(strip.shape[1]), int(strip.shape[2])
    wx, wy, w, h = window_origin(tile, strip_w, strip_h, cfg.lama_window)
    window = strip[:, wy : wy + h, wx : wx + w].clone()  # the working strip changes as regions finish
    if w < cfg.lama_window or h < cfg.lama_window:  # strip smaller than the window: replicate-pad
        pad_r, pad_b = cfg.lama_window - w, cfg.lama_window - h
        window = F.pad(window.float()[None], (0, pad_r, 0, pad_b), mode="replicate")
        window = window.round().to(torch.uint8)[0]
    full_mask = _window_mask(wx, wy, masks, cfg.lama_window, strip.device)
    result = inpainter.inpaint(window, full_mask)
    by0, bx0 = tile.y0 - wy, tile.x0 - wx
    return result[:, by0 : by0 + tile.height, bx0 : bx0 + tile.width]


def _paste(target: torch.Tensor, box: BBox, pixels: torch.Tensor, mask: torch.Tensor) -> None:
    """Write `pixels` into `target` inside `box` where `mask` is set (in place)."""
    region = target[:, box.y0 : box.y1, box.x0 : box.x1]
    region.copy_(torch.where(mask, pixels, region))


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

    No lettering may serve as context: LaMa works on a copy of the strip with the flat-filled regions
    already cleaned and each finished region written back, and every window masks all the not yet
    cleaned regions it reaches — the rest of a tiled region included (masking only the tile's part made
    LaMa copy the neighbouring letters' colour into the hole).
    """
    items: list[InpaintItem] = []
    patches_out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    limit = cfg.lama_window - 2 * cfg.lama_context_px
    work = strip.clone()
    for item in inpaint.items:
        if item.needs_lama or item.region_id not in patches:
            continue
        pixels_np, mask_np = patches[item.region_id]
        pixels = torch.from_numpy(np.ascontiguousarray(pixels_np)).permute(2, 0, 1).to(strip.device)
        _paste(work, item.box, pixels, torch.from_numpy(np.ascontiguousarray(mask_np)).to(strip.device))
    pending = [
        (
            item,
            dilate_mask(
                torch.from_numpy(np.ascontiguousarray(patches[item.region_id][1])).to(strip.device),
                cfg.lama_dilate_px,
            ),
        )
        for item in inpaint.items
        if item.needs_lama
    ]
    for index, (item, mask) in enumerate(pending):
        box = item.box
        still_dirty = [(other.box, other_mask) for other, other_mask in pending[index:]]
        tiles = (
            [box] if box.width <= limit and box.height <= limit else _tile_boxes(box, limit, _TILE_OVERLAP_PX)
        )
        cleaned = torch.zeros((3, box.height, box.width), dtype=strip.dtype, device=strip.device)
        for tile in tiles:
            tx0, ty0 = tile.x0 - box.x0, tile.y0 - box.y0
            cleaned[:, ty0 : ty0 + tile.height, tx0 : tx0 + tile.width] = _lama_crop(
                work, tile, still_dirty, inpainter, cfg
            )
        pixels = torch.where(mask, cleaned, strip[:, box.y0 : box.y1, box.x0 : box.x1])
        _paste(work, box, pixels, mask)
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

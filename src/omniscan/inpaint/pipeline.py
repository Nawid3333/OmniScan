"""Region inpainting pipeline: OCR regions over the strip -> cleaned patches + the InpaintArtifact."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from omniscan.core.config import InpaintConfig
from omniscan.core.schemas import BBox, InpaintArtifact, InpaintItem, OcrLine, Region
from omniscan.inpaint.flat import flat_fill, line_mask


def _union(lines: Sequence[OcrLine]) -> BBox:
    """Smallest box containing every line box of a region."""
    return BBox(
        x0=min(line.bbox.x0 for line in lines),
        y0=min(line.bbox.y0 for line in lines),
        x1=max(line.bbox.x1 for line in lines),
        y1=max(line.bbox.y1 for line in lines),
    )


def inpaint_regions(
    strip: torch.Tensor, regions: Sequence[Region], cfg: InpaintConfig
) -> tuple[InpaintArtifact, dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, float]]:
    """Clean every region's text from the uint8 [3, H, W] strip: artifact + patches + metrics.

    Watermarks and regions without lines are skipped entirely; sfx and regions whose ring is not a
    flat colour keep their pixels and get `needs_lama=True`.
    """
    items: list[InpaintItem] = []
    patches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    strip_height, strip_width = int(strip.shape[1]), int(strip.shape[2])
    for region in regions:
        if region.kind == "watermark" or not region.lines:
            continue
        union = _union(region.lines)
        patch = BBox(
            x0=max(0, union.x0 - cfg.pad_px),
            y0=max(0, union.y0 - cfg.pad_px),
            x1=min(strip_width, union.x1 + cfg.pad_px),
            y1=min(strip_height, union.y1 + cfg.pad_px),
        )
        crop = strip[:, patch.y0 : patch.y1, patch.x0 : patch.x1]
        boxes = [
            (
                line.bbox.x0 - patch.x0,
                line.bbox.y0 - patch.y0,
                line.bbox.x1 - patch.x0,
                line.bbox.y1 - patch.y0,
            )
            for line in region.lines
        ]
        mask = line_mask(
            crop.shape[1], crop.shape[2], boxes, dilate_px=cfg.mask_dilate_px, device=strip.device
        )
        mask_px = int(mask.sum())
        if region.kind == "sfx":
            item = InpaintItem(
                region_id=region.id, box=patch, method="none", fill=None, needs_lama=True, mask_px=mask_px
            )
            pixels = crop.clone()
        else:
            result = flat_fill(crop, mask, flat_tol=cfg.flat_tol, min_ring_px=cfg.min_ring_px)
            item = InpaintItem(
                region_id=region.id,
                box=patch,
                method="flat" if result.ok else "none",
                fill=result.fill if result.ok else None,
                needs_lama=not result.ok,
                mask_px=mask_px,
            )
            pixels = result.pixels
        items.append(item)
        patches[region.id] = (pixels, mask)
    metrics = {
        "regions": float(len(items)),
        "flat": float(sum(1 for item in items if item.method == "flat")),
        "needs_lama": float(sum(1 for item in items if item.needs_lama)),
        "skipped": float(len(regions) - len(items)),
        "mask_px": float(sum(item.mask_px for item in items)),
    }
    return InpaintArtifact(items=items), patches, metrics

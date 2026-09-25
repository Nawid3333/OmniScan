"""Region inpainting pipeline: OCR regions over the strip -> cleaned patches + the InpaintArtifact."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from omniscan.core.config import InpaintConfig
from omniscan.core.schemas import BBox, InpaintArtifact, InpaintItem, OcrLine, Region
from omniscan.inpaint.flat import flat_fill, line_mask
from omniscan.inpaint.glyph_mask import find_glyphs, local_ring


def _union(lines: Sequence[OcrLine]) -> BBox:
    """Smallest box containing every line box of a region."""
    return BBox(
        x0=min(line.bbox.x0 for line in lines),
        y0=min(line.bbox.y0 for line in lines),
        x1=max(line.bbox.x1 for line in lines),
        y1=max(line.bbox.y1 for line in lines),
    )


def inpaint_regions(
    strip: torch.Tensor,
    regions: Sequence[Region],
    cfg: InpaintConfig,
    *,
    sfx_mode: str = "replace",
) -> tuple[InpaintArtifact, dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, float]]:
    """Clean every region's text from the uint8 [3, H, W] strip: artifact + patches + metrics.

    Watermarks and regions without lines are skipped entirely, and so are sfx regions unless
    `sfx_mode` is "replace" (the original art stays). Per region, cheapest clean result first:

    1. the whole line-box mask flat-filled when the ring around it is one colour (bubble interiors;
       never for sfx, whose boxes are mostly art);
    2. with `cfg.glyph_mask`, only the glyphs (inpaint/glyph_mask.py) flat-filled when the band around
       them is one colour (text touching a bubble outline, lettering on a flat panel);
    3. otherwise `needs_lama=True`, with the glyph mask when one was found (LaMa then only rebuilds the
       lettering, not the art around it) or the line-box mask.
    """
    items: list[InpaintItem] = []
    patches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    strip_height, strip_width = int(strip.shape[1]), int(strip.shape[2])
    glyph_masks = 0
    for region in regions:
        if region.kind == "watermark" or not region.lines:
            continue
        if region.kind == "sfx" and sfx_mode != "replace":
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
        result = None
        if region.kind != "sfx":
            result = flat_fill(crop, mask, flat_tol=cfg.flat_tol, min_ring_px=cfg.min_ring_px)
        if (result is None or not result.ok) and cfg.glyph_mask:
            glyphs = find_glyphs(
                crop,
                boxes,
                grow=cfg.glyph_grow_sfx if region.kind == "sfx" else cfg.glyph_grow,
                min_grow_px=cfg.glyph_grow_min_px,
                max_grow_px=cfg.glyph_grow_max_px,
            )
            if glyphs is not None:
                glyph_masks += 1
                mask = glyphs.mask
                result = flat_fill(
                    crop,
                    mask,
                    flat_tol=cfg.flat_tol,
                    min_ring_px=cfg.min_ring_px,
                    ring_mask=local_ring(mask, cfg.glyph_ring_px),
                )
        ok = result is not None and result.ok
        item = InpaintItem(
            region_id=region.id,
            box=patch,
            method="flat" if ok else "none",
            fill=result.fill if result is not None and ok else None,
            needs_lama=not ok,
            mask_px=int(mask.sum()),
        )
        items.append(item)
        patches[region.id] = (result.pixels if result is not None and ok else crop.clone(), mask)
    metrics = {
        "regions": float(len(items)),
        "flat": float(sum(1 for item in items if item.method == "flat")),
        "needs_lama": float(sum(1 for item in items if item.needs_lama)),
        "skipped": float(len(regions) - len(items)),
        "mask_px": float(sum(item.mask_px for item in items)),
        "glyph_masks": float(glyph_masks),
    }
    return InpaintArtifact(items=items), patches, metrics

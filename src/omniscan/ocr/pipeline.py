"""OCR pipeline: strip + regions -> text, two ways (card O1b).

`read_regions` (ppocr): tiles -> line boxes -> assignment -> line crops -> readings -> regions.
`read_region_crops` (manga_ocr, paddleocr_vl): one padded crop per region -> readings -> regions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch

from omniscan.core.config import OcrConfig
from omniscan.core.schemas import Region
from omniscan.detect.postprocess import Box
from omniscan.detect.tiles import keep_tiles, plan_tiles
from omniscan.ocr.assemble import build_ocr_regions
from omniscan.ocr.crop_readers import TextReader
from omniscan.ocr.lines import LineBox, assign_lines, merge_lines
from omniscan.ocr.model import LineDetector, LineRecognizer

_TILE_BATCH = 4  # tiles handed to the detector per call (the detector itself runs one tile per forward)


def read_regions(
    strip: torch.Tensor,
    regions: Sequence[Region],
    detector: LineDetector,
    recognizer: LineRecognizer,
    cfg: OcrConfig,
    *,
    direction: Literal["ltr", "rtl"],
    engine: str,
) -> tuple[list[Region], dict[str, float]]:
    """Read the text of `regions` from the strip tensor; returns (regions with text, metrics)."""
    height, width = int(strip.shape[-2]), int(strip.shape[-1])
    tiles = keep_tiles(
        plan_tiles(width, height, cfg.tile_px, cfg.overlap), [(r.bbox.y0, r.bbox.y1) for r in regions]
    )

    shifted: list[LineBox] = []
    for start in range(0, len(tiles), _TILE_BATCH):
        batch = tiles[start : start + _TILE_BATCH]
        crops = [strip[:, t.y0 : t.y1, t.x0 : t.x1] for t in batch]  # views, never copies
        for tile, lines in zip(batch, detector.detect(crops), strict=True):
            shifted.extend(
                LineBox(_shift(line.box, float(tile.x0), float(tile.y0)), line.score) for line in lines
            )

    lines = merge_lines(shifted, nms_iou=cfg.nms_iou)
    by_region, orphans = assign_lines(
        regions, lines, min_ioa=cfg.assign_min_ioa, pad_px=cfg.region_pad_px, direction=direction
    )

    crop_list: list[torch.Tensor] = []
    keys: list[tuple[str, int]] = []
    for region in regions:
        for index, line in enumerate(by_region.get(region.id, ())):
            x0, y0, x1, y1 = _crop_box(line.box, cfg.line_pad_px, width, height)
            crop_list.append(strip[:, y0:y1, x0:x1])  # view, never a copy
            keys.append((region.id, index))
    readings = dict(zip(keys, recognizer.read(crop_list), strict=True))

    out = build_ocr_regions(
        regions,
        by_region,
        readings,
        strip,
        engine=engine,
        lang=cfg.lang,
        strip_width=width,
        strip_height=height,
    )
    metrics = {
        "tiles": float(len(tiles)),
        "lines": float(len(lines)),
        "orphan_lines": float(len(orphans)),
        "regions": float(len(regions)),
        "regions_empty": float(sum(1 for region in out if not region.lines)),
        "regions_low_conf": float(
            sum(1 for region in out if region.lines and region.confidence < cfg.low_conf)
        ),
    }
    return out, metrics


def read_region_crops(
    strip: torch.Tensor,
    regions: Sequence[Region],
    reader: TextReader,
    cfg: OcrConfig,
    *,
    engine: str,
) -> tuple[list[Region], dict[str, float]]:
    """Read whole region crops (manga_ocr and friends): one padded crop per region, no line detection."""
    height, width = int(strip.shape[-2]), int(strip.shape[-1])
    crop_list: list[torch.Tensor] = []
    for region in regions:
        x0, y0, x1, y1 = _crop_box(
            (float(region.bbox.x0), float(region.bbox.y0), float(region.bbox.x1), float(region.bbox.y1)),
            cfg.crop_pad_px,
            width,
            height,
        )
        crop_list.append(strip[:, y0:y1, x0:x1])  # view, never a copy
    readings = {
        (region.id, 0): reading for region, reading in zip(regions, reader.read(crop_list), strict=True)
    }
    by_region = {
        region.id: [
            LineBox(
                box=(
                    float(region.bbox.x0),
                    float(region.bbox.y0),
                    float(region.bbox.x1),
                    float(region.bbox.y1),
                ),
                score=1.0,
            )
        ]
        for region in regions
    }
    out = build_ocr_regions(
        regions,
        by_region,
        readings,
        strip,
        engine=engine,
        lang=cfg.lang,
        strip_width=width,
        strip_height=height,
    )
    metrics = {
        "tiles": 0.0,
        "lines": float(len(regions)),
        "orphan_lines": 0.0,
        "regions": float(len(regions)),
        "regions_empty": float(sum(1 for region in out if not region.lines)),
        "regions_low_conf": float(
            sum(1 for region in out if region.lines and region.confidence < cfg.low_conf)
        ),
    }
    return out, metrics


def _shift(box: Box, dx: float, dy: float) -> Box:
    """Move a line box from tile pixels to strip pixels."""
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def _crop_box(box: Box, pad: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Padded line box as integer pixel bounds: floor/ceil, clamped to the strip, at least 1x1."""
    x0 = max(0, min(math.floor(box[0] - pad), width - 1))
    y0 = max(0, min(math.floor(box[1] - pad), height - 1))
    x1 = min(width, max(x0 + 1, math.ceil(box[2] + pad)))
    y1 = min(height, max(y0 + 1, math.ceil(box[3] + pad)))
    return x0, y0, x1, y1

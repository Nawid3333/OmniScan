"""One page as the finished release will show it, rendered on the CPU for the studio (no torch).

Raw page, automatic cleaning, hand cleanup (cleanup/store.py::current_crop), then every layout item drawn
by the same rasteriser export uses (typeset/render.py) — a preview, not the export: the strip export
encodes is decoded and resized on the GPU, so single pixels may differ.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from omniscan.cleanup.store import current_crop
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import BBox, IngestArtifact, LayoutItem
from omniscan.typeset.render import render_item


def page_box(ingest: IngestArtifact, page: int) -> BBox:
    """The strip rows of page `page` (its SourceFile index); ValueError for an unknown page."""
    source = next((file for file in ingest.files if file.index == page), None)
    if source is None:
        raise ValueError(f"no page with index {page} in ingest.json")
    return BBox(x0=0, y0=source.y0, x1=ingest.strip_width, y1=source.y1)


def blend(canvas: np.ndarray, x: int, y: int, rgba: np.ndarray) -> None:
    """Alpha-blend straight-alpha `rgba` into `canvas` (uint8 [h, w, 3]) with its top-left at (x, y), clipped."""
    h, w = canvas.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + rgba.shape[1], w), min(y + rgba.shape[0], h)
    if x0 >= x1 or y0 >= y1:
        return
    src = rgba[y0 - y : y1 - y, x0 - x : x1 - x].astype(np.float32)
    alpha = src[..., 3:4] / 255.0
    target = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = np.clip(np.round(src[..., :3] * alpha + target * (1.0 - alpha)), 0, 255).astype(
        np.uint8
    )


def render_page(
    paths: ChapterPaths, ingest: IngestArtifact, page: int, items: Sequence[LayoutItem]
) -> np.ndarray:
    """Page `page` at strip resolution (uint8 [h, strip_width, 3]): cleaned, then lettered with `items`."""
    box = page_box(ingest, page)
    canvas = current_crop(paths, ingest, box)
    for item in items:
        if item.box.y1 + item.stroke_px + 2 < box.y0 or item.box.y0 - item.stroke_px - 2 > box.y1:
            continue  # nowhere near this page (a rotated item may reach a little beyond its box)
        glyph = render_item(item)
        if glyph is not None:
            blend(canvas, glyph.x, glyph.y - box.y0, glyph.rgba)
    return canvas

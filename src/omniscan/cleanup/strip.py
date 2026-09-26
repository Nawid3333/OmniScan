"""Strip-space crops of a chapter's raw pages without torch (Pillow), for the editing tools.

The strip itself is built on the GPU (ingest/strip.py); an edit only needs the few rows under a brush
stroke, read from the same files and scaled to the strip width the same way (antialiased bilinear).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from omniscan.core.paths import ChapterPaths, jpeg_paths
from omniscan.core.schemas import BBox, IngestArtifact


@lru_cache(maxsize=8)
def _page(path: str, mtime_ns: int, width: int, height: int) -> np.ndarray:
    """One page as uint8 [height, width, 3], scaled when its size differs (cached per file version)."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(rgb).copy()


def strip_crop(paths: ChapterPaths, ingest: IngestArtifact, box: BBox) -> np.ndarray:
    """The strip's pixels inside `box` (uint8 [h, w, 3]) read from the chapter's page files; rows no page
    covers are black. FileNotFoundError when a page file is missing."""
    crop = np.zeros((box.height, box.width, 3), dtype=np.uint8)
    files = jpeg_paths(ingest, paths.raw_dir, paths.work_dir / "converted")
    for page, path in zip(ingest.files, files, strict=True):
        y0, y1 = max(box.y0, page.y0), min(box.y1, page.y1)
        if y0 >= y1:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"page file {Path(path).name} not found")
        pixels = _page(str(path), path.stat().st_mtime_ns, ingest.strip_width, page.y1 - page.y0)
        crop[y0 - box.y0 : y1 - box.y0] = pixels[y0 - page.y0 : y1 - page.y0, box.x0 : box.x1]
    return crop

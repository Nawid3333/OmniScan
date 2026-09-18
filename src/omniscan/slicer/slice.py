"""Top-level slice extraction from a chapter strip."""

from __future__ import annotations

import torch

from omniscan.core.config import SlicerConfig
from omniscan.core.schemas import Slice, SlicesArtifact, SourceFile
from omniscan.slicer.bands import find_uniform_bands, row_stats
from omniscan.slicer.cuts import plan_cuts


def slice_strip(
    strip: torch.Tensor, cfg: SlicerConfig, source_files: list[SourceFile] | None = None
) -> SlicesArtifact:
    """Cut a uint8 [3, H, W] strip into slices, flagging blanks and forced cuts."""
    height, width = strip.shape[1], strip.shape[2]
    stats = row_stats(strip, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    cuts = plan_cuts(height, bands, stats.detail, cfg)

    boundaries = [0, *(c.y for c in cuts), height]
    slices: list[Slice] = []
    for i in range(len(boundaries) - 1):
        y0, y1 = boundaries[i], boundaries[i + 1]
        forced = cuts[i].forced if i < len(cuts) else False
        blank = bool(stats.uniform[y0:y1].all().item())
        files = [f.index for f in source_files if f.y0 < y1 and f.y1 > y0] if source_files is not None else []
        slices.append(Slice(index=i, y0=y0, y1=y1, blank=blank, forced_cut=forced, source_files=files))

    return SlicesArtifact(
        strip_width=width,
        strip_height=height,
        bands=bands,
        slices=slices,
        params=cfg.model_dump(),
    )

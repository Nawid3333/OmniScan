"""Slicer strategies: four ways to cut a chapter strip behind one dispatcher (card S2).

`smart` is the default (uniform bands + cut planning), `page` passes raw pages through, `fixed`
cuts every target_height rows, `simple_gutter` is the owner's gutter script as a baseline. A
strategy is chosen per series (`series.toml` `[slicer] strategy`); `slice_with_strategy` is what
stages and the CLI call.
"""

from __future__ import annotations

import torch

from omniscan.core.config import SlicerConfig
from omniscan.core.schemas import Slice, SlicesArtifact, SourceFile
from omniscan.slicer.bands import row_stats
from omniscan.slicer.page_mode import slice_by_pages
from omniscan.slicer.slice import slice_strip

STRATEGIES: tuple[str, ...] = ("smart", "page", "fixed", "simple_gutter")


def slice_with_strategy(
    strip: torch.Tensor,
    cfg: SlicerConfig,
    source_files: list[SourceFile] | None = None,
    *,
    strategy: str | None = None,
) -> SlicesArtifact:
    """Cut the strip with the named strategy (`strategy=None` follows cfg.strategy)."""
    name = cfg.strategy if strategy is None else strategy
    if name not in STRATEGIES:
        raise ValueError(f"unknown slicer strategy {name!r}; choose one of {', '.join(STRATEGIES)}")
    if name == "smart":
        artifact = slice_strip(strip, cfg, source_files)
        artifact.params = {**artifact.params, "strategy": name}
        return artifact
    if name == "page":
        return slice_page(strip, cfg, source_files)
    if name == "fixed":
        return slice_fixed(strip, cfg, source_files)
    return slice_simple_gutter(strip, cfg, source_files)


def slice_page(
    strip: torch.Tensor, cfg: SlicerConfig, source_files: list[SourceFile] | None = None
) -> SlicesArtifact:
    """One slice per source file; falls back to `smart` when the strip has no page layout."""
    height, width = strip.shape[1], strip.shape[2]
    if not source_files and height > 0:
        artifact = slice_strip(strip, cfg, source_files)
        artifact.params = {**artifact.params, "strategy": "page", "fallback": "smart"}
        return artifact
    pages = slice_by_pages(source_files or [], width, height)
    boundaries = [(s.y0, s.y1, s.forced_cut) for s in pages.slices]
    return _finalize(strip, cfg, boundaries, source_files, "page")


def slice_fixed(
    strip: torch.Tensor, cfg: SlicerConfig, source_files: list[SourceFile] | None = None
) -> SlicesArtifact:
    """Cut every cfg.target_height rows; a tail shorter than cfg.min_height merges into the previous slice."""
    height = strip.shape[1]
    boundaries: list[tuple[int, int, bool]] = []
    pos = 0
    while height - pos > cfg.target_height:
        boundaries.append((pos, pos + cfg.target_height, True))
        pos += cfg.target_height
    if height > pos:
        boundaries.append((pos, height, False))
    if len(boundaries) > 1 and height - pos < cfg.min_height:
        y0, _, forced = boundaries[-2]
        boundaries = [*boundaries[:-2], (y0, height, forced)]
    return _finalize(strip, cfg, boundaries, source_files, "fixed")


def gutter_cut_rows(strip: torch.Tensor, cfg: SlicerConfig) -> list[int]:
    """Centre row of every maximal low-variance gutter run at least cfg.gutter_min_rows tall."""
    luma = strip.float().mean(0)  # [H, W]
    gutter = luma.var(dim=1, unbiased=False) < cfg.gutter_variance  # bool [H]
    padded = torch.cat(
        (
            torch.zeros(1, dtype=torch.bool, device=strip.device),
            gutter,
            torch.zeros(1, dtype=torch.bool, device=strip.device),
        )
    )
    change = padded[1:].to(torch.int8) - padded[:-1].to(torch.int8)  # +1 at run starts, -1 at run ends
    starts = torch.nonzero(change == 1).squeeze(1).tolist()  # original start row of every run
    ends = torch.nonzero(change == -1).squeeze(1).tolist()  # exclusive end row of every run
    return [(s + e) // 2 for s, e in zip(starts, ends, strict=True) if e - s >= cfg.gutter_min_rows]


def slice_simple_gutter(
    strip: torch.Tensor, cfg: SlicerConfig, source_files: list[SourceFile] | None = None
) -> SlicesArtifact:
    """Cut at the gutter run nearest target_height inside every [min_height, max_height] window."""
    height = strip.shape[1]
    cut_rows = gutter_cut_rows(strip, cfg)
    boundaries: list[tuple[int, int, bool]] = []
    pos = 0
    while height - pos > cfg.max_height:
        lo, hi, target = pos + cfg.min_height, pos + cfg.max_height, pos + cfg.target_height
        candidates = [c for c in cut_rows if lo <= c <= hi]
        if candidates:
            cut = min(candidates, key=lambda c: (abs(c - target), c))
            boundaries.append((pos, cut, False))
            pos = cut
        else:
            boundaries.append((pos, target, True))
            pos = target
    if height > pos:
        boundaries.append((pos, height, False))
    return _finalize(strip, cfg, boundaries, source_files, "simple_gutter")


def _finalize(
    strip: torch.Tensor,
    cfg: SlicerConfig,
    boundaries: list[tuple[int, int, bool]],
    source_files: list[SourceFile] | None,
    name: str,
) -> SlicesArtifact:
    """Build the artifact from (y0, y1, forced) boundaries: blank flags, source-file mapping, params."""
    stats = row_stats(strip, cfg.uniform_tol)
    slices = [
        Slice(
            index=i,
            y0=y0,
            y1=y1,
            blank=bool(stats.uniform[y0:y1].all().item()),
            forced_cut=forced,
            source_files=[f.index for f in source_files if f.y0 < y1 and f.y1 > y0]
            if source_files is not None
            else [],
        )
        for i, (y0, y1, forced) in enumerate(boundaries)
    ]
    return SlicesArtifact(
        strip_width=strip.shape[2],
        strip_height=strip.shape[1],
        bands=[],
        slices=slices,
        params={**cfg.model_dump(), "strategy": name},
    )

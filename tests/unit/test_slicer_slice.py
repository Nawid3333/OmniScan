"""Unit tests for slice_strip."""

import time

import pytest
import torch

from omniscan.core.config import SlicerConfig
from omniscan.core.schemas import SourceFile
from omniscan.slicer.bands import find_uniform_bands, row_stats
from omniscan.slicer.cuts import plan_cuts
from omniscan.slicer.slice import slice_strip
from tests.fixtures.strips import art, bubble, solid, stack

W = 720


def test_cuts_never_inside_bubbles() -> None:
    cfg = SlicerConfig()
    block_h = 3000
    n = 4
    blocks = []
    block_ranges: list[tuple[int, int]] = []
    cursor = 0
    for k in range(n):
        blocks.append(bubble(block_h, W, seed=100 + k))
        block_ranges.append((cursor, cursor + block_h))
        cursor += block_h
        if k < n - 1:
            blocks.append(solid(60, W, (255, 255, 255)))
            cursor += 60
    strip = stack(*blocks)

    stats = row_stats(strip, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    cuts = plan_cuts(strip.shape[1], bands, stats.detail, cfg)
    assert cuts
    for cut in cuts:
        assert not any(y0 <= cut.y < y1 for y0, y1 in block_ranges)


def test_blank_and_forced_flags_match_cuts() -> None:
    cfg = SlicerConfig()

    # A long uniform block becomes its own (blank) slice via edge cuts.
    mixed = stack(art(2000, W, 1), solid(4000, W, (0, 0, 0)), art(2000, W, 2))
    arti = slice_strip(mixed, cfg)
    stats = row_stats(mixed, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    cuts = plan_cuts(mixed.shape[1], bands, stats.detail, cfg)
    assert [s.blank for s in arti.slices] == [False, True, False]
    for i, s in enumerate(arti.slices):
        assert s.blank == bool(stats.uniform[s.y0 : s.y1].all())
        assert s.forced_cut == (cuts[i].forced if i < len(cuts) else False)

    # Pure art has no bands -> all cuts forced, no blank slices.
    art_only = art(40000, W, 7)
    arti2 = slice_strip(art_only, cfg)
    assert len(arti2.slices) > 1
    assert all(not s.blank for s in arti2.slices)
    assert all(s.forced_cut for s in arti2.slices[:-1])
    assert arti2.slices[-1].forced_cut is False


def test_source_files_overlap() -> None:
    cfg = SlicerConfig()
    strip = art(20000, W, seed=5)
    files = [
        SourceFile(index=0, name="a.jpg", sha256="0" * 64, width=720, height=5000, y0=0, y1=5000),
        SourceFile(index=1, name="b.jpg", sha256="0" * 64, width=720, height=7000, y0=5000, y1=12000),
        SourceFile(index=2, name="c.jpg", sha256="0" * 64, width=720, height=8000, y0=12000, y1=20000),
    ]
    arti = slice_strip(strip, cfg, source_files=files)
    assert len(arti.slices) > 1
    for s in arti.slices:
        assert s.source_files == [f.index for f in files if f.y0 < s.y1 and f.y1 > s.y0]
    assert {i for s in arti.slices for i in s.source_files} == {0, 1, 2}


@pytest.mark.gpu
def test_gpu_performance() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    cfg = SlicerConfig()
    blocks = []
    for k in range(60):
        blocks.append(art(2420, W, seed=k))
        blocks.append(solid(80, W, (255, 255, 255)))
    strip = stack(*blocks).to("cuda")
    assert strip.shape[1] == 150_000

    slice_strip(strip, cfg)  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    slice_strip(strip, cfg)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated()
    assert elapsed < 3.0
    assert peak < 2.5 * 2**30

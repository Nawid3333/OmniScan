"""Mutation-gap tests for the slicer (card Q1): exact-value pins that surviving mutants break."""

from __future__ import annotations

import itertools

import torch

from omniscan.core.config import SlicerConfig
from omniscan.slicer.bands import find_uniform_bands, row_stats
from omniscan.slicer.cuts import plan_cuts
from tests.fixtures.strips import art, gradient, solid, stack

W = 720


def _bands(strip: torch.Tensor, min_band: int = 50, tol: int = 10, max_drift: float = 2.0) -> list:
    return find_uniform_bands(row_stats(strip, tol), min_band, tol, max_drift)


def _one_outlier_row_strip(h: int, value: int, col: int = 5) -> torch.Tensor:
    """[3, h, W] block where every row has exactly one dark pixel at `col` (given value)."""
    strip = solid(h, W, (100, 100, 100))
    strip[:, :, col] = value
    return strip


def test_tolerance_bounds_are_inclusive() -> None:
    """A pixel at med-tol / med+tol keeps the row uniform; med-tol-1 / med+tol+1 breaks it."""
    inside = _one_outlier_row_strip(200, 100 - 10)  # med 100, tol 10: pixel == lo
    assert [(b.y0, b.y1) for b in _bands(inside)] == [(0, 200)]

    inside_hi = _one_outlier_row_strip(200, 100 + 10)  # pixel == hi
    assert [(b.y0, b.y1) for b in _bands(inside_hi)] == [(0, 200)]

    outside = _one_outlier_row_strip(200, 100 - 11)  # pixel < lo: no uniform rows
    assert _bands(outside) == []

    outside_hi = _one_outlier_row_strip(200, 100 + 11)  # pixel > hi
    assert _bands(outside_hi) == []


def test_uniformity_needs_every_pixel_in_range() -> None:
    """A row whose outlier pixel is far outside tol is non-uniform even though its median matches."""
    strip = _one_outlier_row_strip(150, 0, col=5)  # outlier at 0 vs med 100
    stats = row_stats(strip, 10)
    assert bool(~stats.uniform.all())
    assert _bands(strip) == []


def test_drift_is_mean_over_channels() -> None:
    """Two solid rows differing by 6 in one channel only: mean drift 2 <= max_drift keeps one band."""
    a = solid(1, W, (100, 100, 100))
    b = solid(1, W, (106, 100, 100))
    strip = stack(*(a if i % 2 == 0 else b for i in range(100)))
    bands = _bands(stack(solid(20, W, (100, 100, 100)), strip, solid(20, W, (100, 100, 100))))
    assert [(b.y0, b.y1) for b in bands] == [(0, 140)]
    assert bands[0].color == (100, 100, 100)


def test_non_uniform_row_breaks_band_even_when_median_matches() -> None:
    """A non-uniform row splits the band even when both neighbours share its median colour."""
    strip = solid(101, W, (255, 255, 255))
    strip[0, 50, 5] = 0  # row 50 has one black pixel: non-uniform, median still 255
    bands = _bands(strip)
    assert [(b.y0, b.y1) for b in bands] == [(0, 50), (51, 101)]


def test_band_color_uses_all_rows_and_all_channels() -> None:
    """Band colour is the per-channel row median over the full band: channels are not swapped."""
    strip = stack(art(300, W, 0), solid(200, W, (200, 40, 40)), art(300, W, 1))
    bands = _bands(strip)
    assert len(bands) == 1
    assert bands[0].color == (200, 40, 40)


def test_band_color_is_lower_median_for_odd_length() -> None:
    """99-row ramp 0..98: band colour is the middle row 49 (torch.median lower median, all rows used)."""
    top = torch.arange(99, dtype=torch.float32).view(1, 99, 1).expand(3, 99, W).round().to(torch.uint8)
    strip = stack(art(300, W, 2), top, art(300, W, 3))
    bands = _bands(strip)
    assert [(b.y0, b.y1) for b in bands] == [(300, 399)]
    assert bands[0].color == (49, 49, 49)


def test_gradient_flag_threshold_is_exclusive() -> None:
    """A band whose first/last row medians differ by exactly tol is NOT a gradient (> tol, not >=)."""
    strip = stack(art(300, W, 0), gradient(51, W, (0, 0, 0), (10, 10, 10)), art(300, W, 1))
    bands = _bands(strip)
    assert [(b.y0, b.y1, b.is_gradient) for b in bands] == [(300, 351, False)]


def test_forced_cut_lands_on_least_detail_row() -> None:
    """Forced cuts pick the row with minimum detail (ties -> closest to target), driven by row_stats.detail."""
    cfg = SlicerConfig()
    strip = art(16000, W, seed=42)
    strip[:, 5100:5103, :] = 128  # flat rows: detail 0, too short (< 50) to be a band
    strip[:, 5300, :360] = 0  # crafted worst row: detail 255, median 0
    stats = row_stats(strip, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    assert bands == []
    cuts = plan_cuts(strip.shape[1], bands, stats.detail, cfg)
    assert [c.y for c in cuts] == [5102, 10690]  # 5102 = least detail (tie -> closest to t=5333)
    assert all(c.forced for c in cuts)
    heights = [b - a for a, b in itertools.pairwise([0, *(c.y for c in cuts), 16000])]
    assert heights == [5102, 5588, 5310]
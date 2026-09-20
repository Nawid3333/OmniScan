"""Mutation-gap tests for the slicer (card Q1): exact-value pins that surviving mutants break."""

from __future__ import annotations

import itertools
from typing import Any

import pytest
import torch

from omniscan.core.config import SlicerConfig
from omniscan.core.schemas import SourceFile
from omniscan.slicer import slice_by_pages
from omniscan.slicer.bands import find_uniform_bands, row_stats
from omniscan.slicer.cuts import plan_cuts
from omniscan.slicer.slice import slice_strip
from tests.fixtures.strips import art, gradient, noisy_solid, solid, stack

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


# ---------------------------------------------------------------- cost penalties (SMALL config)

SMALL_CFG: dict[str, Any] = dict(  # **-unpacked into SlicerConfig
    band_min_px=50,
    target_height=500,
    min_height=250,
    max_height=1000,
    hard_max_height=2500,
    uniform_tol=10,
    max_drift=2,
)


def _small_cuts(strip: torch.Tensor) -> list:
    cfg = SlicerConfig(**SMALL_CFG)
    stats = row_stats(strip, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    return plan_cuts(strip.shape[1], bands, stats.detail, cfg)


def test_cost_penalises_only_below_min_height() -> None:
    """A segment of exactly min_height carries no penalty, so DP prefers the 250-cut over one 1250 slice."""
    strip = stack(art(200, W, 10), solid(100, W, (5, 5, 5)), art(950, W, 11))  # centre 250
    assert [c.y for c in _small_cuts(strip)] == [250]
    assert all(not c.forced for c in _small_cuts(strip))


def test_cost_penalises_only_above_max_height() -> None:
    """A segment of exactly max_height carries no penalty, so DP prefers cuts [500] over one 1500 slice."""
    strip = stack(art(400, W, 12), solid(200, W, (6, 6, 6)), art(900, W, 13))  # centre 500
    assert [c.y for c in _small_cuts(strip)] == [500]


def test_dp_breaks_position_ties_by_earlier_cut() -> None:
    """Equal-cost paths tie-break on fewest cuts, then earliest positions: [250] beats [750]."""
    strip = stack(
        art(200, W, 14), solid(100, W, (7, 7, 7)), art(400, W, 15), solid(100, W, (8, 8, 8)), art(200, W, 16)
    )
    assert [c.y for c in _small_cuts(strip)] == [250]


def test_band_at_edge_threshold_becomes_edges() -> None:
    """A band of exactly min_height + band_min rows contributes edges (y0+25, y1-25), not a centre."""
    strip = stack(art(300, W, 17), solid(300, W, (9, 9, 9)), art(300, W, 18))  # band [300, 600)
    assert [c.y for c in _small_cuts(strip)] == [325, 575]
    assert all(not c.forced for c in _small_cuts(strip))


def test_no_cuts_when_gap_equals_hard_max() -> None:
    """A candidate-free gap of exactly hard_max_height needs no forced cut (strict > in the guard)."""
    cfg = SlicerConfig()
    strip = art(15000, W, seed=19)
    stats = row_stats(strip, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    assert bands == []
    assert plan_cuts(strip.shape[1], bands, stats.detail, cfg) == []


def test_min_height_penalty_is_small_enough_to_lose() -> None:
    """With a wide max_height the 4-point sub-minimum penalty loses to one huge slice: cuts == [100]."""
    cfg = SlicerConfig(
        band_min_px=50, target_height=500, min_height=250, max_height=10000, hard_max_height=25000
    )
    strip = stack(art(50, W, 20), solid(100, W, (11, 11, 11)), art(19850, W, 21))  # centre 100, H=20000
    stats = row_stats(strip, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    cuts = plan_cuts(strip.shape[1], bands, stats.detail, cfg)
    assert [c.y for c in cuts] == [100]
    assert not cuts[0].forced


def test_max_height_penalty_at_exact_boundary() -> None:
    """A segment of exactly max_height is penalty-free, so DP keeps one 1000px slice (cuts == [])."""
    cfg = SlicerConfig(**{**SMALL_CFG, "hard_max_height": 1000})
    strip = stack(art(50, W, 22), solid(100, W, (12, 12, 12)), art(850, W, 23))  # centre 100, H=1000
    stats = row_stats(strip, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    assert plan_cuts(strip.shape[1], bands, stats.detail, cfg) == []


def test_forced_cut_window_bounds() -> None:
    """The least-detail window is [t-256, t+256] and t is round(i*gap/(k+1)): a flat row at t+256 wins."""
    cfg = SlicerConfig()
    strip = art(16000, W, seed=44)
    strip[:, 10200:10300, :] = 200  # mid-detail rows (detail 50) below the t2 window
    strip[:, 10200:10300, 5] = 150
    strip[:, 10923, :] = 200  # detail-0 row exactly at hi = t2 + 256
    stats = row_stats(strip, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    assert bands == []
    cuts = plan_cuts(strip.shape[1], bands, stats.detail, cfg)
    assert len(cuts) == 2
    assert cuts[1].y == 10923


def test_centre_for_odd_length_band() -> None:
    """An odd-length short band's centre is (y0+y1)//2: 101 rows at [300, 401) -> centre 350."""
    strip = stack(art(300, W, 24), solid(101, W, (13, 13, 13)), art(349, W, 25))  # H=750
    assert [c.y for c in _small_cuts(strip)] == [350]


def test_forced_cut_window_lower_bound() -> None:
    """Widening the window below t-256 must not move the cut: rows below lo lose to window rows."""
    cfg = SlicerConfig()
    strip = art(16000, W, seed=45)
    strip[:, 10300:10350, :] = 200  # detail 25, just below the t2 window (lo = 10411)
    strip[:, 10300:10350, 5] = 175
    strip[:, 10500:10600, :] = 200  # detail 50, inside the window
    strip[:, 10500:10600, 5] = 150
    stats = row_stats(strip, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    assert bands == []
    cuts = plan_cuts(strip.shape[1], bands, stats.detail, cfg)
    assert len(cuts) == 2
    assert cuts[1].y == 10599


def test_max_height_penalty_magnitude_is_decisive() -> None:
    """Two over-max paths differ by 99 in cost: penalty 100 vs 400 flips which one DP picks."""
    cfg = SlicerConfig(band_min_px=50, target_height=100, min_height=50, max_height=200, hard_max_height=2000)
    strip = stack(art(950, W, 26), solid(99, W, (14, 14, 14)), art(951, W, 27))  # centre 999, H=2000
    stats = row_stats(strip, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    cuts = plan_cuts(strip.shape[1], bands, stats.detail, cfg)
    assert [c.y for c in cuts] == [999]
    assert not cuts[0].forced


# ---------------------------------------------------------------- slice_strip artifact pins


def test_artifact_records_params_and_bands() -> None:
    """slice_strip copies the effective config into params and the detected bands into the artifact."""
    cfg = SlicerConfig()
    strip = stack(art(2000, W, 1), solid(4000, W, (0, 0, 0)), art(2000, W, 2))
    arti = slice_strip(strip, cfg)
    stats = row_stats(strip, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    assert arti.params == cfg.model_dump()
    assert arti.bands == bands
    assert len(arti.bands) == 1


def test_slice_strip_uses_configured_tol() -> None:
    """Noisy rows beyond uniform_tol are not uniform: an amp-20 block is neither a band nor blank."""
    noisy = noisy_solid(3000, W, (50, 50, 50), amp=20, seed=31)
    arti = slice_strip(noisy, SlicerConfig())
    assert len(arti.slices) == 1
    assert arti.slices[0].blank is False
    assert arti.bands == []


def test_band_of_exactly_min_band_px_drives_cut() -> None:
    """A band of exactly band_min_px rows yields its centre cut (band_min_px + 1 would drop it)."""
    cfg = SlicerConfig(
        band_min_px=50, target_height=1250, min_height=250, max_height=2000, hard_max_height=2500
    )
    strip = stack(art(1000, W, 33), solid(50, W, (60, 60, 60)), art(1450, W, 34))  # band [1000, 1050), H=2500
    arti = slice_strip(strip, cfg)
    assert len(arti.bands) == 1
    assert [s.y0 for s in arti.slices] == [0, 1025]
    assert len(arti.slices) == 2
    assert not arti.slices[0].forced_cut


def test_cut_on_file_boundary_excludes_next_file() -> None:
    """A slice ending exactly at a file's y0 does not list that file (strict overlap test)."""
    cfg = SlicerConfig(
        band_min_px=50, target_height=1000, min_height=500, max_height=2000, hard_max_height=2000
    )
    strip = stack(art(950, W, 35), solid(100, W, (70, 70, 70)), art(950, W, 36))  # centre 1000
    files = [_page(0, 0, 1000), _page(1, 1000, 2000)]
    arti = slice_strip(strip, cfg, source_files=files)
    assert [c.y for c in _cuts_for(strip, cfg)] == [1000]
    assert [s.source_files for s in arti.slices] == [[0], [1]]


def test_least_detail_window_excludes_segment_start() -> None:
    """The least-detail window is clipped to (a, b): a detail-0 row AT the segment start must not win."""
    cfg = SlicerConfig(band_min_px=50, target_height=100, min_height=50, max_height=100, hard_max_height=200)
    strip = stack(art(100, W, 70), solid(40, W, (1, 1, 1)), art(61, W, 71))  # H=201 > hard 200: forced cuts
    strip[:, 0, :] = 128  # row 0: detail 0, but row 0 == a is outside the window
    strip[:, 100:140, 5] = 11  # solid block rows 100-139: detail 10 (no band: 40 < 50)
    stats = row_stats(strip, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    assert bands == []
    cuts = plan_cuts(strip.shape[1], bands, stats.detail, cfg)
    assert [c.y for c in cuts] == [100, 134]  # least-detail rows inside (0, 201), ties -> closest to t
    assert all(c.forced for c in cuts)


def test_slice_strip_tol_boundary_row_splits_band() -> None:
    """A row whose outlier sits at med+(tol+1) is non-uniform: slice_strip must split the band there."""
    cfg = SlicerConfig(**SMALL_CFG)
    outlier = _one_outlier_row_strip(100, 89)  # |89 - 100| = 11 > tol 10
    strip = stack(solid(900, W, (100, 100, 100)), outlier, art(500, W, 61))  # H=1500
    arti = slice_strip(strip, cfg)
    assert [s.y0 for s in arti.slices] == [0, 25, 875]  # band (0, 900) -> edges 25, 875


def _cuts_for(strip: torch.Tensor, cfg: SlicerConfig) -> list:
    stats = row_stats(strip, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    return plan_cuts(strip.shape[1], bands, stats.detail, cfg)


def _page(index: int, y0: int, y1: int) -> SourceFile:
    return SourceFile(
        index=index, name=f"p{index:02d}.jpg", sha256="0" * 64, width=W, height=y1 - y0, y0=y0, y1=y1
    )


def test_first_page_negative_y0_raises() -> None:
    """A page starting above the strip top (y0 < 0) is rejected, not just y0 > 0."""
    files = [_page(0, -10, 1000), _page(1, 1000, 3000)]
    with pytest.raises(ValueError, match="expected 0"):
        slice_by_pages(files, W, 3000)


def test_last_page_overshooting_strip_raises() -> None:
    """A page ending past the strip bottom (y1 > strip_height) is rejected."""
    files = [_page(0, 0, 1000), _page(1, 1000, 3100)]
    with pytest.raises(ValueError, match="strip_height=3000"):
        slice_by_pages(files, W, 3000)

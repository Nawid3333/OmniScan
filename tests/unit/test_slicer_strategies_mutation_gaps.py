"""Gap tests from the card Q6 mutation review: each one fails under a mutant that survived the
existing tests (`tests/mutants/slicer_strategies/strategies.py` lists which mutant each kills)."""

import torch

from omniscan.core.config import SlicerConfig
from omniscan.core.schemas import SourceFile
from omniscan.slicer.strategies import (
    gutter_cut_rows,
    slice_fixed,
    slice_page,
    slice_simple_gutter,
    slice_with_strategy,
)
from tests.fixtures.strips import art, solid

W = 720
FIXED_CFG = SlicerConfig(min_height=1500, target_height=3000, max_height=6000)


def _noise_with_gutters(h: int, w: int, seed: int, gutters: list[tuple[int, int]]) -> torch.Tensor:
    strip = art(h, w, seed)
    for y0, y1 in gutters:
        strip[:, y0:y1, :] = 255
    return strip


def _variance_rows(h: int, w: int, hi: int) -> torch.Tensor:
    """[3, h, w] rows alternating 100/`hi` per column (all channels equal); row variance = (hi-100)^2/4."""
    row = (100 + (hi - 100) * (torch.arange(w) % 2)).to(torch.uint8)
    return row.view(1, 1, w).expand(3, h, w).contiguous()


def _page(index: int, y0: int, y1: int) -> SourceFile:
    return SourceFile(
        index=index, name=f"p{index:02d}.jpg", sha256="0" * 64, width=W, height=y1 - y0, y0=y0, y1=y1
    )


# mutant "drop empty-strip page guard": a 0-row strip must take the page path (no files -> empty
# artifact), not fall back to smart
def test_page_empty_strip_with_no_files_has_no_fallback() -> None:
    artifact = slice_page(art(0, W, 0), FIXED_CFG)
    assert artifact.slices == []
    assert artifact.params["strategy"] == "page"
    assert "fallback" not in artifact.params


# mutant "merge threshold < -> <=": a tail of exactly min_height rows is a slice of its own
def test_fixed_tail_exactly_min_height_is_not_merged() -> None:
    artifact = slice_fixed(art(7500, W, 0), FIXED_CFG)
    assert [(s.y0, s.y1, s.forced_cut) for s in artifact.slices] == [
        (0, 3000, True),
        (3000, 6000, True),
        (6000, 7500, False),
    ]


# mutant "variance < -> <=": rows with variance exactly at the threshold are not gutters
def test_gutter_variance_threshold_is_strict() -> None:
    # rows alternating 100/104 have row variance exactly 4.0 == cfg.gutter_variance
    strip = _variance_rows(10000, 100, 104)
    assert gutter_cut_rows(strip, SlicerConfig(gutter_variance=4.0)) == []


# mutant "run centre floor -> ceil": an odd-length run cuts at its floored centre
def test_gutter_run_centre_is_floored() -> None:
    strip = _noise_with_gutters(10000, 100, 8, [(2000, 2009)])  # centre (2000 + 2009) // 2 = 2004
    assert gutter_cut_rows(strip, FIXED_CFG) == [2004]


# mutants "run length >= -> >" and "min_rows threshold - 1": a 7-row run is below the 8-row
# threshold, an 8-row run is exactly at it
def test_gutter_min_rows_boundary() -> None:
    strip = _noise_with_gutters(10000, 100, 9, [(3000, 3007), (5000, 5008)])
    assert gutter_cut_rows(strip, FIXED_CFG) == [5004]


# mutant "gutter loop > -> >=": a strip of exactly max_height rows is one slice
def test_simple_gutter_strip_exactly_max_height_is_one_slice() -> None:
    artifact = slice_simple_gutter(art(6000, 100, 11), FIXED_CFG)
    assert [(s.y0, s.y1, s.forced_cut) for s in artifact.slices] == [(0, 6000, False)]


# mutant "window top <= -> <": a gutter centred exactly on the window's top bound is a candidate
def test_simple_gutter_cut_exactly_at_window_max_is_used() -> None:
    strip = _noise_with_gutters(10000, 100, 12, [(5980, 6020)])  # centre 6000 == first window hi
    artifact = slice_simple_gutter(strip, FIXED_CFG)
    assert [(s.y0, s.y1, s.forced_cut) for s in artifact.slices] == [
        (0, 6000, False),
        (6000, 10000, False),
    ]


# mutant "measure from window start": the candidate closest to target_height wins, not to pos
def test_simple_gutter_picks_the_cut_closest_to_target() -> None:
    # gutters at 1500 and 4000: 4000 is closest to target 3000, 1500 would be closest to pos 0
    strip = _noise_with_gutters(10000, 100, 13, [(1480, 1520), (3980, 4020)])
    artifact = slice_simple_gutter(strip, FIXED_CFG)
    assert [(s.y0, s.y1, s.forced_cut) for s in artifact.slices] == [
        (0, 4000, False),
        (4000, 10000, False),
    ]


# mutant "blank all -> any": a single uneven row makes the whole slice non-blank
def test_slice_with_one_uneven_row_is_not_blank() -> None:
    strip = solid(6000, W, (255, 255, 255))
    strip[:, 1000, ::2] = 0  # alternating 0/255: the only non-uniform row of the first slice
    artifact = slice_fixed(strip, FIXED_CFG)
    assert [(s.y0, s.y1, s.blank) for s in artifact.slices] == [(0, 3000, False), (3000, 6000, True)]


# mutant "file start < -> <=": a file starting exactly at a slice's y1 belongs to later slices only
def test_source_file_starting_exactly_at_slice_start_is_excluded() -> None:
    files = [_page(0, 0, 2000), _page(1, 3000, 5000)]
    artifact = slice_fixed(art(5000, W, 0), FIXED_CFG, files)
    assert [s.source_files for s in artifact.slices] == [[0], [1]]


# mutant "strip dims swapped": _finalize must report the strip's real width and height
def test_finalize_records_the_strip_dimensions() -> None:
    strip = art(2000, W, 0)
    for name in ("fixed", "simple_gutter"):
        artifact = slice_with_strategy(strip, FIXED_CFG, strategy=name)
        assert (artifact.strip_width, artifact.strip_height) == (W, 2000)


# mutant "file end > -> >=": a file ending exactly at a slice's y0 belongs to later slices only
def test_source_file_ending_exactly_at_slice_start_is_excluded() -> None:
    cfg = SlicerConfig(min_height=1500, target_height=2000, max_height=6000)
    files = [_page(0, 0, 2000), _page(1, 2000, 4000)]
    artifact = slice_fixed(art(4000, W, 0), cfg, files)
    assert [s.source_files for s in artifact.slices] == [[0], [1]]


# mutant "uniform_tol + 1": blankness honours the configured tolerance exactly
def test_blank_flag_uses_the_configured_tolerance() -> None:
    strip = solid(6000, W, (100, 100, 100))
    strip[:, 1000, 0] = 111  # 11 above the row median: outside tol=10, inside tol=11
    artifact = slice_fixed(strip, FIXED_CFG)
    assert [(s.y0, s.y1, s.blank) for s in artifact.slices] == [(0, 3000, False), (3000, 6000, True)]

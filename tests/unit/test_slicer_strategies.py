"""Unit tests for the slicer strategies and the stage dispatcher wiring (card S2)."""

from itertools import pairwise
from pathlib import Path

import pytest
import torch

from omniscan.core.config import Config, GpuConfig, PathsConfig, SlicerConfig
from omniscan.core.schemas import SlicesArtifact, SourceFile
from omniscan.core.stage import make_context, run_chapter
from omniscan.ingest.stage import IngestStage
from omniscan.slicer import slice_strip
from omniscan.slicer.stage import SliceStage
from omniscan.slicer.strategies import (
    STRATEGIES,
    gutter_cut_rows,
    slice_fixed,
    slice_simple_gutter,
    slice_with_strategy,
)
from tests.fixtures import images
from tests.fixtures.strip_layouts import random_strip
from tests.fixtures.strips import art, solid, stack

W = 720
SEEDS = range(24)
FIXED_CFG = SlicerConfig(min_height=1500, target_height=3000, max_height=6000)


def _page(index: int, y0: int, y1: int) -> SourceFile:
    return SourceFile(
        index=index, name=f"p{index:02d}.jpg", sha256="0" * 64, width=W, height=y1 - y0, y0=y0, y1=y1
    )


# ---------------------------------------------------------------- dispatcher


def test_dispatcher_follows_cfg_and_explicit_strategy() -> None:
    strip, _ = random_strip(0)
    assert slice_with_strategy(strip, SlicerConfig(strategy="fixed")).params["strategy"] == "fixed"
    assert (
        slice_with_strategy(strip, SlicerConfig(strategy="smart"), strategy="fixed").params["strategy"] == "fixed"
    )
    assert (
        slice_with_strategy(strip, SlicerConfig(strategy="fixed"), strategy="smart").params["strategy"] == "smart"
    )


def test_dispatcher_unknown_name_message() -> None:
    strip, _ = random_strip(0)
    with pytest.raises(ValueError) as excinfo:
        slice_with_strategy(strip, SlicerConfig(), strategy="nope")
    assert str(excinfo.value) == "unknown slicer strategy 'nope'; choose one of smart, page, fixed, simple_gutter"


def test_smart_matches_slice_strip_apart_from_params() -> None:
    cfg = SlicerConfig()
    files = [_page(0, 0, 2000), _page(1, 2000, 4000)]
    strip = stack(art(2000, W, 1), art(2000, W, 2))
    got = slice_with_strategy(strip, cfg, files, strategy="smart")
    expected = slice_strip(strip, cfg, files)
    assert got.slices == expected.slices
    assert got.bands == expected.bands
    assert (got.strip_width, got.strip_height) == (expected.strip_width, expected.strip_height)
    assert got.params == {**cfg.model_dump(), "strategy": "smart"}


@pytest.mark.parametrize("seed", SEEDS)
def test_every_strategy_tiles_the_strip(seed: int) -> None:
    strip, _ = random_strip(seed)
    height = strip.shape[1]
    for name in STRATEGIES:
        artifact = slice_with_strategy(strip, SlicerConfig(), strategy=name)
        assert artifact.params["strategy"] == name
        assert artifact.slices  # random strips are never empty
        assert artifact.slices[0].y0 == 0
        assert artifact.slices[-1].y1 == height
        for i, s in enumerate(artifact.slices):
            assert s.index == i
            assert s.y0 < s.y1
        for a, b in pairwise(artifact.slices):
            assert a.y1 == b.y0


# ---------------------------------------------------------------- fixed


def test_fixed_examples() -> None:
    def slices_of(h: int) -> list[tuple[int, int, bool]]:
        artifact = slice_fixed(art(h, W, 0), FIXED_CFG)
        return [(s.y0, s.y1, s.forced_cut) for s in artifact.slices]

    # H=10000: cuts 3000/6000/9000, the 1000-row tail merges; the merged slice keeps its forced flag.
    assert slices_of(10000) == [(0, 3000, True), (3000, 6000, True), (6000, 10000, True)]
    assert slices_of(9000) == [(0, 3000, True), (3000, 6000, True), (6000, 9000, False)]
    assert slices_of(2000) == [(0, 2000, False)]
    assert slices_of(0) == []


def test_fixed_merges_short_tail_into_previous() -> None:
    # H=6500: cuts at 3000/6000, the 500-row tail merges; the merged slice keeps its own forced flag.
    artifact = slice_fixed(art(6500, W, 0), FIXED_CFG)
    assert [(s.y0, s.y1, s.forced_cut) for s in artifact.slices] == [(0, 3000, True), (3000, 6500, True)]
    # H=4000: the tail after the 3000 cut is 1000 < 1500 -> one slice whose own cut was forced.
    artifact = slice_fixed(art(4000, W, 0), FIXED_CFG)
    assert [(s.y0, s.y1, s.forced_cut) for s in artifact.slices] == [(0, 4000, True)]


def test_fixed_blank_flags_are_computed() -> None:
    artifact = slice_fixed(solid(5000, W, (255, 255, 255)), FIXED_CFG)
    assert [(s.y0, s.y1, s.blank) for s in artifact.slices] == [(0, 3000, True), (3000, 5000, True)]


def test_fixed_source_files_mapping() -> None:
    files = [_page(0, 0, 2000), _page(1, 2000, 5000)]
    artifact = slice_fixed(art(5000, W, 0), FIXED_CFG, files)
    assert [s.source_files for s in artifact.slices] == [[0, 1], [1]]


# ---------------------------------------------------------------- simple_gutter


def _noise_with_gutters(h: int, w: int, seed: int, gutters: list[tuple[int, int]]) -> torch.Tensor:
    strip = art(h, w, seed)
    for y0, y1 in gutters:
        strip[:, y0:y1, :] = 255
    return strip


def test_simple_gutter_golden() -> None:
    strip = _noise_with_gutters(10000, 100, 0, [(2900, 2940), (5000, 5100), (8000, 8050)])
    assert gutter_cut_rows(strip, FIXED_CFG) == [2920, 5050, 8025]
    artifact = slice_simple_gutter(strip, FIXED_CFG)
    assert [(s.y0, s.y1, s.forced_cut) for s in artifact.slices] == [
        (0, 2920, False),
        (2920, 5050, False),
        (5050, 10000, False),
    ]


def test_simple_gutter_no_gutter_forces_target_cuts() -> None:
    strip = art(10000, 100, 1)
    assert gutter_cut_rows(strip, FIXED_CFG) == []
    artifact = slice_simple_gutter(strip, FIXED_CFG)
    assert [(s.y0, s.y1, s.forced_cut) for s in artifact.slices] == [
        (0, 3000, True),
        (3000, 6000, True),
        (6000, 10000, False),
    ]


def test_simple_gutter_ignores_gutters_outside_window() -> None:
    strip = _noise_with_gutters(7000, 100, 2, [(450, 560), (6400, 6500)])
    assert gutter_cut_rows(strip, FIXED_CFG) == [505, 6450]
    artifact = slice_simple_gutter(strip, FIXED_CFG)
    assert [(s.y0, s.y1, s.forced_cut) for s in artifact.slices] == [(0, 3000, True), (3000, 7000, False)]


def test_simple_gutter_tie_prefers_smaller_cut_row() -> None:
    strip = _noise_with_gutters(10000, 100, 3, [(2880, 2920), (3080, 3120)])
    assert gutter_cut_rows(strip, FIXED_CFG) == [2900, 3100]
    artifact = slice_simple_gutter(strip, FIXED_CFG)
    assert [(s.y0, s.y1, s.forced_cut) for s in artifact.slices] == [
        (0, 2900, False),
        (2900, 5900, True),
        (5900, 10000, False),
    ]


def test_simple_gutter_ignores_short_runs() -> None:
    strip = _noise_with_gutters(10000, 100, 4, [(2000, 2004)])  # 4 rows < gutter_min_rows 8
    assert gutter_cut_rows(strip, FIXED_CFG) == []
    artifact = slice_simple_gutter(strip, FIXED_CFG)
    assert [(s.y0, s.y1, s.forced_cut) for s in artifact.slices] == [
        (0, 3000, True),
        (3000, 6000, True),
        (6000, 10000, False),
    ]


def _variance_rows(h: int, w: int, hi: int) -> torch.Tensor:
    """[3, h, w] rows alternating 100/`hi` per column (all channels equal); row variance = (hi-100)^2/4."""
    row = (100 + (hi - 100) * (torch.arange(w) % 2)).to(torch.uint8)
    return row.view(1, 1, w).expand(3, h, w).contiguous()


def test_simple_gutter_variance_threshold_is_respected() -> None:
    # gutter rows alternate 100/104 (row variance 4.0 < 5); art rows alternate 100/106 (variance 9.0)
    strip = _variance_rows(10000, 100, 106)
    strip[:, 4000:4100, :] = _variance_rows(100, 100, 104)
    assert gutter_cut_rows(strip, FIXED_CFG) == [4050]
    artifact = slice_simple_gutter(strip, FIXED_CFG)
    assert [(s.y0, s.y1, s.forced_cut) for s in artifact.slices] == [(0, 4050, False), (4050, 10000, False)]


def test_simple_gutter_short_strip_is_one_slice() -> None:
    strip = _noise_with_gutters(5000, 100, 5, [(2000, 2100)])
    artifact = slice_simple_gutter(strip, FIXED_CFG)
    assert [(s.y0, s.y1) for s in artifact.slices] == [(0, 5000)]


def test_simple_gutter_handles_60k_rows() -> None:
    strip = art(60000, 100, 6)
    artifact = slice_simple_gutter(strip, FIXED_CFG)
    assert len(artifact.slices) == 19  # forced cuts at 3000..54000, then the 6000-row remainder
    assert [s.forced_cut for s in artifact.slices] == [True] * 18 + [False]


# ---------------------------------------------------------------- page


def test_page_cuts_at_file_boundaries() -> None:
    strip = stack(art(1000, W, 0), solid(2000, W, (255, 255, 255)), art(1500, W, 1))
    files = [_page(0, 0, 1000), _page(1, 1000, 3000), _page(2, 3000, 4500)]
    artifact = slice_with_strategy(strip, SlicerConfig(), files, strategy="page")
    assert [(s.y0, s.y1, s.blank, s.source_files) for s in artifact.slices] == [
        (0, 1000, False, [0]),
        (1000, 3000, True, [1]),
        (3000, 4500, False, [2]),
    ]
    assert artifact.bands == []
    assert artifact.params["strategy"] == "page"


def test_page_falls_back_to_smart_without_files() -> None:
    strip = art(4500, W, 2)
    for files in (None, []):
        artifact = slice_with_strategy(strip, SlicerConfig(), files, strategy="page")
        assert artifact.params["fallback"] == "smart"
        assert artifact.params["strategy"] == "page"
        assert len(artifact.slices) >= 1


def test_page_propagates_tiling_errors() -> None:
    strip = art(4500, W, 3)
    files = [_page(0, 0, 1000), _page(1, 1010, 4500)]  # 10px gap
    with pytest.raises(ValueError, match="gap of 10px"):
        slice_with_strategy(strip, SlicerConfig(), files, strategy="page")


# ---------------------------------------------------------------- stage wiring


def _cpu_cfg(tmp_path: Path) -> Config:
    return Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
    )


def _raw_chapter(cfg: Config) -> None:
    raw = cfg.paths.library_root / "S" / "Chapter 1"
    raw.mkdir(parents=True, exist_ok=True)
    for i in (1, 2, 3):
        images.plain_jpeg(raw / f"{i:03d}.jpg", size=(400, 300), color=(200, 60, 60))


def test_stage_uses_configured_strategy(tmp_path: Path) -> None:
    cfg = _cpu_cfg(tmp_path)
    _raw_chapter(cfg)
    cfg_page = cfg.model_copy(update={"slicer": SlicerConfig(strategy="page")})
    ctx = make_context(cfg_page, "S", "Chapter 1")
    assert run_chapter([IngestStage(), SliceStage()], ctx)[1].status == "done"
    artifact = SlicesArtifact.load(ctx.paths.artifact("slices.json"))
    assert artifact.params["strategy"] == "page"
    assert len(artifact.slices) == 3


def test_series_toml_strategy_reaches_stage(tmp_path: Path) -> None:
    cfg = _cpu_cfg(tmp_path)
    _raw_chapter(cfg)
    (cfg.paths.library_root / "S" / "series.toml").write_text('[slicer]\nstrategy = "fixed"\n', encoding="utf-8")
    ctx = make_context(cfg, "S", "Chapter 1")
    assert ctx.cfg.slicer.strategy == "fixed"
    assert run_chapter([IngestStage(), SliceStage()], ctx)[1].status == "done"
    artifact = SlicesArtifact.load(ctx.paths.artifact("slices.json"))
    assert artifact.params["strategy"] == "fixed"


def test_stage_version_is_3_and_config_hash_tracks_strategy() -> None:
    assert SliceStage.version == 3
    plain = SliceStage().config_subset(Config(slicer=SlicerConfig()))
    paged = SliceStage().config_subset(Config(slicer=SlicerConfig(strategy="page")))
    assert plain != paged
"""Tests for the ingest/slice stage wiring, strip building, codec selection and CLI (card B24)."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, GpuConfig, PathsConfig, SlicerConfig
from omniscan.core.schemas import IngestArtifact, SlicesArtifact, SourceFile
from omniscan.core.stage import make_context, run_chapter, run_stage
from omniscan.gpu.codec.base import CodecUnavailableError
from omniscan.gpu.codec.select import get_codec
from omniscan.gpu.codec.turbo import TurboCodec
from omniscan.ingest import ingest_chapter
from omniscan.ingest.stage import IngestStage
from omniscan.ingest.strip import build_strip, jpeg_paths
from omniscan.slicer.stage import SliceStage
from tests.fixtures import images

SERIES = "S"
CHAPTER = "Chapter 1"
COLORS = [(200, 60, 60), (60, 200, 120), (80, 100, 220)]
runner = CliRunner()


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """CPU-only config with all paths under tmp_path."""
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


def raw_dir(cfg: Config, series: str = SERIES, chapter: str = CHAPTER) -> Path:
    """Create and return the raw chapter dir."""
    path = cfg.paths.library_root / series / chapter
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jpegs(raw: Path, sizes: list[tuple[int, int]], colors: list[tuple[int, int, int]]) -> None:
    for i, (size, color) in enumerate(zip(sizes, colors, strict=True)):
        images.plain_jpeg(raw / f"{i + 1:03d}.jpg", size=size, color=color)


def art_page(path: Path, width: int, white: int, noise: int, rng: np.random.Generator) -> None:
    """Solid-white bands around a block of random noise, so the slicer finds real uniform cut zones."""
    rows = [
        np.full((white, width, 3), 255, dtype=np.uint8),
        rng.integers(0, 256, size=(noise, width, 3), dtype=np.uint8),
        np.full((white, width, 3), 255, dtype=np.uint8),
    ]
    Image.fromarray(np.concatenate(rows)).save(path, format="JPEG", quality=95)


def assert_segment_color(
    strip: torch.Tensor, y0: int, y1: int, color: tuple[int, int, int], tol: int = 3
) -> None:
    mean = strip[:, y0:y1, :].float().mean(dim=(1, 2)).tolist()
    assert all(abs(m - c) <= tol for m, c in zip(mean, color, strict=True))


# ---------------------------------------------------------------- strip building / codec


def test_jpeg_paths_raw_and_converted(tmp_path: Path) -> None:
    ingest = IngestArtifact(
        series=SERIES,
        chapter=CHAPTER,
        strip_width=400,
        strip_height=600,
        files=[
            SourceFile(index=0, name="page1.jpg", sha256="a" * 64, width=400, height=300, y0=0, y1=300),
            SourceFile(
                index=1,
                name="page2.png",
                sha256="b" * 64,
                width=400,
                height=300,
                y0=300,
                y1=600,
                converted_from=".png",
            ),
        ],
    )
    raw_dir, cache_dir = tmp_path / "raw", tmp_path / "converted"
    assert jpeg_paths(ingest, raw_dir, cache_dir) == [raw_dir / "page1.jpg", cache_dir / "0001_page2.jpg"]


def test_build_strip_same_width(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_jpegs(raw, [(400, 300), (400, 200), (400, 150)], COLORS)
    ingest = ingest_chapter(raw, SERIES, CHAPTER, tmp_path / "cache").artifact
    strip = build_strip(ingest, jpeg_paths(ingest, raw, tmp_path / "cache"), TurboCodec(device="cpu"))
    assert strip.shape == (3, 650, 400)
    assert strip.dtype == torch.uint8
    for file, color in zip(ingest.files, COLORS, strict=True):
        assert_segment_color(strip, file.y0, file.y1, color)


def test_build_strip_resizes_off_width(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_jpegs(raw, [(400, 200), (400, 150), (300, 120)], COLORS)
    ingest = ingest_chapter(raw, SERIES, CHAPTER, tmp_path / "cache").artifact
    assert ingest.strip_width == 400
    odd = ingest.files[2]
    assert odd.width == 300
    assert odd.y1 - odd.y0 == 160  # round(120 * 400 / 300)
    strip = build_strip(ingest, jpeg_paths(ingest, raw, tmp_path / "cache"), TurboCodec(device="cpu"))
    assert strip.shape == (3, ingest.strip_height, 400)
    assert strip[:, odd.y0 : odd.y1, :].shape == (3, odd.y1 - odd.y0, 400)
    assert_segment_color(strip, odd.y0, odd.y1, COLORS[2])


def test_build_strip_rejects_wrong_path_count(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_jpegs(raw, [(400, 100), (400, 100)], COLORS[:2])
    ingest = ingest_chapter(raw, SERIES, CHAPTER, tmp_path / "cache").artifact
    with pytest.raises(ValueError, match="paths"):
        build_strip(ingest, jpeg_paths(ingest, raw, tmp_path / "cache")[:1], TurboCodec(device="cpu"))


def test_get_codec(tmp_path: Path) -> None:
    def cfg_with(codec: Literal["auto", "rocjpeg", "hybrid", "turbo"]) -> Config:
        return Config(
            gpu=GpuConfig(device="cpu", codec=codec),
            paths=PathsConfig(
                library_root=tmp_path / "library",
                work_root=tmp_path / "work",
                output_root=tmp_path / "output",
                promo_examples=tmp_path / "promo",
                models_dir=tmp_path / "models",
            ),
        )

    for codec in ("auto", "turbo"):
        selected = get_codec(cfg_with(codec))
        assert isinstance(selected, TurboCodec)
        assert selected.device.type == "cpu"
    for codec in ("rocjpeg", "hybrid"):
        with pytest.raises(CodecUnavailableError, match="not implemented yet"):
            get_codec(cfg_with(codec))


# ---------------------------------------------------------------- stages


def test_ingest_stage_is_resumable(cfg: Config) -> None:
    raw = raw_dir(cfg)
    write_jpegs(raw, [(400, 300), (400, 200), (400, 150)], COLORS)

    ctx = make_context(cfg, SERIES, CHAPTER)
    outcome = run_stage(IngestStage(), ctx)
    assert outcome.status == "done"
    artifact = IngestArtifact.load(ctx.paths.artifact("ingest.json"))
    assert len(artifact.files) == 3
    assert artifact.strip_height == 650

    assert run_stage(IngestStage(), make_context(cfg, SERIES, CHAPTER)).status == "skipped"

    images.plain_jpeg(raw / "001.jpg", size=(400, 300), color=(10, 20, 30))
    assert run_stage(IngestStage(), make_context(cfg, SERIES, CHAPTER)).status == "done"
    assert run_stage(IngestStage(), make_context(cfg, SERIES, CHAPTER), force=True).status == "done"


def test_png_page_converted_then_decoded_by_slice(cfg: Config) -> None:
    raw = raw_dir(cfg)
    images.plain_jpeg(raw / "001.jpg", size=(400, 300))
    images.png_with_alpha(raw / "002.png", size=(400, 200))

    ctx = make_context(cfg, SERIES, CHAPTER)
    outcome = run_stage(IngestStage(), ctx)
    assert outcome.status == "done"
    assert outcome.metrics["converted"] == 1.0
    converted = ctx.paths.work_dir / "converted" / f"{1:04d}_{Path('002.png').stem}.jpg"
    assert converted.is_file()

    # the slice stage decodes the converted cache file while building the strip
    slice_outcome = run_stage(SliceStage(), make_context(cfg, SERIES, CHAPTER))
    assert slice_outcome.status == "done"
    assert ctx.paths.artifact("slices.json").is_file()


def test_slice_stage_tiles_strip(cfg: Config) -> None:
    raw = raw_dir(cfg)
    rng = np.random.default_rng(0)
    art_page(raw / "001.jpg", 400, 1700, 2000, rng)
    art_page(raw / "002.jpg", 400, 1700, 2000, rng)

    ctx = make_context(cfg, SERIES, CHAPTER)
    outcomes = run_chapter([IngestStage(), SliceStage()], ctx)
    assert [o.status for o in outcomes] == ["done", "done"]

    ingest = IngestArtifact.load(ctx.paths.artifact("ingest.json"))
    slices = SlicesArtifact.load(ctx.paths.artifact("slices.json"))
    assert slices.strip_height == ingest.strip_height
    assert slices.slices[0].y0 == 0
    assert slices.slices[-1].y1 == slices.strip_height
    for a, b in pairwise(slices.slices):
        assert a.y1 == b.y0
    indices = {i for s in slices.slices for i in s.source_files}
    assert indices <= set(range(len(ingest.files)))
    assert all(isinstance(s.blank, bool) and isinstance(s.forced_cut, bool) for s in slices.slices)
    assert len(slices.slices) > 1  # the white bands between art blocks produced real cuts


def test_slicer_config_change_invalidates_only_slice(cfg: Config) -> None:
    raw = raw_dir(cfg)
    images.plain_jpeg(raw / "001.jpg", size=(400, 300))

    stages = [IngestStage(), SliceStage()]
    assert [o.status for o in run_chapter(stages, make_context(cfg, SERIES, CHAPTER))] == ["done", "done"]
    assert [o.status for o in run_chapter(stages, make_context(cfg, SERIES, CHAPTER))] == [
        "skipped",
        "skipped",
    ]

    cfg2 = cfg.model_copy(update={"slicer": SlicerConfig(target_height=2600)})
    outcomes = run_chapter(stages, make_context(cfg2, SERIES, CHAPTER))
    assert [o.status for o in outcomes] == ["skipped", "done"]


def test_slice_stage_without_ingest_fails(cfg: Config) -> None:
    raw_dir(cfg)
    outcome = run_stage(SliceStage(), make_context(cfg, SERIES, CHAPTER))
    assert outcome.status == "failed"
    assert outcome.error is not None
    assert "ingest.json" in outcome.error


# ---------------------------------------------------------------- CLI


def test_cli_ingest_and_slice(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    for chapter in ("Chapter 1", "Chapter 2"):
        write_jpegs(raw_dir(cfg, SERIES, chapter), [(400, 300)] * 3, COLORS)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["ingest", SERIES])
    assert result.exit_code == 0
    for chapter in ("Chapter 1", "Chapter 2"):
        assert f"{SERIES}/{chapter} ingest: done" in result.output
        assert (cfg.paths.work_root / SERIES / chapter / "ingest.json").is_file()

    result = runner.invoke(app, ["slice", SERIES, "--chapter", "Chapter 1"])
    assert result.exit_code == 0
    work = cfg.paths.work_root / SERIES / "Chapter 1"
    assert (work / "ingest.json").is_file()
    assert (work / "slices.json").is_file()

    result = runner.invoke(app, ["slice", SERIES, "--chapter", "Chapter 1"])
    assert result.exit_code == 0
    assert "skipped" in result.output

    result = runner.invoke(app, ["slice", "NoSuchSeries"])
    assert result.exit_code == 2
    assert "no chapters found" in result.output

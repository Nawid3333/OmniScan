"""Tests for the `omniscan filter` CLI group (card F2a): add, run, restore, force."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.core.schemas import IngestArtifact, SlicesArtifact
from tests.fixtures import images

SERIES = "S"
CHAPTER = "Chapter 1"
COLORS = [(200, 60, 60), (60, 200, 120), (80, 100, 220)]
runner = CliRunner()


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
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


def write_library(cfg: Config) -> None:
    """5 raw pages: 001/003/005 solid colours, 002/004 near-copies of a global promo example."""
    raw = cfg.paths.library_root / SERIES / CHAPTER
    promo = cfg.paths.promo_examples / "global"
    promo.mkdir(parents=True)
    raw.mkdir(parents=True)
    example = images.gradient_jpeg(promo / "end_card.jpg", invert=True)
    for i, color in enumerate(COLORS):
        images.plain_jpeg(raw / f"{2 * i + 1:03d}.jpg", color=color)
    with Image.open(example) as img:
        resized = img.convert("RGB").resize((384, 290))
    rng = np.random.default_rng(0)
    for i in (2, 4):
        pixels = np.asarray(resized, dtype=np.int16) + rng.integers(-3, 4, (290, 384, 3), dtype=np.int16)
        Image.fromarray(pixels.clip(0, 255).astype(np.uint8)).save(raw / f"{i:03d}.jpg", format="JPEG")


# ---------------------------------------------------------------- filter add


def test_filter_add_copies_into_series_and_global(tmp_path: Path, cfg: Config, monkeypatch) -> None:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    source = images.gradient_jpeg(tmp_path / "ad.jpg")

    result = runner.invoke(app, ["filter", "add", SERIES, str(source)])
    assert result.exit_code == 0
    dest = cfg.paths.promo_examples / SERIES / "ad.jpg"
    assert dest.is_file()
    assert dest.read_bytes() == source.read_bytes()
    assert f"filter: added example {dest}" in result.output

    result = runner.invoke(app, ["filter", "add", SERIES, str(source), "--global", "--name", "card.jpg"])
    assert result.exit_code == 0
    assert (cfg.paths.promo_examples / "global" / "card.jpg").is_file()
    assert not (cfg.paths.promo_examples / "global" / "ad.jpg").exists()


def test_filter_add_refuses_overwrites_and_bad_names(tmp_path: Path, cfg: Config, monkeypatch) -> None:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    source = images.gradient_jpeg(tmp_path / "ad.jpg")

    result = runner.invoke(app, ["filter", "add", SERIES, str(source)])
    assert result.exit_code == 0
    result = runner.invoke(app, ["filter", "add", SERIES, str(source)])  # same destination again
    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output

    result = runner.invoke(app, ["filter", "add", SERIES, str(source), "--name", "sub/dir.jpg"])
    assert result.exit_code == 2
    assert "plain file name" in result.output


def test_filter_add_rejects_non_image_and_non_jpeg_png(tmp_path: Path, cfg: Config, monkeypatch) -> None:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    fake = tmp_path / "fake.jpg"
    fake.write_text("not an image", encoding="utf-8")
    result = runner.invoke(app, ["filter", "add", SERIES, str(fake)])
    assert result.exit_code == 2
    assert "not a readable image" in result.output

    webp = images.webp_image(tmp_path / "ad.webp")
    result = runner.invoke(app, ["filter", "add", SERIES, str(webp)])
    assert result.exit_code == 2
    assert "not a JPEG/PNG example" in result.output


# ---------------------------------------------------------------- filter run


def test_filter_run_reports_counts_and_second_run_is_a_no_op(
    tmp_path: Path, cfg: Config, monkeypatch
) -> None:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    write_library(cfg)

    result = runner.invoke(app, ["filter", "run", SERIES])
    assert result.exit_code == 0
    assert f"{SERIES}/{CHAPTER} ingest: done" in result.output
    assert f"{SERIES}/{CHAPTER} slice: done" in result.output
    assert f"{SERIES}/{CHAPTER}: filtered files: 2, filtered slices: 0" in result.output
    assert "filter: 2 file(s), 0 slice(s) filtered in 1 chapter(s)" in result.output

    result = runner.invoke(app, ["filter", "run", SERIES, "--chapter", CHAPTER])
    assert result.exit_code == 0
    assert "skipped" in result.output  # second run is a no-op
    assert f"{SERIES}/{CHAPTER}: filtered files: 2, filtered slices: 0" in result.output


def test_filter_run_json_shape(tmp_path: Path, cfg: Config, monkeypatch) -> None:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    write_library(cfg)

    result = runner.invoke(app, ["filter", "run", SERIES, "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["series"] == SERIES
    assert payload["totals"] == {"files": 2, "slices": 0, "chapters": 1}
    assert payload["chapters"] == [{"chapter": CHAPTER, "files": ["002.jpg", "004.jpg"], "slices": []}]


def test_filter_run_unknown_series_exits_2(tmp_path: Path, cfg: Config, monkeypatch) -> None:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    result = runner.invoke(app, ["filter", "run", "NoSuchSeries"])
    assert result.exit_code == 2
    assert "no chapters found" in result.output


# ---------------------------------------------------------------- filter restore / force


def test_filter_restore_then_run_keeps_the_file(tmp_path: Path, cfg: Config, monkeypatch) -> None:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    write_library(cfg)

    result = runner.invoke(app, ["filter", "run", SERIES])
    assert result.exit_code == 0

    result = runner.invoke(app, ["filter", "restore", SERIES, CHAPTER, "file", "1"])
    assert result.exit_code == 0
    assert "will apply on the next run" in result.output

    result = runner.invoke(app, ["filter", "run", SERIES])
    assert result.exit_code == 0
    assert f"{SERIES}/{CHAPTER}: filtered files: 1, filtered slices: 0" in result.output
    work = cfg.paths.work_root / SERIES / CHAPTER
    ingest = IngestArtifact.load(work / "ingest.json")
    assert ingest.filtered_files == ["004.jpg"]
    assert [f.index for f in ingest.files] == [0, 1, 2, 4]


def test_filter_force_filters_without_examples(tmp_path: Path, cfg: Config, monkeypatch) -> None:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    raw = cfg.paths.library_root / SERIES / CHAPTER
    raw.mkdir(parents=True)
    for i, color in enumerate(COLORS):
        images.plain_jpeg(raw / f"{i + 1:03d}.jpg", color=color)

    result = runner.invoke(app, ["filter", "force", SERIES, CHAPTER, "file", "0"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["filter", "run", SERIES])
    assert result.exit_code == 0
    assert f"{SERIES}/{CHAPTER}: filtered files: 1, filtered slices: 0" in result.output
    ingest = IngestArtifact.load(cfg.paths.work_root / SERIES / CHAPTER / "ingest.json")
    assert ingest.filtered_files == ["001.jpg"]
    slices = SlicesArtifact.load(cfg.paths.work_root / SERIES / CHAPTER / "slices.json")
    assert not any(s.filtered for s in slices.slices)


def test_filter_restore_rejects_bad_target(tmp_path: Path, cfg: Config, monkeypatch) -> None:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    result = runner.invoke(app, ["filter", "restore", SERIES, CHAPTER, "page", "0"])
    assert result.exit_code != 0
    assert "target must be 'file' or 'slice'" in result.output

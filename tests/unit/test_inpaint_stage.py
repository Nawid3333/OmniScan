"""Tests for the inpaint stage wiring and the `omniscan inpaint` CLI (card C6a, part 4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, GpuConfig, InpaintConfig, PathsConfig
from omniscan.core.schemas import (
    Band,
    BBox,
    InpaintArtifact,
    Manifest,
    OcrLine,
    Region,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
)
from omniscan.core.stage import make_context, run_stage
from omniscan.ingest.stage import IngestStage
from omniscan.inpaint.patches import load_patches
from omniscan.inpaint.stage import InpaintStage
from tests.fixtures import images

SERIES = "S"
CHAPTER = "Chapter 1"
PAGE = (400, 300)
STRIP_HEIGHT = 600
LINE_BOX = BBox(x0=30, y0=30, x1=120, y1=60)
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


def write_pages(cfg: Config, chapter: str = CHAPTER) -> None:
    """Two near-white raw JPEG pages (the flat-fill ring survives JPEG noise)."""
    raw = cfg.paths.library_root / SERIES / chapter
    raw.mkdir(parents=True, exist_ok=True)
    for i in (1, 2):
        images.plain_jpeg(raw / f"{i:03d}.jpg", size=PAGE, color=(250, 250, 250))


def write_slices(cfg: Config, chapter: str = CHAPTER) -> None:
    """A hand-written slices.json for the two-page strip."""
    work = cfg.paths.work_root / SERIES / chapter
    work.mkdir(parents=True, exist_ok=True)
    SlicesArtifact(
        strip_width=PAGE[0],
        strip_height=STRIP_HEIGHT,
        bands=[Band(y0=0, y1=STRIP_HEIGHT, color=(250, 250, 250))],
        slices=[Slice(index=0, y0=0, y1=STRIP_HEIGHT)],
    ).save(work / "slices.json")


def write_ocr(cfg: Config, chapter: str = CHAPTER, text: str = "안녕") -> None:
    """A hand-written ocr.json with one bubble-text region."""
    work = cfg.paths.work_root / SERIES / chapter
    work.mkdir(parents=True, exist_ok=True)
    RegionsArtifact(
        regions=[
            Region(
                id="r0001",
                slice_index=0,
                kind="bubble_text",
                bbox=LINE_BOX,
                lines=[OcrLine(bbox=LINE_BOX, text=text, score=0.9, engine="test")],
                text=text,
                confidence=0.9,
            )
        ]
    ).save(work / "ocr.json")


# ---------------------------------------------------------------- stage


def test_inpaint_stage_round_trip(cfg: Config) -> None:
    write_pages(cfg)
    ctx = make_context(cfg, SERIES, CHAPTER)
    assert run_stage(IngestStage(), ctx).status == "done"
    write_slices(cfg)
    write_ocr(cfg)

    outcome = run_stage(InpaintStage(), ctx)
    assert outcome.status == "done"
    assert outcome.metrics["regions"] == 1.0
    assert outcome.metrics["flat"] == 1.0

    artifact = InpaintArtifact.load(ctx.paths.artifact("inpaint.json"))
    (item,) = artifact.items
    assert item.region_id == "r0001"
    assert item.method == "flat"
    patches = load_patches(ctx.paths.artifact("patches.npz"))
    assert set(patches) == {"r0001"}
    pixels, mask = patches["r0001"]
    assert pixels.shape == (item.box.height, item.box.width, 3)
    assert mask.shape == (item.box.height, item.box.width)
    assert Manifest.load(ctx.paths.manifest).stages["inpaint"].status == "done"

    assert run_stage(InpaintStage(), make_context(cfg, SERIES, CHAPTER)).status == "skipped"

    cfg2 = cfg.model_copy(update={"inpaint": InpaintConfig(flat_tol=9.0)})
    assert run_stage(InpaintStage(), make_context(cfg2, SERIES, CHAPTER)).status == "done"
    assert run_stage(InpaintStage(), make_context(cfg2, SERIES, CHAPTER)).status == "skipped"

    write_ocr(cfg, text="다시")  # changed ocr.json re-runs the stage
    assert run_stage(InpaintStage(), make_context(cfg, SERIES, CHAPTER)).status == "done"


def test_inpaint_stage_missing_inputs_fail_with_documented_messages(cfg: Config) -> None:
    outcome = run_stage(InpaintStage(), make_context(cfg, SERIES, CHAPTER))
    assert outcome.status == "failed"
    assert outcome.error is not None
    assert "ingest.json missing — run the ingest stage first" in outcome.error

    write_pages(cfg)
    ctx = make_context(cfg, SERIES, CHAPTER)
    run_stage(IngestStage(), ctx)
    write_slices(cfg)
    outcome = run_stage(InpaintStage(), ctx)
    assert outcome.status == "failed"
    assert outcome.error is not None
    assert "ocr.json missing — run the ocr stage first" in outcome.error


def test_inpaint_stage_erases_watermark_and_sfx(cfg: Config) -> None:
    write_pages(cfg)
    ctx = make_context(cfg, SERIES, CHAPTER)
    assert run_stage(IngestStage(), ctx).status == "done"
    write_slices(cfg)
    work = cfg.paths.work_root / SERIES / CHAPTER
    RegionsArtifact(
        regions=[
            Region(
                id="r0001",
                slice_index=0,
                kind="watermark",
                bbox=LINE_BOX,
                lines=[OcrLine(bbox=LINE_BOX, text="wm", score=0.9, engine="test")],
            ),
            Region(
                id="r0002",
                slice_index=0,
                kind="sfx",
                bbox=LINE_BOX,
                lines=[OcrLine(bbox=LINE_BOX, text="쾅!", score=0.9, engine="test")],
            ),
        ]
    ).save(work / "ocr.json")

    outcome = run_stage(InpaintStage(), ctx)
    assert outcome.status == "done"
    assert outcome.metrics["regions"] == 2.0
    assert outcome.metrics["flat"] == 1.0
    assert outcome.metrics["needs_lama"] == 1.0
    assert outcome.metrics["skipped"] == 0.0
    artifact = InpaintArtifact.load(work / "inpaint.json")
    # the watermark's box sits on the flat page: filled; an sfx box is never flat-filled whole
    assert [(item.region_id, item.method, item.needs_lama) for item in artifact.items] == [
        ("r0001", "flat", False),
        ("r0002", "none", True),
    ]
    assert artifact.items[1].mask_px == 3456  # the sfx line box dilated by 3


def test_inpaint_stage_keeps_watermarks_when_told_to(cfg: Config) -> None:
    cfg = cfg.model_copy(update={"inpaint": cfg.inpaint.model_copy(update={"remove_watermarks": False})})
    write_pages(cfg)
    ctx = make_context(cfg, SERIES, CHAPTER)
    assert run_stage(IngestStage(), ctx).status == "done"
    write_slices(cfg)
    work = cfg.paths.work_root / SERIES / CHAPTER
    RegionsArtifact(
        regions=[
            Region(
                id="r0001",
                slice_index=0,
                kind="watermark",
                bbox=LINE_BOX,
                lines=[OcrLine(bbox=LINE_BOX, text="wm", score=0.9, engine="test")],
            )
        ]
    ).save(work / "ocr.json")

    outcome = run_stage(InpaintStage(), ctx)
    assert outcome.status == "done"
    assert outcome.metrics["regions"] == 0.0 and outcome.metrics["skipped"] == 1.0
    assert InpaintArtifact.load(work / "inpaint.json").items == []


# ---------------------------------------------------------------- CLI


def test_cli_inpaint_runs_only_the_inpaint_stage(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    for chapter in ("Chapter 1", "Chapter 2"):
        write_pages(cfg, chapter)
        write_slices(cfg, chapter)
        write_ocr(cfg, chapter)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["ingest", SERIES])
    assert result.exit_code == 0

    result = runner.invoke(app, ["inpaint", SERIES])
    assert result.exit_code == 0
    for chapter in ("Chapter 1", "Chapter 2"):
        assert f"{SERIES}/{chapter} inpaint: done" in result.output
        assert (cfg.paths.work_root / SERIES / chapter / "inpaint.json").is_file()
        assert (cfg.paths.work_root / SERIES / chapter / "patches.npz").is_file()
    assert "ingest:" not in result.output and "slice:" not in result.output  # only the inpaint stage ran

    result = runner.invoke(app, ["inpaint", SERIES])
    assert result.exit_code == 0
    assert result.output.count("skipped") == 2

    result = runner.invoke(app, ["inpaint", SERIES, "--chapter", "Chapter 1", "--force"])
    assert result.exit_code == 0
    assert f"{SERIES}/Chapter 1 inpaint: done" in result.output
    assert "Chapter 2" not in result.output

    result = runner.invoke(app, ["inpaint", "NoSuchSeries"])
    assert result.exit_code == 2
    assert "no chapters found" in result.output

    result = runner.invoke(app, ["slice", SERIES])
    assert result.exit_code == 0
    assert result.output.count("skipped") == 2


def test_cli_inpaint_is_not_a_stub() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "inpaint" in result.output

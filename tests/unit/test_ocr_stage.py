"""OCR stage wiring: real ingest/slice/strip, fake models, resumability and failure records (card C4a)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from omniscan.core.config import Config, GpuConfig, OcrConfig, PathsConfig
from omniscan.core.schemas import BBox, Region, RegionsArtifact
from omniscan.core.stage import ChapterContext, make_context, run_chapter, run_stage
from omniscan.ingest.stage import IngestStage
from omniscan.ocr.lines import LineBox
from omniscan.ocr.stage import OcrStage
from omniscan.slicer.stage import SliceStage

SERIES = "S"
CHAPTER = "Chapter 1"


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """CPU-only config with all paths under tmp_path and a strip-sized OCR tile."""
    return Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
        ocr=OcrConfig(tile_px=400),
    )


class FakeScheduler:
    """GpuScheduler handing out pre-built models (stands in for the VramManager)."""

    def __init__(self, models: dict[str, Any]) -> None:
        self.models = models

    def acquire(self, group: str) -> dict[str, Any]:
        assert group == "vision"
        return self.models

    def reset_peak(self) -> None:
        pass

    def peak_gib(self) -> float:
        return 0.0


class FakeLineDetector:
    """Scripted line detector: per cumulative tile index a list of (box, score)."""

    def __init__(
        self, script: dict[int, list[tuple[tuple[float, float, float, float], float]]] | None = None
    ) -> None:
        self.script = script or {}
        self.next_index = 0

    def detect(self, tiles: list[torch.Tensor]) -> list[list[LineBox]]:
        out = []
        for _ in tiles:
            out.append([LineBox(box=box, score=score) for box, score in self.script.get(self.next_index, [])])
            self.next_index += 1
        return out


class FakeRecognizer:
    """Scripted recognizer: the same reading for every crop."""

    def __init__(self, reading: tuple[str, float] = ("텍스트", 0.95)) -> None:
        self.reading = reading

    def read(self, crops: list[torch.Tensor]) -> list[tuple[str, float]]:
        return [self.reading for _ in crops]


def write_raw(cfg: Config) -> None:
    """Two 400x300 noise pages: a 400x600 strip."""
    rng = np.random.default_rng(0)
    raw = cfg.paths.library_root / SERIES / CHAPTER
    raw.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        Image.fromarray(rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)).save(
            raw / f"{i + 1:03d}.jpg", format="JPEG", quality=95
        )


def write_regions(ctx: ChapterContext, bbox: tuple[int, int, int, int] = (10, 50, 390, 200)) -> None:
    RegionsArtifact(
        regions=[
            Region(
                id="r0001",
                slice_index=0,
                kind="bubble_text",
                bbox=BBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
            )
        ]
    ).save(ctx.paths.artifact("regions.json"))


def prepared(cfg: Config) -> ChapterContext:
    """A chapter with ingest + slices done and a region waiting for OCR."""
    write_raw(cfg)
    ctx = make_context(cfg, SERIES, CHAPTER)
    assert [o.status for o in run_chapter([IngestStage(), SliceStage()], ctx)] == ["done", "done"]
    write_regions(ctx)
    return ctx


def test_ocr_stage_writes_ocr_json_and_is_resumable(cfg: Config) -> None:
    def scheduler() -> FakeScheduler:
        """Fresh scripted models — the fake detector counts tiles across runs."""
        return FakeScheduler(
            {
                "line_detector": FakeLineDetector({0: [((20, 60, 200, 90), 0.95)]}),
                "recognizer": FakeRecognizer(),
            }
        )

    ctx = prepared(cfg)
    ctx.gpu = scheduler()

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.status == "done"
    assert outcome.metrics["tiles"] == 1.0 and outcome.metrics["lines"] == 1.0
    assert outcome.metrics["regions"] == 1.0 and outcome.metrics["regions_empty"] == 0.0
    assert outcome.metrics["orphan_lines"] == 0.0
    built = RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions[0]
    assert built.text == "텍스트" and built.confidence == 0.95
    assert built.lines[0].bbox == BBox(x0=20, y0=60, x1=200, y1=90)
    assert built.lines[0].engine == cfg.ocr.rec_repo.split("/")[-1]

    manifest = make_context(cfg, SERIES, CHAPTER).manifest
    assert manifest.stages["ocr"].status == "done" and manifest.stages["ocr"].outputs == ["ocr.json"]
    assert run_stage(OcrStage(), make_context(cfg, SERIES, CHAPTER, scheduler())).status == "skipped"

    # changing ocr.rec_repo re-runs the stage (and the engine name follows the repo)
    rec = cfg.model_copy(update={"ocr": cfg.ocr.model_copy(update={"rec_repo": "PaddlePaddle/other_rec"})})
    run_stage(OcrStage(), make_context(rec, SERIES, CHAPTER, scheduler()))
    rebuilt = RegionsArtifact.load(rec.paths.work_root / SERIES / CHAPTER / "ocr.json").regions[0]
    assert rebuilt.lines[0].engine == "other_rec"

    # changing detect.reading_direction re-runs it
    direction = cfg.model_copy(update={"detect": cfg.detect.model_copy(update={"reading_direction": "rtl"})})
    assert run_stage(OcrStage(), make_context(direction, SERIES, CHAPTER, scheduler())).status == "done"

    # changing regions.json re-runs it
    write_regions(make_context(cfg, SERIES, CHAPTER), bbox=(10, 50, 390, 100))
    assert run_stage(OcrStage(), make_context(cfg, SERIES, CHAPTER, scheduler())).status == "done"


def test_ocr_stage_missing_upstream_artifacts_fail_and_are_recorded(cfg: Config) -> None:
    write_raw(cfg)
    scheduler = FakeScheduler({"line_detector": FakeLineDetector(), "recognizer": FakeRecognizer()})

    outcome = run_stage(OcrStage(), make_context(cfg, SERIES, CHAPTER, scheduler))

    assert outcome.status == "failed"
    assert outcome.error is not None and "regions.json missing — run the detect stage first" in outcome.error
    manifest = make_context(cfg, SERIES, CHAPTER).manifest
    assert manifest.stages["ocr"].status == "failed"
    assert (
        manifest.stages["ocr"].error is not None
        and "run the detect stage first" in manifest.stages["ocr"].error
    )

    ctx = make_context(cfg, SERIES, CHAPTER)
    assert [o.status for o in run_chapter([IngestStage(), SliceStage()], ctx)] == ["done", "done"]
    write_regions(ctx)
    (ctx.paths.work_dir / "ingest.json").unlink()
    outcome = run_stage(OcrStage(), make_context(cfg, SERIES, CHAPTER, scheduler))
    assert outcome.status == "failed"
    assert outcome.error is not None and "ingest.json missing — run the slice stage first" in outcome.error


def _scheduler_with(lines: dict[int, list[tuple[tuple[float, float, float, float], float]]]) -> FakeScheduler:
    return FakeScheduler({"line_detector": FakeLineDetector(lines), "recognizer": FakeRecognizer()})


def test_regions_below_drop_conf_are_dropped_and_counted(cfg: Config) -> None:
    strict = cfg.model_copy(update={"ocr": OcrConfig(tile_px=400, drop_conf=0.99)})
    ctx = prepared(strict)
    ctx.gpu = _scheduler_with({0: [((20, 60, 200, 90), 0.95)]})

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.status == "done"
    assert outcome.metrics["regions"] == 1.0 and outcome.metrics["regions_dropped"] == 1.0
    assert RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions == []


def test_regions_without_any_text_are_dropped(cfg: Config) -> None:
    ctx = prepared(cfg)
    ctx.gpu = _scheduler_with({})  # the line detector finds nothing

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.metrics["regions_empty"] == 1.0 and outcome.metrics["regions_dropped"] == 1.0
    assert RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions == []


def test_confident_regions_are_kept(cfg: Config) -> None:
    ctx = prepared(cfg)
    ctx.gpu = _scheduler_with({0: [((20, 60, 200, 90), 0.95)]})

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.metrics["regions_dropped"] == 0.0
    assert len(RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions) == 1

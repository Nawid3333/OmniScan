"""Detect stage wiring: batching, tile skipping, resumability, artifacts — real ingest/slice, fake detector (C3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from omniscan.core.config import Config, DetectConfig, GpuConfig, PathsConfig, SlicerConfig
from omniscan.core.schemas import IngestArtifact, Region, RegionsArtifact, SlicesArtifact
from omniscan.core.stage import ChapterContext, make_context, run_chapter, run_stage
from omniscan.detect.model import RawDet
from omniscan.detect.postprocess import build_regions, merge_detections, tile_det_to_strip
from omniscan.detect.stage import DetectStage
from omniscan.detect.tiles import keep_tiles, plan_tiles
from omniscan.ingest.stage import IngestStage
from omniscan.slicer.stage import SliceStage
from omniscan.watermark.store import WatermarkStore

SERIES = "S"
CHAPTER = "Chapter 1"
SLICER = SlicerConfig(
    band_min_px=50,
    target_height=300,
    min_height=100,
    max_height=400,
    hard_max_height=1000,
    uniform_tol=10,
    max_drift=2.0,
)


def stage_cfg(tmp_path: Path, **detect: Any) -> Config:
    """CPU-only config with all paths under tmp_path and a small detect tile."""
    return Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
        slicer=SLICER,
        detect=DetectConfig(tile_px=100, **detect),
    )


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return stage_cfg(tmp_path)


class FakeScheduler:
    """GpuScheduler handing out pre-built models (stands in for the VramManager)."""

    def __init__(self, models: dict[str, Any]) -> None:
        self.models = models

    def acquire(self, group: str) -> dict[str, Any]:
        return self.models

    def reset_peak(self) -> None:
        pass

    def peak_gib(self) -> float:
        return 0.0


class FakeDetector:
    """Scripted detector: returns RawDets per cumulative crop index and records each call's crop shapes."""

    def __init__(self, script: dict[int, list[RawDet]] | None = None) -> None:
        self.script = script or {}
        self.shapes: list[list[tuple[int, int]]] = []
        self.next_index = 0

    def detect(self, crops: list[torch.Tensor]) -> list[list[RawDet]]:
        shapes = []
        for crop in crops:
            shapes.append((int(crop.shape[-2]), int(crop.shape[-1])))
            self.next_index += 1
        self.shapes.append(shapes)
        start = self.next_index - len(crops)
        return [list(self.script.get(start + offset, [])) for offset in range(len(crops))]


def noise_page(path: Path, width: int, height: int, rng: np.random.Generator) -> None:
    Image.fromarray(rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)).save(
        path, format="JPEG", quality=95
    )


def banded_page(path: Path, width: int, white: int, noise: int, rng: np.random.Generator) -> None:
    rows = [
        np.full((white, width, 3), 255, dtype=np.uint8),
        rng.integers(0, 256, size=(noise, width, 3), dtype=np.uint8),
        np.full((white, width, 3), 255, dtype=np.uint8),
    ]
    Image.fromarray(np.concatenate(rows)).save(path, format="JPEG", quality=95)


def prepare(cfg: Config, pages: int, size: tuple[int, int], banded: bool) -> ChapterContext:
    """Write a chapter of synthetic pages (no stages run yet)."""
    rng = np.random.default_rng(0)
    raw = cfg.paths.library_root / SERIES / CHAPTER
    raw.mkdir(parents=True, exist_ok=True)
    for i in range(pages):
        path = raw / f"{i + 1:03d}.jpg"
        if banded:
            banded_page(path, size[0], 100, size[1], rng)
        else:
            noise_page(path, size[0], size[1], rng)
    return make_context(cfg, SERIES, CHAPTER)


def expected_regions(ctx: ChapterContext, script: dict[int, list[RawDet]]) -> list[Region]:
    """What build_regions(merge_detections(...)) gives for the scripted detections (independent recomputation)."""
    ingest = IngestArtifact.load(ctx.paths.artifact("ingest.json"))
    slices = SlicesArtifact.load(ctx.paths.artifact("slices.json"))
    active = [(s.y0, s.y1) for s in slices.slices if not s.blank and not s.filtered]
    tiles = keep_tiles(
        plan_tiles(ingest.strip_width, ingest.strip_height, ctx.cfg.detect.tile_px, ctx.cfg.detect.overlap),
        active,
    )
    dets = [
        tile_det_to_strip(tile, det.cls, det.score, det.box)
        for index, tile in enumerate(tiles)
        for det in script.get(index, [])
    ]
    merged = merge_detections(
        dets,
        nms_iou=ctx.cfg.detect.nms_iou,
        contain_thr=ctx.cfg.detect.contain_thr,
        edge_penalty=ctx.cfg.detect.edge_penalty,
    )
    return build_regions(
        merged,
        slices.slices,
        strip_width=ingest.strip_width,
        strip_height=ingest.strip_height,
        merge_bubble_text=ctx.cfg.detect.merge_bubble_text,
        direction=ctx.cfg.detect.reading_direction,
    )


def test_detect_stage_batches_tiles(tmp_path: Path) -> None:
    # 3x150 noise rows -> a 100x450 strip -> 8 tiles of 100x100 at overlap 0.5, three per forward pass
    cfg = stage_cfg(tmp_path, batch_size=3)
    script = {
        0: [RawDet("text_bubble", 0.9, (10, 10, 90, 90))],
        7: [RawDet("text_free", 0.6, (20, 30, 80, 70))],
    }
    detector = FakeDetector(script)
    ctx = prepare(cfg, pages=3, size=(100, 150), banded=False)
    run_chapter([IngestStage(), SliceStage()], ctx)
    ctx.gpu = FakeScheduler({"detector": detector})

    outcome = run_stage(DetectStage(), ctx)

    assert outcome.status == "done"
    assert detector.shapes == [[(100, 100)] * 3, [(100, 100)] * 3, [(100, 100)] * 2]
    assert outcome.metrics["tiles"] == 8.0 and outcome.metrics["tiles_skipped"] == 0.0
    assert outcome.metrics["raw_detections"] == 2.0 and outcome.metrics["merged_detections"] == 2.0
    assert outcome.metrics["regions"] == 2.0
    slices = SlicesArtifact.load(ctx.paths.artifact("slices.json"))
    artifact = RegionsArtifact.load(ctx.paths.artifact("regions.json"))
    assert artifact.regions == expected_regions(ctx, script)
    assert len(artifact.regions) == 2  # sanity: the scripted boxes really produced two regions
    assert slices.slices  # and the chapter really has active slices


def test_detect_stage_skips_tiles_of_blank_slices(cfg: Config) -> None:
    # 2 banded pages -> a 100x800 strip; the slicer leaves a blank band the middle tiles fall into
    detector = FakeDetector()
    ctx = prepare(cfg, pages=2, size=(100, 200), banded=True)
    run_chapter([IngestStage(), SliceStage()], ctx)
    ctx.gpu = FakeScheduler({"detector": detector})

    outcome = run_stage(DetectStage(), ctx)

    slices = SlicesArtifact.load(ctx.paths.artifact("slices.json"))
    assert any(s.blank for s in slices.slices)  # the fixture really has a blank band
    planned = len(plan_tiles(100, 800, 100, 0.5))
    assert outcome.metrics["tiles"] + outcome.metrics["tiles_skipped"] == float(planned)
    assert outcome.metrics["tiles_skipped"] >= 1.0
    assert sum(len(call) for call in detector.shapes) == int(outcome.metrics["tiles"])
    assert all(shape == (100, 100) for call in detector.shapes for shape in call)


def test_detect_stage_is_resumable_and_invalidated_by_config(cfg: Config) -> None:
    stages = [IngestStage(), SliceStage(), DetectStage()]

    def run(pipeline_cfg: Config) -> list[str]:
        scheduler = FakeScheduler({"detector": FakeDetector()})
        outcomes = run_chapter(stages, make_context(pipeline_cfg, SERIES, CHAPTER, scheduler))
        return [o.status for o in outcomes]

    prepare(cfg, pages=2, size=(100, 200), banded=True)
    assert run(cfg) == ["done", "done", "done"]
    assert make_context(cfg, SERIES, CHAPTER).manifest.stages["detect"].status == "done"
    assert run(cfg) == ["skipped", "skipped", "skipped"]

    threshold = cfg.model_copy(update={"detect": cfg.detect.model_copy(update={"threshold": 0.4})})
    assert run(threshold) == ["skipped", "skipped", "done"]

    slicer = cfg.model_copy(update={"slicer": cfg.slicer.model_copy(update={"uniform_tol": 1})})
    assert run(slicer) == ["skipped", "done", "done"]  # slices.json changed


def test_detect_stage_no_active_slices(cfg: Config) -> None:
    raw = cfg.paths.library_root / SERIES / CHAPTER
    raw.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 450), (255, 255, 255)).save(raw / "001.jpg", format="JPEG", quality=95)
    ctx = make_context(cfg, SERIES, CHAPTER)
    run_chapter([IngestStage(), SliceStage()], ctx)
    detector = FakeDetector()
    ctx.gpu = FakeScheduler({"detector": detector})

    outcome = run_stage(DetectStage(), ctx)

    assert outcome.status == "done"
    assert detector.shapes == []  # the detector was never called
    artifact = RegionsArtifact.load(ctx.paths.artifact("regions.json"))
    assert artifact.regions == []
    assert outcome.metrics["tiles"] == 0.0 and outcome.metrics["tiles_skipped"] == 8.0


def test_detect_stage_reclassifies_fixed_position_watermarks(tmp_path: Path) -> None:
    # 3 noise pages of 100x150 -> a 100x450 strip; tile 0's box lands at strip (20, 30, 80, 70) on
    # page 1 and tile 4's at (20, 230, 80, 270). A stored zone at fractions (0.2, 0.2, 0.8, 0.4)
    # resolves to x 20..80, y file.y0+30..file.y0+60: on page 1 that is (20, 30, 80, 60), which
    # holds 60x30 of the first region's own 60x40 area (IoA 0.75); the second region stays clear.
    cfg = stage_cfg(tmp_path)
    script = {
        0: [RawDet("text_free", 0.6, (20, 30, 80, 70))],
        4: [RawDet("text_free", 0.6, (20, 30, 80, 70))],
    }
    ctx = prepare(cfg, pages=3, size=(100, 150), banded=False)
    run_chapter([IngestStage(), SliceStage()], ctx)
    WatermarkStore(ctx.series.work_dir).add(0.2, 0.2, 0.8, 0.4, note="corner stamp")
    ctx.gpu = FakeScheduler({"detector": FakeDetector(script)})

    outcome = run_stage(DetectStage(), ctx)

    assert outcome.status == "done"
    assert outcome.metrics["regions"] == 2.0 and outcome.metrics["watermarked"] == 1.0
    artifact = RegionsArtifact.load(ctx.paths.artifact("regions.json"))
    kinds = {(r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1): r.kind for r in artifact.regions}
    assert kinds == {(20, 30, 80, 70): "watermark", (20, 230, 80, 270): "free_text"}


def test_detect_stage_invalidated_by_a_new_watermark_region(cfg: Config) -> None:
    ctx = prepare(cfg, pages=2, size=(100, 200), banded=True)
    run_chapter([IngestStage(), SliceStage()], ctx)
    ctx.gpu = FakeScheduler({"detector": FakeDetector()})

    assert run_stage(DetectStage(), ctx).status == "done"  # watermarks.json does not exist yet
    assert run_stage(DetectStage(), ctx).status == "skipped"

    WatermarkStore(ctx.series.work_dir).add(0.9, 0.9, 1.0, 1.0, note="corner site stamp")

    assert run_stage(DetectStage(), ctx).status == "done"  # watermarks.json now exists


def test_detect_stage_without_upstream_artifacts_fails(cfg: Config) -> None:
    prepare(cfg, pages=3, size=(100, 150), banded=False)
    ctx = make_context(cfg, SERIES, CHAPTER, FakeScheduler({"detector": FakeDetector()}))
    outcome = run_stage(DetectStage(), ctx)
    assert outcome.status == "failed"
    assert outcome.error is not None and "ingest.json missing — run the slice stage first" in outcome.error

    run_chapter([IngestStage()], make_context(cfg, SERIES, CHAPTER))
    outcome = run_stage(
        DetectStage(), make_context(cfg, SERIES, CHAPTER, FakeScheduler({"detector": FakeDetector()}))
    )
    assert outcome.status == "failed"
    assert outcome.error is not None and "slices.json missing — run the slice stage first" in outcome.error

"""VRAM model groups and the CLI's detect/ocr commands: manager wiring, no transformers at import (cards C3, C4a)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, GpuConfig, OllamaConfig, PathsConfig
from omniscan.core.schemas import RegionsArtifact
from omniscan.detect.model import Detector, RawDet
from omniscan.ocr.lines import LineBox
from omniscan.ocr.model import LineDetector, LineRecognizer

SERIES = "S"
runner = CliRunner()


def cli_cfg(tmp_path: Path) -> Config:
    """CPU-only config with all paths under tmp_path and an unreachable Ollama (nothing to evict)."""
    return Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
        ollama=OllamaConfig(local_url="http://127.0.0.1:9"),
    )


class FakeManager:
    """The GpuScheduler/release surface _run_stages needs, with a scripted vision loader."""

    def __init__(self, detector: Any, line_detector: Any, recognizer: Any) -> None:
        self.detector = detector
        self.line_detector = line_detector
        self.recognizer = recognizer
        self.released = False

    def acquire(self, group: str) -> dict[str, Any]:
        assert group == "vision"
        return {
            "detector": self.detector,
            "line_detector": self.line_detector,
            "recognizer": self.recognizer,
        }

    def reset_peak(self) -> None:
        pass

    def peak_gib(self) -> float:
        return 0.0

    def release(self) -> None:
        self.released = True


class FakeDetector:
    """Scripted detector: one text_bubble in the very first crop, nothing in any later crop."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[int, int]]] = []
        self.seen = 0

    def detect(self, crops: list[torch.Tensor]) -> list[list[RawDet]]:
        self.calls.append([(int(c.shape[-2]), int(c.shape[-1])) for c in crops])
        out = [[RawDet("text_bubble", 0.9, (10, 10, 90, 90))] if self.seen == 0 else []]
        self.seen += 1
        for _ in crops[1:]:
            out.append([])
            self.seen += 1
        return out


class FakeLineDetector:
    """Scripted line detector: one line inside the first region, nothing in later crops."""

    def __init__(self) -> None:
        self.seen = 0

    def detect(self, tiles: list[torch.Tensor]) -> list[list[LineBox]]:
        out = [[LineBox(box=(20.0, 20.0, 80.0, 40.0), score=0.95)] if self.seen == 0 else []]
        self.seen += 1
        return out


class FakeRecognizer:
    """Scripted recognizer: the same reading for every crop."""

    def read(self, crops: list[torch.Tensor]) -> list[tuple[str, float]]:
        return [("안녕", 0.97) for _ in crops]


def raw_jpegs(cfg: Config, chapter: str) -> None:
    """Three 400x300 noise pages: strip 400x900, one non-blank slice, four 400x400 detect tiles."""
    rng = np.random.default_rng(0)
    raw = cfg.paths.library_root / SERIES / chapter
    raw.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        path = raw / f"{i + 1:03d}.jpg"
        Image.fromarray(rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)).save(
            path, format="JPEG", quality=95
        )


# ---------------------------------------------------------------- groups (test 18)


def test_build_vram_manager_registers_vision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from omniscan.gpu.groups import VISION_GROUP, build_vram_manager

    fakes = (object(), object(), object())
    seen: dict[str, Any] = {}

    def fake_detector_load(cfg: Any, device: torch.device) -> object:
        seen["detector"] = (cfg, device)
        return fakes[0]

    def fake_line_detector_load(cfg: Any, device: torch.device) -> object:
        seen["line_detector"] = (cfg, device)
        return fakes[1]

    def fake_recognizer_load(cfg: Any, device: torch.device) -> object:
        seen["recognizer"] = (cfg, device)
        return fakes[2]

    monkeypatch.setattr(Detector, "load", fake_detector_load)
    monkeypatch.setattr(LineDetector, "load", fake_line_detector_load)
    monkeypatch.setattr(LineRecognizer, "load", fake_recognizer_load)

    cfg = cli_cfg(tmp_path)
    manager = build_vram_manager(cfg)
    assert manager.device.type == "cpu"
    models = manager.acquire(VISION_GROUP)
    assert models == {"detector": fakes[0], "line_detector": fakes[1], "recognizer": fakes[2]}
    assert seen["detector"][0] is cfg.detect and seen["detector"][1].type == "cpu"
    assert seen["line_detector"][0] is cfg.ocr and seen["recognizer"][0] is cfg.ocr
    assert manager._groups[VISION_GROUP].est_gib == 3.0
    assert manager.resident == VISION_GROUP
    manager.release()
    assert manager.resident is None


def test_import_groups_does_not_import_transformers() -> None:
    code = (
        "import sys; import omniscan.gpu.groups; raise SystemExit(1 if 'transformers' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ---------------------------------------------------------------- CLI (tests 19, 20)


def test_cli_detect_runs_pipeline_with_vram_manager(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = cli_cfg(tmp_path)
    for chapter in ("Chapter 1", "Chapter 2"):
        raw_jpegs(cfg, chapter)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    manager = FakeManager(FakeDetector(), FakeLineDetector(), FakeRecognizer())
    monkeypatch.setattr("omniscan.gpu.groups.build_vram_manager", lambda cfg_: manager)

    result = runner.invoke(app, ["detect", SERIES])
    assert result.exit_code == 0
    for chapter in ("Chapter 1", "Chapter 2"):
        assert f"{SERIES}/{chapter} detect: done" in result.output
        assert (cfg.paths.work_root / SERIES / chapter / "regions.json").is_file()
    assert manager.released  # the models left VRAM when the command ended

    limited = runner.invoke(app, ["detect", SERIES, "--chapter", "Chapter 2"])
    assert limited.exit_code == 0
    assert "Chapter 1" not in limited.output and f"{SERIES}/Chapter 2 detect: skipped" in limited.output

    forced = runner.invoke(app, ["detect", SERIES, "--chapter", "Chapter 2", "--force"])
    assert forced.exit_code == 0
    assert f"{SERIES}/Chapter 2 detect: done" in forced.output

    unknown = runner.invoke(app, ["detect", "NoSuchSeries"])
    assert unknown.exit_code == 2
    assert "no chapters found" in unknown.output


def test_cli_detect_writes_regions_from_fake_detector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = cli_cfg(tmp_path)
    raw_jpegs(cfg, "Chapter 1")
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    detector = FakeDetector()
    monkeypatch.setattr(
        "omniscan.gpu.groups.build_vram_manager",
        lambda cfg_: FakeManager(detector, FakeLineDetector(), FakeRecognizer()),
    )

    result = runner.invoke(app, ["detect", SERIES])
    assert result.exit_code == 0
    assert detector.calls and len(detector.calls[0]) == 4  # a 400x900 strip -> four 400x400 tiles

    regions = RegionsArtifact.load(cfg.paths.work_root / SERIES / "Chapter 1" / "regions.json").regions
    assert [(r.kind, r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1) for r in regions] == [
        ("bubble_text", 10, 10, 90, 90)
    ]


def test_cli_slice_builds_no_vram_manager(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = cli_cfg(tmp_path)
    raw_jpegs(cfg, "Chapter 1")
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    def explode(cfg_: Config) -> None:
        raise AssertionError("slice must not build a VRAM manager")

    monkeypatch.setattr("omniscan.gpu.groups.build_vram_manager", explode)
    result = runner.invoke(app, ["slice", SERIES])
    assert result.exit_code == 0
    assert f"{SERIES}/Chapter 1 slice: done" in result.output


def test_cli_ocr_runs_pipeline_with_vram_manager(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = cli_cfg(tmp_path)
    for chapter in ("Chapter 1", "Chapter 2"):
        raw_jpegs(cfg, chapter)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    manager = FakeManager(FakeDetector(), FakeLineDetector(), FakeRecognizer())
    monkeypatch.setattr("omniscan.gpu.groups.build_vram_manager", lambda cfg_: manager)

    result = runner.invoke(app, ["ocr", SERIES])
    assert result.exit_code == 0
    for chapter in ("Chapter 1", "Chapter 2"):
        assert f"{SERIES}/{chapter} ingest: done" in result.output
        assert f"{SERIES}/{chapter} detect: done" in result.output
        assert f"{SERIES}/{chapter} ocr: done" in result.output
    assert manager.released  # the models left VRAM when the command ended

    regions = RegionsArtifact.load(cfg.paths.work_root / SERIES / "Chapter 1" / "ocr.json").regions
    assert [(r.text, r.confidence) for r in regions] == [("안녕", 0.97)]

    limited = runner.invoke(app, ["ocr", SERIES, "--chapter", "Chapter 2"])
    assert limited.exit_code == 0
    assert "Chapter 1" not in limited.output and f"{SERIES}/Chapter 2 ocr: skipped" in limited.output

    forced = runner.invoke(app, ["ocr", SERIES, "--chapter", "Chapter 2", "--force"])
    assert forced.exit_code == 0
    assert f"{SERIES}/Chapter 2 ocr: done" in forced.output


def test_cli_ocr_writes_regions_from_fake_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = cli_cfg(tmp_path)
    raw_jpegs(cfg, "Chapter 1")
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    line_detector = FakeLineDetector()
    monkeypatch.setattr(
        "omniscan.gpu.groups.build_vram_manager",
        lambda cfg_: FakeManager(FakeDetector(), line_detector, FakeRecognizer()),
    )

    result = runner.invoke(app, ["ocr", SERIES])
    assert result.exit_code == 0
    assert line_detector.seen == 1  # only the tile touching the detected region reaches the detector

    regions = RegionsArtifact.load(cfg.paths.work_root / SERIES / "Chapter 1" / "ocr.json").regions
    assert [(line.text, line.score) for line in regions[0].lines] == [("안녕", 0.97)]


def test_detect_is_no_longer_a_stub() -> None:
    result = runner.invoke(app, ["detect", "NoSuchSeries"])
    assert "not implemented yet" not in result.output


def test_ocr_is_no_longer_a_stub() -> None:
    result = runner.invoke(app, ["ocr", "NoSuchSeries"])
    assert "not implemented yet" not in result.output

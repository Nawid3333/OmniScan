"""VRAM model groups and the CLI's detect command: manager wiring, no transformers at import (card C3)."""

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
from omniscan.detect.model import Detector, RawDet

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

    def __init__(self, detector: Any) -> None:
        self.detector = detector
        self.released = False

    def acquire(self, group: str) -> dict[str, Any]:
        assert group == "vision"
        return {"detector": self.detector}

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

    fake = object()
    seen: dict[str, Any] = {}

    def fake_load(cfg: Any, device: torch.device) -> object:
        seen["call"] = (cfg, device)
        return fake

    monkeypatch.setattr(Detector, "load", fake_load)

    cfg = cli_cfg(tmp_path)
    manager = build_vram_manager(cfg)
    assert manager.device.type == "cpu"
    models = manager.acquire(VISION_GROUP)
    assert models == {"detector": fake}
    assert seen["call"][0] is cfg.detect and seen["call"][1].type == "cpu"
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
    manager = FakeManager(FakeDetector())
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
    monkeypatch.setattr("omniscan.gpu.groups.build_vram_manager", lambda cfg_: FakeManager(detector))

    result = runner.invoke(app, ["detect", SERIES])
    assert result.exit_code == 0
    assert detector.calls and len(detector.calls[0]) == 4  # a 400x900 strip -> four 400x400 tiles

    from omniscan.core.schemas import RegionsArtifact

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


def test_detect_is_no_longer_a_stub() -> None:
    result = runner.invoke(app, ["detect", "NoSuchSeries"])
    assert "not implemented yet" not in result.output

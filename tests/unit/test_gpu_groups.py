"""VRAM model groups and the CLI's detect/ocr commands: manager wiring, no transformers at import (cards C3, C4a)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, GpuConfig, OcrConfig, OllamaConfig, PathsConfig
from omniscan.core.schemas import BBox, OcrLine, Region, RegionsArtifact, Slice, SlicesArtifact
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

    def fake_detector_load(cfg: Any, device: torch.device, models_dir: Path | None) -> object:
        seen["detector"] = (cfg, device, models_dir)
        return fakes[0]

    def fake_line_detector_load(cfg: Any, device: torch.device, models_dir: Path | None) -> object:
        seen["line_detector"] = (cfg, device, models_dir)
        return fakes[1]

    def fake_recognizer_load(cfg: Any, device: torch.device, models_dir: Path | None) -> object:
        seen["recognizer"] = (cfg, device, models_dir)
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
    assert seen["detector"][2] == seen["line_detector"][2] == seen["recognizer"][2] == cfg.paths.models_dir
    assert manager._groups[VISION_GROUP].est_gib == 3.0
    assert manager.resident == VISION_GROUP
    manager.release()
    assert manager.resident is None


def test_import_groups_does_not_import_transformers() -> None:
    code = (
        "import sys; import omniscan.gpu.groups; raise SystemExit(1 if 'transformers' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_build_vram_manager_registers_inpaint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from omniscan.gpu.groups import INPAINT_GROUP, VISION_GROUP, build_vram_manager
    from omniscan.inpaint.lama import LamaInpainter

    fake_lama, fake_detector = object(), object()
    seen: dict[str, Any] = {}

    def fake_load(cfg: Any, models_dir: Path, device: torch.device) -> object:
        seen["call"] = (cfg, models_dir, device)
        return fake_lama

    monkeypatch.setattr(LamaInpainter, "load", fake_load)
    monkeypatch.setattr(Detector, "load", lambda cfg_, device, models_dir=None: fake_detector)
    monkeypatch.setattr(LineDetector, "load", lambda cfg_, device, models_dir=None: object())
    monkeypatch.setattr(LineRecognizer, "load", lambda cfg_, device, models_dir=None: object())

    cfg = cli_cfg(tmp_path)
    manager = build_vram_manager(cfg)
    assert manager.acquire(INPAINT_GROUP) == {"lama": fake_lama}
    assert seen["call"] == (cfg.inpaint, cfg.paths.models_dir, manager.device)
    manager.release()
    assert manager.acquire(VISION_GROUP)["detector"] is fake_detector  # the vision group stays
    manager.release()


# ---------------------------------------------------------------- GPU warm-up (card G2, test 2)


def test_build_vram_manager_starts_warmup_once_on_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from omniscan.gpu import groups as groups_module
    from omniscan.gpu.groups import build_vram_manager
    from omniscan.gpu.warmup import GpuWarmup

    started: list[torch.device] = []

    def fake_start_warmup(device: torch.device) -> GpuWarmup:
        started.append(device)
        return GpuWarmup(torch.device("cpu"))  # an already-done stand-in handle

    monkeypatch.setattr(groups_module, "start_warmup", fake_start_warmup)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=16 * 2**30)
    )
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, device: None)

    cfg = cli_cfg(tmp_path).model_copy(update={"gpu": GpuConfig(device="cuda")})
    manager = build_vram_manager(cfg)
    assert started == [torch.device("cuda")]  # exactly one start_warmup call for the resolved device
    assert manager.warmup is not None and manager.warmup.done
    assert manager.device.type == "cuda"

    manager = build_vram_manager(cfg.model_copy(update={"gpu": GpuConfig(device="cuda", warmup=False)}))
    assert started == [torch.device("cuda")]  # warm-up disabled: not called again
    assert manager.warmup is None

    manager = build_vram_manager(cli_cfg(tmp_path))  # cpu device: nothing starts
    assert started == [torch.device("cuda")]
    assert manager.warmup is None


# ---------------------------------------------------------------- engine dispatch (card O1b, test 8)


def test_vision_group_for_manga_ocr_loads_the_reader_not_the_detector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from omniscan.gpu.groups import VISION_GROUP, build_vram_manager
    from omniscan.ocr.crop_readers import MangaOcrReader

    fake_reader, fake_detector = object(), object()
    seen: dict[str, Any] = {}

    def fake_reader_load(cfg: Any, device: torch.device, models_dir: Path | None) -> object:
        seen["reader"] = (cfg, device, models_dir)
        return fake_reader

    def fail_line_detector(cfg_: Any, device: torch.device, models_dir: Path | None = None) -> object:
        raise AssertionError("LineDetector.load must not run for manga_ocr")

    monkeypatch.setattr(MangaOcrReader, "load", fake_reader_load)
    monkeypatch.setattr(Detector, "load", lambda cfg_, device, models_dir=None: fake_detector)
    monkeypatch.setattr(LineDetector, "load", fail_line_detector)

    cfg = cli_cfg(tmp_path).model_copy(update={"ocr": OcrConfig(engine="manga_ocr")})
    manager = build_vram_manager(cfg)

    models = manager.acquire(VISION_GROUP)

    assert models == {"detector": fake_detector, "reader": fake_reader}
    assert seen["reader"] == (cfg.ocr, manager.device, cfg.paths.models_dir)
    assert manager._groups[VISION_GROUP].est_gib == 3.0  # only paddleocr_vl is budgeted higher
    manager.release()


def test_vision_group_for_paddleocr_vl_loads_the_reader_not_the_detector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from omniscan.gpu.groups import VISION_GROUP, build_vram_manager
    from omniscan.ocr.crop_readers import PaddleOcrVlReader

    fake_reader, fake_detector = object(), object()
    seen: dict[str, Any] = {}

    def fake_reader_load(cfg: Any, device: torch.device, models_dir: Path | None) -> object:
        seen["reader"] = (cfg, device, models_dir)
        return fake_reader

    def fail_line_detector(cfg_: Any, device: torch.device, models_dir: Path | None = None) -> object:
        raise AssertionError("LineDetector.load must not run for paddleocr_vl")

    monkeypatch.setattr(PaddleOcrVlReader, "load", fake_reader_load)
    monkeypatch.setattr(Detector, "load", lambda cfg_, device, models_dir=None: fake_detector)
    monkeypatch.setattr(LineDetector, "load", fail_line_detector)

    cfg = cli_cfg(tmp_path).model_copy(update={"ocr": OcrConfig(engine="paddleocr_vl")})
    manager = build_vram_manager(cfg)

    models = manager.acquire(VISION_GROUP)

    assert models == {"detector": fake_detector, "reader": fake_reader}
    assert seen["reader"] == (cfg.ocr, manager.device, cfg.paths.models_dir)
    assert manager._groups[VISION_GROUP].est_gib == 6.6  # 3.0 detector + 3.6 for the 0.9 B VLM
    manager.release()


def test_vision_group_for_unknown_engine_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from omniscan.gpu.groups import VISION_GROUP, build_vram_manager

    monkeypatch.setattr(Detector, "load", lambda cfg_, device, models_dir=None: object())

    cfg = cli_cfg(tmp_path).model_copy(update={"ocr": OcrConfig(engine="paddleocr_vl")})
    monkeypatch.setattr(cfg.ocr, "engine", "not_an_engine")  # a Literal can hold no other engine
    manager = build_vram_manager(cfg)

    with pytest.raises(ValueError, match="OCR engine 'not_an_engine' is not available yet"):
        manager.acquire(VISION_GROUP)


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


# ---------------------------------------------------------------- CLI inpaint --lama (test 13)


class FakeLamaInpainter:
    """No-op inpainting: outside its mask the model output is the input anyway."""

    def inpaint(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return image


class FakeLamaManager:
    """The GpuScheduler/release surface _run_stages needs, for the inpaint group."""

    def __init__(self) -> None:
        self.released = False

    def acquire(self, group: str) -> dict[str, Any]:
        assert group == "inpaint"
        return {"lama": FakeLamaInpainter()}

    def reset_peak(self) -> None:
        pass

    def peak_gib(self) -> float:
        return 0.0

    def release(self) -> None:
        self.released = True


def write_slices_and_ocr(cfg: Config, chapter: str) -> None:
    """A slices.json for the 400x900 strip and one SFX region (always needs_lama)."""
    work = cfg.paths.work_root / SERIES / chapter
    work.mkdir(parents=True, exist_ok=True)
    SlicesArtifact(
        strip_width=400,
        strip_height=900,
        bands=[],
        slices=[Slice(index=0, y0=0, y1=900)],
    ).save(work / "slices.json")
    box = BBox(x0=10, y0=10, x1=90, y1=90)
    RegionsArtifact(
        regions=[
            Region(
                id="r0001",
                slice_index=0,
                kind="sfx",
                bbox=box,
                lines=[OcrLine(bbox=box, text="쾅!", score=0.9, engine="test")],
            )
        ]
    ).save(work / "ocr.json")


def test_cli_inpaint_lama_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = cli_cfg(tmp_path)
    raw_jpegs(cfg, "Chapter 1")
    write_slices_and_ocr(cfg, "Chapter 1")
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    def explode(cfg_: Config) -> None:
        raise AssertionError("inpaint without --lama must not build a VRAM manager")

    monkeypatch.setattr("omniscan.gpu.groups.build_vram_manager", explode)
    result = runner.invoke(app, ["ingest", SERIES])
    assert result.exit_code == 0
    result = runner.invoke(app, ["inpaint", SERIES])
    assert result.exit_code == 0
    assert f"{SERIES}/Chapter 1 inpaint: done" in result.output
    assert "inpaint_lama" not in result.output  # without --lama only the flat stage runs

    manager = FakeLamaManager()
    monkeypatch.setattr("omniscan.gpu.groups.build_vram_manager", lambda cfg_: manager)
    result = runner.invoke(app, ["inpaint", SERIES, "--force", "--lama"])
    assert result.exit_code == 0
    assert f"{SERIES}/Chapter 1 inpaint: done" in result.output
    assert f"{SERIES}/Chapter 1 inpaint_lama: done" in result.output  # one line per stage per chapter
    assert manager.released  # the models left VRAM when the command ended
    work = cfg.paths.work_root / SERIES / "Chapter 1"
    assert (work / "inpaint_lama.json").is_file() and (work / "patches_lama.npz").is_file()

    again = runner.invoke(app, ["inpaint", SERIES, "--lama"])
    assert again.exit_code == 0
    assert again.output.count("skipped") == 2  # both stages are up to date

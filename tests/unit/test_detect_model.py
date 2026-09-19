"""Detector wrapper against a fake model/processor (no downloads), plus real-model GPU checks (card C3)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from omniscan.core.config import DetectConfig
from omniscan.detect.model import Detector
from omniscan.gpu.device import resolve_device
from omniscan.models.catalog import ModelEntry
from omniscan.models.store import MARKER_NAME

_CATALOG_SHA = "b" * 64


class FakeModel:
    """Stand-in for RTDetrV2ForObjectDetection: records the batch, exposes a parameter and id2label."""

    def __init__(self, id2label: dict[int, str]) -> None:
        self.config = SimpleNamespace(id2label=id2label)
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.batches: list[torch.Tensor] = []
        self.eval_called = False

    def __call__(self, pixel_values: torch.Tensor) -> dict[str, Any]:
        self.batches.append(pixel_values)
        return {}

    def parameters(self) -> Any:
        return iter((self.param,))

    def to(self, device: torch.device) -> FakeModel:
        self.to_device = device
        return self

    def eval(self) -> FakeModel:
        self.eval_called = True
        return self


class FakeProcessor:
    """Stand-in for the HF image processor: records calls, replays scripted per-image results."""

    def __init__(self, results: list[dict[str, torch.Tensor]] | None = None) -> None:
        self.results: list[dict[str, torch.Tensor]] = results or []
        self.calls: list[dict[str, Any]] = []

    def post_process_object_detection(
        self, outputs: Any, threshold: float, target_sizes: list[tuple[int, int]]
    ) -> list[dict[str, torch.Tensor]]:
        self.calls.append({"threshold": threshold, "target_sizes": target_sizes})
        return self.results


def make_detector(
    results: list[dict[str, torch.Tensor]] | None = None,
    id2label: dict[int, str] | None = None,
) -> tuple[Detector, FakeModel, FakeProcessor]:
    model = FakeModel(id2label or {0: "bubble", 1: "text_bubble", 2: "text_free", 3: "sfx_extra"})
    processor = FakeProcessor(results)
    detector = Detector(model, processor, torch.device("cpu"), threshold=0.3)
    return detector, model, processor


# ---------------------------------------------------------------- fakes


def test_detect_resizes_batches_and_maps_labels() -> None:
    results = [
        {
            "scores": torch.tensor([0.9, 0.5, 0.7]),
            "labels": torch.tensor([0, 3, 1]),
            "boxes": torch.tensor(
                [[-5.0, 10.0, 200.0, 50.0], [0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 50.0, 50.0]]
            ),
        },
        {
            "scores": torch.tensor([0.3]),
            "labels": torch.tensor([2]),
            "boxes": torch.tensor([[0.0, 0.0, 90.0, 64.0]]),
        },
    ]
    detector, model, processor = make_detector(results)
    tiles = [torch.zeros(3, 100, 120, dtype=torch.uint8), torch.ones(3, 64, 80, dtype=torch.uint8)]

    out = detector.detect(tiles)

    assert len(out) == 2
    assert [(d.cls, d.box) for d in out[0]] == [
        ("bubble", (0.0, 10.0, 120.0, 50.0)),  # clipped into the 100x120 tile
        ("text_bubble", (1.0, 1.0, 50.0, 50.0)),  # unknown label 3 dropped, sorted after
    ]
    assert [d.score for d in out[0]] == pytest.approx([0.9, 0.7])
    assert [(d.cls, d.box) for d in out[1]] == [
        ("text_free", (0.0, 0.0, 80.0, 64.0))
    ]  # x1 clipped to tile width
    assert [d.score for d in out[1]] == pytest.approx([0.3])

    batch = model.batches[0]
    assert batch.shape == (2, 3, 640, 640)
    assert batch.dtype == model.param.dtype
    assert batch.min() >= 0.0 and batch.max() <= 1.0  # rescaled, no mean/std normalisation
    call = processor.calls[0]
    assert call["threshold"] == 0.3
    assert call["target_sizes"] == [(100, 120), (64, 80)]  # each tile's own (h, w), in order


def test_detect_sorts_by_score_within_tile() -> None:
    results = [
        {
            "scores": torch.tensor([0.2, 0.9, 0.6]),
            "labels": torch.tensor([1, 0, 2]),
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 20.0, 20.0], [1.0, 1.0, 8.0, 8.0]]),
        }
    ]
    detector, _model, _processor = make_detector(results)
    out = detector.detect([torch.zeros(3, 32, 32, dtype=torch.uint8)])
    assert [d.cls for d in out[0]] == ["bubble", "text_free", "text_bubble"]
    assert [d.score for d in out[0]] == pytest.approx([0.9, 0.6, 0.2])


def test_detect_empty_tiles_never_calls_the_model() -> None:
    detector, model, processor = make_detector()
    assert detector.detect([]) == []
    assert model.batches == [] and processor.calls == []


def test_detect_handles_unknown_and_empty_results() -> None:
    detector, _model, _processor = make_detector(
        [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3]),
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            }
        ]
    )
    assert detector.detect([torch.zeros(3, 32, 32, dtype=torch.uint8)]) == [
        []
    ]  # label 3 is not a known class


# ---------------------------------------------------------------- load


def test_load_passes_revision_and_stays_fp32_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    fake_model = FakeModel({0: "bubble"})
    seen: dict[str, Any] = {}

    def fake_model_from_pretrained(repo: str, revision: str | None = None) -> FakeModel:
        seen["model"] = (repo, revision)
        return fake_model

    def fake_processor_from_pretrained(repo: str, revision: str | None = None) -> FakeProcessor:
        seen["processor"] = (repo, revision)
        return FakeProcessor()

    monkeypatch.setattr(RTDetrV2ForObjectDetection, "from_pretrained", fake_model_from_pretrained)
    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", fake_processor_from_pretrained)

    detector = Detector.load(DetectConfig(repo="org/repo", revision="abc123"), torch.device("cpu"))

    assert seen == {"model": ("org/repo", "abc123"), "processor": ("org/repo", "abc123")}
    assert detector.device == torch.device("cpu")
    assert fake_model.eval_called is True
    assert next(fake_model.parameters()).dtype == torch.float32  # fp32 on CPU: no .half()


# ---------------------------------------------------------------- load from models_dir (card U2b)


def _catalog_entry(model_id: str, repo: str) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        name="Detector",
        kind="vision",
        required=True,
        format="zip",
        size_mb=1,
        license="Apache-2.0",
        description="d",
        mirror_url=f"https://mirror/{model_id}.zip",
        sha256=_CATALOG_SHA,
        bytes=10,
        upstream_repo=repo,
        upstream_revision="rev1",
    )


def _install_fake_zip(models_dir: Path, model_id: str, sha: str) -> None:
    folder = models_dir / model_id
    folder.mkdir(parents=True)
    (folder / "weights.bin").write_text("hello", encoding="utf-8")
    (folder / MARKER_NAME).write_text(json.dumps({"id": model_id, "sha256": sha}), encoding="utf-8")


def test_load_prefers_the_installed_models_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    from omniscan.models import resolve as resolve_module

    repo = "ogkalu/comic-text-and-bubble-detector"
    entry = _catalog_entry("detector-comic-text-bubble", repo)
    _install_fake_zip(tmp_path, entry.id, _CATALOG_SHA)
    monkeypatch.setattr(resolve_module, "load_catalog", lambda: [entry])

    fake_model = FakeModel({0: "bubble"})
    seen: dict[str, Any] = {}

    def fake_model_from(source: str, **kwargs: Any) -> FakeModel:
        seen["model"] = (source, kwargs)
        return fake_model

    def fake_processor_from(source: str, **kwargs: Any) -> FakeProcessor:
        seen["processor"] = (source, kwargs)
        return FakeProcessor()

    monkeypatch.setattr(RTDetrV2ForObjectDetection, "from_pretrained", fake_model_from)
    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", fake_processor_from)

    with caplog.at_level(logging.INFO):
        detector = Detector.load(DetectConfig(repo=repo, revision="pinned"), torch.device("cpu"), tmp_path)

    folder = str(tmp_path / entry.id)
    assert seen == {
        "model": (folder, {"local_files_only": True}),
        "processor": (folder, {"local_files_only": True}),
    }  # the installed folder, no revision even though the config pins one
    assert detector.device == torch.device("cpu")
    assert fake_model.eval_called is True
    assert f"loading {repo} from {folder}" in caplog.text


def test_load_falls_back_to_hub_with_a_warning_when_not_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    from omniscan.models import resolve as resolve_module

    monkeypatch.setattr(resolve_module, "load_catalog", lambda: [])  # nothing installed
    seen: dict[str, Any] = {}

    def fake_model_from(source: str, **kwargs: Any) -> FakeModel:
        seen["model"] = (source, kwargs)
        return FakeModel({0: "bubble"})

    def fake_processor_from(source: str, **kwargs: Any) -> FakeProcessor:
        seen["processor"] = (source, kwargs)
        return FakeProcessor()

    monkeypatch.setattr(RTDetrV2ForObjectDetection, "from_pretrained", fake_model_from)
    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", fake_processor_from)

    with caplog.at_level(logging.WARNING):
        Detector.load(DetectConfig(repo="org/repo", revision="abc123"), torch.device("cpu"), tmp_path)

    assert seen == {
        "model": ("org/repo", {"revision": "abc123"}),
        "processor": ("org/repo", {"revision": "abc123"}),
    }  # exactly today's hub behaviour, with the pinned revision
    warnings = [
        r for r in caplog.records if r.name == "omniscan.detect.model" and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "omniscan models download --required" in warnings[0].message
    assert str(tmp_path) in warnings[0].message


def test_load_without_models_dir_keeps_hub_behaviour_and_no_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    seen: dict[str, Any] = {}

    def fake_model_from(source: str, **kwargs: Any) -> FakeModel:
        seen["model"] = (source, kwargs)
        return FakeModel({0: "bubble"})

    def fake_processor_from(source: str, **kwargs: Any) -> FakeProcessor:
        seen["processor"] = (source, kwargs)
        return FakeProcessor()

    monkeypatch.setattr(RTDetrV2ForObjectDetection, "from_pretrained", fake_model_from)
    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", fake_processor_from)

    with caplog.at_level(logging.INFO):
        Detector.load(DetectConfig(repo="org/repo", revision="abc123"), torch.device("cpu"))

    assert seen == {
        "model": ("org/repo", {"revision": "abc123"}),
        "processor": ("org/repo", {"revision": "abc123"}),
    }
    assert not [r for r in caplog.records if r.name == "omniscan.detect.model"]


# ---------------------------------------------------------------- real model (GPU)


@pytest.mark.gpu
def test_real_detector_detects_on_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    device = resolve_device()
    detector = Detector.load(DetectConfig(), device)
    # fp32 on every device: MIOpen's fp16 conv fails at small batch sizes on this stack
    assert all(p.dtype == torch.float32 for p in detector.model.parameters())

    white = torch.full((3, 1000, 1000), 255, dtype=torch.uint8, device=device)
    rng = np.random.default_rng(0)
    noise = torch.from_numpy(rng.integers(0, 256, size=(3, 1000, 1000), dtype=np.uint8)).to(device)

    for n in (1, 2, 3, 8):  # separate calls; batch 1-3 crashed in fp16
        tiles = ([white] + [noise] * (n - 1))[:n]
        results = detector.detect(tiles)
        assert len(results) == n
        for dets, tile in zip(results, tiles, strict=True):
            for det in dets:
                x0, y0, x1, y1 = det.box
                assert 0.0 <= x0 <= x1 <= 1000 and 0.0 <= y0 <= y1 <= 1000
                assert 0.0 <= det.score <= 1.0
                assert tile.shape[-2] == 1000  # boxes stay in their own tile's pixel space

    # throughput for the report (printed, not asserted)
    batch = [torch.randint(0, 256, (3, 1000, 1000), dtype=torch.uint8, device=device) for _ in range(8)]
    detector.detect(batch)  # warm-up
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    detector.detect(batch)
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    print(f"\ndetect fp32: batch of 8: {8 / dt:.1f} tiles/s ({dt * 1000:.0f} ms)")
    # release the cached blocks so later GPU tests (e.g. the VRAM manager's free-memory check) start clean
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_preprocessing_matches_hf_processor() -> None:
    from transformers import AutoImageProcessor

    rng = np.random.default_rng(0)
    tile = np.full((1280, 1280, 3), 255, dtype=np.uint8)
    for _ in range(12):
        x0, y0 = (int(v) for v in rng.integers(0, 1100, size=2))
        w, h = (int(v) for v in rng.integers(40, 180, size=2))
        tile[y0 : y0 + h, x0 : x0 + w] = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

    processor = AutoImageProcessor.from_pretrained(DetectConfig().repo)
    theirs = processor(images=Image.fromarray(tile), return_tensors="pt")["pixel_values"][0]

    empty = {"scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)}
    model = FakeModel({0: "bubble"})
    detector = Detector(model, FakeProcessor([empty]), torch.device("cpu"), threshold=0.3)
    detector.detect([torch.from_numpy(tile.copy()).permute(2, 0, 1)])
    ours = model.batches[0][0]

    diff = (ours - theirs).abs().mean().item()
    print(f"\npreprocessing mean abs diff: {diff:.5f}")
    assert diff < 0.02

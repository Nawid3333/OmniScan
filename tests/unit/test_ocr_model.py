"""OCR model wrappers against fake model/processor (no downloads), plus real-model GPU checks (card C4a)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from omniscan.core.config import OcrConfig
from omniscan.ocr.lines import LineBox
from omniscan.ocr.model import LineDetector, LineRecognizer


class FakeDetModel:
    """Stand-in for AutoModelForObjectDetection: records the per-tile forward and whether it ran in inference mode."""

    def __init__(self) -> None:
        self.pixel_values: list[torch.Tensor] = []
        self.inference_flags: list[bool] = []

    def __call__(self, pixel_values: torch.Tensor) -> dict[str, Any]:
        self.pixel_values.append(pixel_values)
        self.inference_flags.append(torch.is_inference_mode_enabled())
        return {"last_hidden_state": object()}

    def parameters(self) -> Any:
        return iter((torch.nn.Parameter(torch.zeros(1)),))


class FakeDetProcessor:
    """Stand-in for the HF image processor: records the image tensor, replays scripted polygons."""

    def __init__(self, results: list[list[tuple[float, list[list[float]]]]]) -> None:
        self.results = results  # one entry per call: (score, corner points) pairs
        self.images: list[torch.Tensor] = []
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, images: torch.Tensor | None = None, return_tensors: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        assert images is not None and return_tensors == "pt"
        self.images.append(images)
        return {
            "pixel_values": images.unsqueeze(0),
            "target_sizes": torch.tensor([[int(images.shape[-2]), int(images.shape[-1])]]),
        }

    def post_process_object_detection(
        self,
        outputs: Any,
        *,
        threshold: float,
        box_threshold: float,
        unclip_ratio: float,
        min_size: int,
        target_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        self.calls.append(
            {
                "threshold": threshold,
                "box_threshold": box_threshold,
                "unclip_ratio": unclip_ratio,
                "min_size": min_size,
                "target_sizes": target_sizes.tolist(),
            }
        )
        pairs = self.results[len(self.calls) - 1]
        return [
            {
                "scores": torch.tensor([score for score, _ in pairs]),
                "labels": torch.zeros(len(pairs), dtype=torch.long),
                "boxes": torch.tensor([points for _, points in pairs]),
            }
        ]


def make_detector(
    results: list[list[tuple[float, list[list[float]]]]] | None = None,
) -> tuple[LineDetector, FakeDetModel, FakeDetProcessor]:
    model = FakeDetModel()
    processor = FakeDetProcessor(results or [])
    detector = LineDetector(
        model,
        processor,
        torch.device("cpu"),
        det_threshold=0.3,
        box_threshold=0.6,
        unclip_ratio=1.5,
        min_size=3,
    )
    return detector, model, processor


# ---------------------------------------------------------------- detector (test 6)


def test_detect_turns_polygons_into_clipped_line_boxes() -> None:
    results = [
        [
            (0.9, [[-5.0, 10.0], [200.0, 10.0], [200.0, 50.0], [-5.0, 50.0]]),
            (0.5, [[0.0, 0.0], [2.0, 0.0], [2.0, 10.0], [0.0, 10.0]]),  # clips to 2 px wide: dropped
        ],
        [(0.7, [[1.0, 1.0], [50.0, 1.0], [50.0, 40.0], [1.0, 40.0]])],
    ]
    detector, model, processor = make_detector(results)
    tiles = [torch.zeros(3, 100, 120, dtype=torch.uint8), torch.ones(3, 64, 80, dtype=torch.uint8)]

    out = detector.detect(tiles)

    # scores travel through a float32 tensor, so compare them approximately
    assert [[(lb.box, lb.score) for lb in tile] for tile in out] == [
        [((0.0, 10.0, 120.0, 50.0), pytest.approx(0.9))],
        [((1.0, 1.0, 50.0, 40.0), pytest.approx(0.7))],
    ]
    assert [tuple(t.shape) for t in model.pixel_values] == [
        (1, 3, 100, 120),
        (1, 3, 64, 80),
    ]  # one forward per tile
    assert [tuple(t.shape) for t in processor.images] == [(3, 100, 120), (3, 64, 80)]
    assert all(t.dtype == torch.uint8 and t.device.type == "cpu" for t in processor.images)
    assert processor.images[0] is tiles[0]  # the tile itself is handed over, never a copy
    assert processor.calls[0] == {
        "threshold": 0.3,
        "box_threshold": 0.6,
        "unclip_ratio": 1.5,
        "min_size": 3,
        "target_sizes": [[100, 120]],
    }
    assert processor.calls[1]["target_sizes"] == [[64, 80]]
    assert all(model.inference_flags)  # every forward ran inside inference_mode


def test_detect_empty_tiles_never_calls_the_model() -> None:
    detector, _model, processor = make_detector()
    assert detector.detect([]) == []
    assert processor.images == [] and processor.calls == []


class FakeRecModel:
    """Stand-in for AutoModelForTextRecognition: records the batch."""

    def __init__(self) -> None:
        self.batches: list[dict[str, Any]] = []
        self.inference_flags: list[bool] = []

    def __call__(self, **inputs: Any) -> dict[str, Any]:
        self.batches.append(inputs)
        self.inference_flags.append(torch.is_inference_mode_enabled())
        return {}


class FakeRecProcessor:
    """Stand-in for the HF processor: records crop widths per chunk, maps a crop's width to its reading."""

    def __init__(self) -> None:
        self.chunks: list[list[int]] = []

    def __call__(
        self, images: list[torch.Tensor] | None = None, return_tensors: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        assert images is not None and return_tensors == "pt"
        self.chunks.append([int(image.shape[-1]) for image in images])
        return {"pixel_values": images[0]}

    def post_process_text_recognition(self, outputs: Any) -> list[dict[str, Any]]:
        return [{"text": f"w{width}", "score": width / 300.0} for width in self.chunks[-1]]


def make_recognizer() -> tuple[LineRecognizer, FakeRecModel, FakeRecProcessor]:
    model = FakeRecModel()
    processor = FakeRecProcessor()
    recognizer = LineRecognizer(model, processor, torch.device("cpu"), batch_size=2)
    return recognizer, model, processor


# ---------------------------------------------------------------- recognizer (test 8)


def test_read_sorts_by_width_chunks_and_restores_the_order() -> None:
    recognizer, model, processor = make_recognizer()
    crops = [torch.zeros(3, 48, w, dtype=torch.uint8) for w in (50, 200, 20, 120, 80)]

    out = recognizer.read(crops)

    assert [text for text, _score in out] == ["w50", "w200", "w20", "w120", "w80"]
    assert [score for _text, score in out] == [w / 300.0 for w in (50, 200, 20, 120, 80)]
    assert all(isinstance(score, float) for _text, score in out)
    assert processor.chunks == [[20, 50], [80, 120], [200]]  # 3 calls, ascending width, 2 crops each
    assert all(model.inference_flags)


def test_read_empty_never_calls_the_model() -> None:
    recognizer, model, processor = make_recognizer()
    assert recognizer.read([]) == []
    assert processor.chunks == [] and model.batches == []


# ---------------------------------------------------------------- load (test 7)


class FakeLoadModel:
    """A from_pretrained result: `.to(device)` + `.eval()` chainable, one fp32 parameter."""

    def __init__(self) -> None:
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.to_device: torch.device | None = None
        self.eval_called = False

    def to(self, device: torch.device) -> FakeLoadModel:
        self.to_device = device
        return self

    def eval(self) -> FakeLoadModel:
        self.eval_called = True
        return self

    def parameters(self) -> Any:
        return iter((self.param,))


def test_line_detector_load(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    fake_model = FakeLoadModel()
    seen: dict[str, Any] = {}

    def fake_model_from(repo: str, revision: str | None = None) -> FakeLoadModel:
        seen["model"] = (repo, revision)
        return fake_model

    def fake_processor_from(repo: str, revision: str | None = None) -> FakeDetProcessor:
        seen["processor"] = (repo, revision)
        return FakeDetProcessor([])

    monkeypatch.setattr(AutoModelForObjectDetection, "from_pretrained", fake_model_from)
    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", fake_processor_from)

    detector = LineDetector.load(
        OcrConfig(det_repo="org/det", rec_repo="org/rec", det_revision="abc123"), torch.device("cpu")
    )

    assert seen == {"model": ("org/det", "abc123"), "processor": ("org/det", "abc123")}
    assert detector.device == torch.device("cpu")
    assert fake_model.eval_called is True
    assert next(fake_model.parameters()).dtype == torch.float32  # fp32 on CPU: no .half()


def test_line_recognizer_load(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import AutoImageProcessor, AutoModelForTextRecognition

    fake_model = FakeLoadModel()
    seen: dict[str, Any] = {}

    def fake_model_from(repo: str, revision: str | None = None) -> FakeLoadModel:
        seen["model"] = (repo, revision)
        return fake_model

    def fake_processor_from(repo: str, revision: str | None = None) -> FakeRecProcessor:
        seen["processor"] = (repo, revision)
        return FakeRecProcessor()

    monkeypatch.setattr(AutoModelForTextRecognition, "from_pretrained", fake_model_from)
    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", fake_processor_from)

    LineRecognizer.load(OcrConfig(rec_repo="org/rec", rec_revision="def456"), torch.device("cpu"))

    assert seen == {"model": ("org/rec", "def456"), "processor": ("org/rec", "def456")}
    assert fake_model.to_device == torch.device("cpu")
    assert fake_model.eval_called is True
    assert next(fake_model.parameters()).dtype == torch.float32

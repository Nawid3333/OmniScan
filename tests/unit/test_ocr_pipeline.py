"""read_regions against fake wrappers (CPU strip, no downloads) plus the real-model GPU checks (card C4a)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest
import torch

from omniscan.core.config import OcrConfig
from omniscan.core.schemas import BBox, Region
from omniscan.gpu.device import resolve_device
from omniscan.ocr.lines import LineBox
from omniscan.ocr.model import LineDetector, LineRecognizer
from omniscan.ocr.pipeline import read_region_crops, read_regions


def region(rid: str, box: tuple[int, int, int, int]) -> Region:
    return Region(
        id=rid, slice_index=0, kind="bubble_text", bbox=BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3])
    )


class FakeDetector:
    """Scripted line detector: per cumulative tile index a list of tile-pixel (x0, y0, x1, y1, score)."""

    def __init__(
        self, script: dict[int, list[tuple[float, float, float, float, float]]] | None = None
    ) -> None:
        self.script = script or {}
        self.shapes: list[list[tuple[int, int]]] = []
        self.next_index = 0

    def detect(self, tiles: list[torch.Tensor]) -> list[list[LineBox]]:
        shapes = []
        for tile in tiles:
            shapes.append((int(tile.shape[-2]), int(tile.shape[-1])))
            self.next_index += 1
        self.shapes.append(shapes)
        start = self.next_index - len(tiles)
        return [
            [
                LineBox(box=(x0, y0, x1, y1), score=score)
                for x0, y0, x1, y1, score in self.script.get(start + i, [])
            ]
            for i in range(len(tiles))
        ]


class FakeRecognizer:
    """Scripted recognizer: one scripted reading per cumulative crop index, recording each crop's shape."""

    def __init__(self, script: dict[int, tuple[str, float]] | None = None) -> None:
        self.script = script or {}
        self.shapes: list[tuple[int, int]] = []
        self.next_index = 0

    def read(self, crops: list[torch.Tensor]) -> list[tuple[str, float]]:
        readings: list[tuple[str, float]] = []
        for crop in crops:
            self.shapes.append((int(crop.shape[-2]), int(crop.shape[-1])))
            readings.append(self.script.get(self.next_index, ("", 0.0)))
            self.next_index += 1
        return readings


CFG = OcrConfig(tile_px=200, overlap=0.5)


def _read(
    strip: torch.Tensor,
    regions: Sequence[Region],
    detector: FakeDetector,
    recognizer: FakeRecognizer,
    cfg: OcrConfig = CFG,
) -> tuple[list[Region], dict[str, float]]:
    """read_regions with the fakes cast to the wrapper types (they duck-type the wrappers)."""
    return read_regions(
        strip,
        regions,
        cast("LineDetector", detector),
        cast("LineRecognizer", recognizer),
        cfg,
        direction="ltr",
        engine="e",
    )


# ---------------------------------------------------------------- CPU fakes (tests 9-11)


def test_only_tiles_touching_a_region_are_detected() -> None:
    strip = torch.zeros(3, 600, 200, dtype=torch.uint8)
    regions = [region("r0001", (10, 10, 190, 60)), region("r0002", (10, 510, 190, 560))]
    detector = FakeDetector()
    recognizer = FakeRecognizer()

    _out, _metrics = _read(strip, regions, detector, recognizer)

    # a 200x600 strip plans tiles at y 0, 100, 200, 300, 400; only tile 0 (0-200) and tile 4 (400-600)
    # overlap the regions
    assert detector.shapes == [[(200, 200)] * 2]
    assert recognizer.shapes == []


def test_line_boxes_are_shifted_to_strip_pixels_and_deduplicated() -> None:
    strip = torch.zeros(3, 300, 200, dtype=torch.uint8)
    regions = [region("r0001", (10, 140, 190, 270))]  # both 200-px tiles touch it
    detector = FakeDetector(
        {
            0: [(20.0, 150.0, 100.0, 170.0, 0.9)],  # tile y 0-200: the line sits at strip (20, 150..170)
            1: [(20.0, 50.0, 100.0, 70.0, 0.8)],  # tile y 100-300: the same line, in tile pixels
        }
    )
    recognizer = FakeRecognizer({0: ("한 줄", 0.95)})

    out, metrics = _read(strip, regions, detector, recognizer)

    assert metrics["lines"] == 1.0  # the duplicate from the overlapping tile was merged away
    assert recognizer.shapes == [(26, 86)]  # one reading request, for the padded crop (17, 147, 103, 173)
    assert out[0].text == "한 줄" and out[0].lines[0].bbox == BBox(x0=20, y0=150, x1=100, y1=170)


def test_read_regions_end_to_end_with_fakes() -> None:
    strip = torch.zeros(3, 300, 200, dtype=torch.uint8)
    regions = [region("r0001", (10, 20, 190, 70)), region("r0002", (10, 250, 190, 300))]
    detector = FakeDetector(
        {
            0: [(20.0, 30.0, 100.0, 50.0, 0.9), (20.0, 55.0, 100.0, 75.0, 0.85)],
            1: [(20.0, 160.0, 100.0, 180.0, 0.8)],  # strip pixels (20, 260, 100, 280)
        }
    )
    recognizer = FakeRecognizer({0: ("안녕", 0.99), 1: ("하세요", 0.90), 2: ("잘가", 0.5)})

    out, metrics = _read(strip, regions, detector, recognizer)

    assert metrics == {
        "tiles": 2.0,
        "lines": 3.0,
        "orphan_lines": 0.0,
        "regions": 2.0,
        "regions_empty": 0.0,
        "regions_low_conf": 1.0,  # r0002 at 0.5 < ocr.low_conf 0.85
    }
    assert [r.text for r in out] == ["안녕\n하세요", "잘가"]
    assert [r.confidence for r in out] == [0.9, 0.5]
    assert out[0].lines[0].engine == "e"
    assert [r.lines[0].bbox for r in out] == [
        BBox(x0=20, y0=30, x1=100, y1=50),
        BBox(x0=20, y0=260, x1=100, y1=280),
    ]
    # the crops are the padded, clamped strip views: (17, 27, 103, 53), (17, 52, 103, 78), (17, 257, 103, 283)
    assert recognizer.shapes == [(26, 86), (26, 86), (26, 86)]
    assert recognizer.next_index == 3


def test_read_regions_without_regions_runs_no_model() -> None:
    strip = torch.zeros(3, 300, 200, dtype=torch.uint8)
    detector = FakeDetector()
    recognizer = FakeRecognizer()

    out, metrics = _read(strip, [], detector, recognizer)

    assert out == []
    assert detector.shapes == [] and recognizer.shapes == []
    assert metrics["tiles"] == 0.0 and metrics["regions"] == 0.0


# ---------------------------------------------------------------- read_region_crops (card O1b, test 6)


class FakeCropReader:
    """Scripted crop reader: records the crops (shape, storage offset) and hands out readings in order."""

    def __init__(self, readings: list[tuple[str, float]]) -> None:
        self.readings = readings
        self.shapes: list[tuple[int, int]] = []
        self.data_ptrs: list[int] = []

    def read(self, crops: Sequence[torch.Tensor]) -> list[tuple[str, float]]:
        for crop in crops:
            self.shapes.append((int(crop.shape[-2]), int(crop.shape[-1])))
            self.data_ptrs.append(crop.data_ptr())
        assert len(self.readings) == len(crops)
        return list(self.readings)


CROP_CFG = OcrConfig(crop_pad_px=2)  # crop_batch_size only matters inside a real reader


def test_read_region_crops_views_readings_and_metrics() -> None:
    strip = torch.zeros(3, 100, 200, dtype=torch.uint8)
    regions = [
        region("r0001", (10, 20, 60, 80)),
        region("r0002", (0, 0, 50, 50)),  # touches the top-left corner: clamped
        region("r0003", (190, 95, 200, 100)),  # touches the bottom-right corner: clamped
    ]
    reader = FakeCropReader([("あいう", 0.9), ("", 0.8), ("、", 0.5)])

    out, metrics = read_region_crops(strip, regions, reader, CROP_CFG, engine="ocr-rec-manga-ocr-2025")

    # crops are the padded, clamped strip views: (8, 18, 62, 82), (0, 0, 52, 52), (188, 93, 200, 100)
    assert reader.shapes == [(64, 54), (52, 52), (7, 12)]
    base, end = strip.data_ptr(), strip.data_ptr() + strip.numel()
    assert all(base <= ptr < end for ptr in reader.data_ptrs)  # views of the strip, never copies
    assert [(r.text, r.confidence) for r in out] == [("あいう", 0.9), ("", 0.0), ("、", 0.5)]
    assert out[0].lines[0].engine == "ocr-rec-manga-ocr-2025"
    assert out[0].lines[0].bbox == BBox(x0=10, y0=20, x1=60, y1=80)  # the region's own bbox
    assert out[1].lines == [] and out[1].text == ""  # an empty reading leaves nothing behind
    assert metrics == {
        "tiles": 0.0,
        "lines": 3.0,
        "orphan_lines": 0.0,
        "regions": 3.0,
        "regions_empty": 1.0,
        "regions_low_conf": 1.0,  # r0003 at 0.5 < ocr.low_conf 0.85
    }


def test_read_region_crops_without_regions_never_reads() -> None:
    strip = torch.zeros(3, 100, 200, dtype=torch.uint8)
    reader = FakeCropReader([])

    out, metrics = read_region_crops(strip, [], reader, CROP_CFG, engine="e")

    assert out == [] and reader.shapes == []
    assert metrics == {
        "tiles": 0.0,
        "lines": 0.0,
        "orphan_lines": 0.0,
        "regions": 0.0,
        "regions_empty": 0.0,
        "regions_low_conf": 0.0,
    }


# ---------------------------------------------------------------- real models (GPU; tests 15-16)


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        prev = row[0]
        row[0] = i + 1
        for j, cb in enumerate(b):
            cur = row[j + 1]
            row[j + 1] = min(row[j] + 1, cur + 1, prev + (ca != cb))
            prev = cur
    return row[len(b)]


def _cer(truth: str, read: str) -> float:
    """Character error rate of `read` against `truth`, both with all whitespace removed."""
    a = "".join(truth.split())
    b = "".join(read.split())
    return _levenshtein(a, b) / len(a) if a else (1.0 if b else 0.0)


def _page_strip(image: Any, device: torch.device) -> torch.Tensor:
    """A fixture page as a uint8 [3, H, W] tensor on `device`."""
    return torch.from_numpy(np.asarray(image, dtype=np.uint8)).permute(2, 0, 1).contiguous().to(device)


def _detection_input_regions(truth: Sequence[Region]) -> list[Region]:
    """The truth regions as detection would hand them to OCR: lines, text and confidence blanked."""
    return [r.model_copy(update={"lines": [], "text": "", "confidence": 0.0}) for r in truth]


def _report(
    label: str, truth: Sequence[Region], out: Sequence[Region], metrics: dict[str, float], wall: float
) -> float:
    """Print the run's numbers (printed, never asserted) and return the mean CER."""
    errors = [_cer(t.text, o.text) for t, o in zip(truth, out, strict=True)]
    mean_cer = sum(errors) / len(errors)
    non_empty = sum(1 for r in out if r.lines)
    print(
        f"{label}: non-empty {non_empty}/{len(out)}, mean CER {mean_cer:.3f}, {wall:.2f}s, "
        f"metrics: {{{', '.join(f'{k}: {v:.3f}' for k, v in metrics.items())}}}"
    )
    return mean_cer


@pytest.mark.gpu
def test_real_ocr_reads_a_korean_page() -> None:
    from tests.fixtures.korean_pages import make_korean_page, to_regions_artifact

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    device = resolve_device()
    detector = LineDetector.load(OcrConfig(), device)
    recognizer = LineRecognizer.load(OcrConfig(), device)
    assert all(p.dtype == torch.float32 for p in detector.model.parameters())  # fp32 on the GPU too
    assert all(p.dtype == torch.float32 for p in recognizer.model.parameters())

    page = make_korean_page(seed=1, n_bubbles=4, n_free=1)
    truth = to_regions_artifact(page).regions
    strip = _page_strip(page.image, device)
    t0 = time.perf_counter()
    out, _metrics = read_regions(
        strip,
        _detection_input_regions(truth),
        detector,
        recognizer,
        OcrConfig(),
        direction="ltr",
        engine="test",
    )
    mean_cer = _report("NanumGothic", truth, out, _metrics, time.perf_counter() - t0)
    assert sum(1 for r in out if r.lines) >= 0.9 * len(out)
    assert mean_cer <= 0.15

    gaegu = make_korean_page(seed=1, n_bubbles=4, n_free=1, font="Gaegu-Regular.ttf")
    gaegu_truth = to_regions_artifact(gaegu).regions
    gaegu_strip = _page_strip(gaegu.image, device)
    t0 = time.perf_counter()
    gaegu_out, gaegu_metrics = read_regions(
        gaegu_strip,
        _detection_input_regions(gaegu_truth),
        detector,
        recognizer,
        OcrConfig(),
        direction="ltr",
        engine="test",
    )
    _report("Gaegu", gaegu_truth, gaegu_out, gaegu_metrics, time.perf_counter() - t0)
    torch.cuda.empty_cache()  # release the cached blocks so later GPU tests start clean


@pytest.mark.gpu
def test_real_ocr_robustness() -> None:
    from tests.fixtures.korean_pages import make_korean_page, to_regions_artifact

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    device = resolve_device()
    detector = LineDetector.load(OcrConfig(), device)
    recognizer = LineRecognizer.load(OcrConfig(), device)

    empty = make_korean_page(seed=1, n_bubbles=0, n_free=0)
    out, _metrics = read_regions(
        _page_strip(empty.image, device), [], detector, recognizer, OcrConfig(), direction="ltr", engine="e"
    )
    assert out == []

    page = make_korean_page(seed=1, n_bubbles=4, n_free=1)
    truth = to_regions_artifact(page).regions
    _out2, metrics2 = read_regions(
        _page_strip(page.clean, device),
        _detection_input_regions(truth),
        detector,
        recognizer,
        OcrConfig(),
        direction="ltr",
        engine="e",
    )
    assert metrics2["regions_empty"] == metrics2["regions"]  # no text drawn: nothing to read
    torch.cuda.empty_cache()

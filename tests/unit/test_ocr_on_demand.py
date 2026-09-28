"""read_region: one region read again through the ocr stage's own reading functions, with fake models (CPU)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from omniscan.core.config import Config, GpuConfig, OcrConfig, PathsConfig
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import BBox, IngestArtifact, Region, RegionsArtifact, SourceFile
from omniscan.edits import store
from omniscan.ocr.lines import LineBox
from omniscan.ocr.on_demand import read_region


class CropReader:
    """Records the crops it is handed and reads a fixed text."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.shapes: list[tuple[int, ...]] = []

    def read(self, crops: Sequence[torch.Tensor]) -> list[tuple[str, float]]:
        self.shapes += [tuple(crop.shape) for crop in crops]
        return [(self.text, 0.87) for _ in crops]


class LineDetector:
    """One line over the middle of every tile it sees."""

    def __init__(self) -> None:
        self.tiles: list[tuple[int, ...]] = []

    def detect(self, tiles: list[torch.Tensor]) -> list[list[LineBox]]:
        self.tiles += [tuple(tile.shape) for tile in tiles]
        return [
            [LineBox(box=(20.0, 20.0, float(t.shape[-1]) - 20, float(t.shape[-2]) - 20), score=0.95)]
            for t in tiles
        ]


class LineRecognizer:
    def read(self, crops: list[torch.Tensor]) -> list[tuple[str, float]]:
        return [("안녕하세요", 0.93) for _ in crops]


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Config, ChapterPaths]:
    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        ),
        gpu=GpuConfig(device="cpu"),
    )
    paths = SeriesPaths.from_config(cfg, "S").chapter("Chapter 1")
    paths.raw_dir.mkdir(parents=True)
    Image.fromarray(np.full((400, 300, 3), 200, np.uint8)).save(paths.raw_dir / "001.png")
    IngestArtifact(
        series="S",
        chapter="Chapter 1",
        strip_width=300,
        strip_height=400,
        files=[SourceFile(index=0, name="001.png", sha256="0" * 64, width=300, height=400, y0=0, y1=400)],
    ).save(paths.artifact("ingest.json"))
    RegionsArtifact(
        regions=[
            Region(id="m0001", slice_index=0, kind="bubble_text", bbox=BBox(x0=100, y0=150, x1=220, y1=210))
        ]
    ).save(paths.artifact("ocr.json"))
    return cfg, paths


def test_a_crop_reader_reads_the_region_with_its_padding(paths: tuple[Config, ChapterPaths]) -> None:
    cfg, chapter = paths
    cfg = cfg.model_copy(update={"ocr": OcrConfig(engine="manga_ocr", crop_pad_px=6)})
    reader = CropReader("잘 가")
    reading = read_region(chapter, "m0001", cfg, {"reader": reader})
    assert (reading.region_id, reading.text, reading.confidence) == ("m0001", "잘 가", 0.87)
    assert reader.shapes == [(3, 72, 132)]  # the 120 x 60 box plus the engine's 6 px padding
    assert store.current_regions(chapter)[0].text == ""  # nothing written


def test_ppocr_finds_the_lines_inside_the_region(paths: tuple[Config, ChapterPaths]) -> None:
    cfg, chapter = paths
    cfg = cfg.model_copy(update={"ocr": OcrConfig(engine="ppocr", tile_px=512)})
    detector = LineDetector()
    reading = read_region(chapter, "m0001", cfg, {"line_detector": detector, "recognizer": LineRecognizer()})
    assert reading.text == "안녕하세요" and reading.confidence == pytest.approx(0.93)
    assert detector.tiles == [(3, 108, 168)]  # only the crop around the box (plus 24 px) is searched

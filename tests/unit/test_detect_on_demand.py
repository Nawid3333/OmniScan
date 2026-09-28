"""find_on_page: the detect stage's own functions on one page, then the OCR on the boxes no region covers, with
fake models (CPU torch, so this runs in CI)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from omniscan.core.config import Config, GpuConfig, OcrConfig, PathsConfig
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import (
    BBox,
    IngestArtifact,
    Region,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
    SourceFile,
)
from omniscan.detect.model import RawDet
from omniscan.detect.on_demand import find_on_page

COVERED = RawDet("text_free", 0.8, (50.0, 52.0, 150.0, 118.0))  # r0001's text: dropped
MISSED = RawDet("text_free", 0.6, (60.0, 200.0, 200.0, 250.0))


class Detector:
    """Finds COVERED and MISSED in the first tile it is shown, nothing elsewhere; records the threshold it ran at."""

    def __init__(self, found: Sequence[RawDet] = (COVERED, MISSED)) -> None:
        self.threshold = 0.3
        self.found = list(found)
        self.runs: list[tuple[int, float]] = []

    def detect(self, tiles: Sequence[torch.Tensor]) -> list[list[RawDet]]:
        first = not self.runs
        self.runs.append((len(tiles), self.threshold))
        return [list(self.found) if first and i == 0 else [] for i in range(len(tiles))]


class CropReader:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def read(self, crops: Sequence[torch.Tensor]) -> list[tuple[str, float]]:
        self.shapes += [tuple(crop.shape) for crop in crops]
        return [("쾅", 0.87) for _ in crops]


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Config, ChapterPaths]:
    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        ),
        gpu=GpuConfig(device="cpu"),
        ocr=OcrConfig(engine="manga_ocr", crop_pad_px=6),
    )
    paths = SeriesPaths.from_config(cfg, "S").chapter("Chapter 1")
    paths.raw_dir.mkdir(parents=True)
    for i in range(2):
        Image.fromarray(np.full((400, 300, 3), 200 - 50 * i, np.uint8)).save(
            paths.raw_dir / f"{i + 1:03d}.png"
        )
    IngestArtifact(
        series="S",
        chapter="Chapter 1",
        strip_width=300,
        strip_height=800,
        files=[
            SourceFile(
                index=i,
                name=f"{i + 1:03d}.png",
                sha256="0" * 64,
                width=300,
                height=400,
                y0=400 * i,
                y1=400 * (i + 1),
            )
            for i in range(2)
        ],
    ).save(paths.artifact("ingest.json"))
    SlicesArtifact(
        strip_width=300,
        strip_height=800,
        bands=[],
        slices=[Slice(index=i, y0=400 * i, y1=400 * (i + 1)) for i in range(2)],
    ).save(paths.artifact("slices.json"))
    region = Region(
        id="r0001", slice_index=0, kind="bubble_text", bbox=BBox(x0=50, y0=50, x1=150, y1=120), text="안녕"
    )
    RegionsArtifact(regions=[region]).save(paths.artifact("ocr.json"))
    return cfg, paths


def test_only_the_missed_box_comes_back_read_by_the_ocr(setup: tuple[Config, ChapterPaths]) -> None:
    cfg, paths = setup
    detector, reader = Detector(), CropReader()
    found = find_on_page(paths, 0, cfg, {"detector": detector, "reader": reader}, threshold=0.15)
    assert [(f.kind, f.bbox, f.text, f.confidence) for f in found] == [
        ("free_text", BBox(x0=60, y0=200, x1=200, y1=250), "쾅", 0.87)
    ]
    assert found[0].score == pytest.approx(0.6)
    assert detector.runs == [(2, 0.15)]  # the page's two 300 px tiles, at the threshold asked for
    assert detector.threshold == 0.3  # and the configured one back afterwards
    assert reader.shapes == [(3, 62, 152)]  # only the missed box, with the crop padding


def test_a_later_page_is_searched_in_its_own_rows(setup: tuple[Config, ChapterPaths]) -> None:
    cfg, paths = setup
    found = find_on_page(paths, 1, cfg, {"detector": Detector(), "reader": CropReader()})
    assert [f.bbox for f in found] == [  # r0001 is on page 0, so neither box is covered here
        BBox(x0=50, y0=452, x1=150, y1=518),
        BBox(x0=60, y0=600, x1=200, y1=650),
    ]


def test_nothing_new_reads_nothing(setup: tuple[Config, ChapterPaths]) -> None:
    cfg, paths = setup
    reader = CropReader()
    assert find_on_page(paths, 0, cfg, {"detector": Detector([COVERED]), "reader": reader}) == []
    assert reader.shapes == []

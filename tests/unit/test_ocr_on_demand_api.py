"""Reading one region again: the crop around it, the web route and `omniscan edit ocr`, with the OCR itself faked
(torch-free; tests/unit/test_ocr_on_demand.py runs the real reading path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from typer.testing import CliRunner

import omniscan.edits.cli as edit_cli
from omniscan.core.config import Config, PathsConfig
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
from omniscan.edits import store
from omniscan.ocr.on_demand import Reading, crop_around, in_crop, region_crop
from omniscan.web.app import create_app

WIDTH, HEIGHT = 300, 500


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )


@pytest.fixture
def paths(cfg: Config) -> ChapterPaths:
    """One raw page (a gradient, so a crop shows where it came from) with two read regions."""
    paths = SeriesPaths.from_config(cfg, "S").chapter("Chapter 1")
    paths.raw_dir.mkdir(parents=True)
    pixels = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    pixels[:, :, 0] = np.linspace(0, 255, HEIGHT, dtype=np.uint8)[:, None]  # red grows downwards
    Image.fromarray(pixels).save(paths.raw_dir / "001.png")
    IngestArtifact(
        series="S",
        chapter="Chapter 1",
        strip_width=WIDTH,
        strip_height=HEIGHT,
        files=[
            SourceFile(index=0, name="001.png", sha256="0" * 64, width=WIDTH, height=HEIGHT, y0=0, y1=HEIGHT)
        ],
    ).save(paths.artifact("ingest.json"))
    SlicesArtifact(
        strip_width=WIDTH, strip_height=HEIGHT, bands=[], slices=[Slice(index=0, y0=0, y1=HEIGHT)]
    ).save(paths.artifact("slices.json"))
    RegionsArtifact(
        regions=[
            Region(
                id="r0001",
                slice_index=0,
                kind="bubble_text",
                bbox=BBox(x0=10, y0=10, x1=110, y1=60),
                text="안녕",
            ),
            Region(
                id="r0002",
                slice_index=0,
                kind="bubble_text",
                bbox=BBox(x0=100, y0=300, x1=200, y1=350),
                text="잘가",
            ),
        ]
    ).save(paths.artifact("ocr.json"))
    return paths


def test_the_crop_around_a_region_stays_inside_the_strip(paths: ChapterPaths) -> None:
    assert crop_around(BBox(x0=10, y0=10, x1=110, y1=60), WIDTH, HEIGHT) == BBox(x0=0, y0=0, x1=134, y1=84)
    assert crop_around(BBox(x0=250, y0=460, x1=300, y1=500), WIDTH, HEIGHT, margin=10) == BBox(
        x0=240, y0=450, x1=300, y1=500
    )
    region, crop, pixels = region_crop(paths, "r0002")
    assert crop == BBox(x0=76, y0=276, x1=224, y1=374) and pixels.shape == (98, 148, 3)
    assert abs(int(pixels[0, 0, 0]) - round(276 * 255 / (HEIGHT - 1))) <= 2  # the right rows of the page
    local = in_crop(region, crop)
    assert local.bbox == BBox(x0=24, y0=24, x1=124, y1=74) and local.text == "" and not local.lines
    with pytest.raises(store.EditNotFoundError):
        region_crop(paths, "r0009")
    paths.artifact("ingest.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"ingest\.json missing"):
        region_crop(paths, "r0001")


def fake_reader(text: str) -> object:
    """A reader that reads `text` in any region (and records what it was asked)."""
    asked: list[str] = []

    def read(paths: ChapterPaths, region_id: str, cfg: Config) -> Reading:
        if region_id not in {r.id for r in store.current_regions(paths)}:
            raise store.EditNotFoundError(f"region {region_id!r} not found in ocr.json")
        asked.append(region_id)
        return Reading(region_id=region_id, text=text, confidence=0.91, engine="PaddleOCR-VL-1.5")

    read.asked = asked  # type: ignore[attr-defined]
    return read


def failing_reader(error: Exception) -> object:
    def read(paths: ChapterPaths, region_id: str, cfg: Config) -> Reading:
        raise error

    return read


def test_reading_a_region_again_over_the_web_api(paths: ChapterPaths, cfg: Config) -> None:
    url = "/api/series/S/chapters/Chapter%201/regions"
    client = TestClient(create_app(cfg, ocr_reader=fake_reader("안녕하세요!")))  # type: ignore[arg-type]
    shown = client.post(f"{url}/r0001/ocr", json={})
    assert shown.status_code == 200 and shown.json() == {
        "region_id": "r0001",
        "text": "안녕하세요!",
        "confidence": 0.91,
        "engine": "PaddleOCR-VL-1.5",
        "applied": False,
    }
    assert store.current_regions(paths)[0].text == "안녕"  # a suggestion only
    kept = client.post(f"{url}/r0001/ocr", json={"apply": True})
    assert kept.json()["applied"] and store.current_regions(paths)[0].text == "안녕하세요!"
    assert store.load_edits(paths).regions[0].auto_text == "안녕"  # a hand edit, learned from like any
    assert client.post(f"{url}/r0009/ocr", json={}).status_code == 404
    empty = TestClient(create_app(cfg, ocr_reader=fake_reader(" ")))  # type: ignore[arg-type]
    assert not empty.post(f"{url}/r0002/ocr", json={"apply": True}).json()["applied"]
    assert store.current_regions(paths)[1].text == "잘가"  # nothing read: the text stays
    for error in (RuntimeError("HIP out of memory"), ImportError("No module named 'torch'")):
        down = TestClient(create_app(cfg, ocr_reader=failing_reader(error)))  # type: ignore[arg-type]
        failed = down.post(f"{url}/r0001/ocr", json={})
        assert failed.status_code == 503 and "the OCR could not run" in failed.json()["detail"]


def test_edit_ocr_on_the_command_line(
    paths: ChapterPaths, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(edit_cli, "get_config", lambda: cfg)
    monkeypatch.setattr(edit_cli, "read_again", fake_reader("잘 가요"))
    runner = CliRunner()
    shown = runner.invoke(edit_cli.edit_app, ["ocr", "S", "Chapter 1", "r0002"])
    assert shown.output == "edit: r0002 reads (PaddleOCR-VL-1.5, 0.91): 잘 가요\n"
    assert store.current_regions(paths)[1].text == "잘가"
    kept = runner.invoke(edit_cli.edit_app, ["ocr", "S", "Chapter 1", "r0002", "--apply"])
    assert (
        kept.output.endswith("edit: r0002 source text saved\n")
        and store.current_regions(paths)[1].text == "잘 가요"
    )
    unknown = runner.invoke(edit_cli.edit_app, ["ocr", "S", "Chapter 1", "r0009"])
    assert unknown.exit_code == 2 and "r0009" in unknown.output
    monkeypatch.setattr(edit_cli, "read_again", fake_reader(""))
    nothing = runner.invoke(edit_cli.edit_app, ["ocr", "S", "Chapter 1", "r0001", "--apply"])
    assert nothing.output.endswith("edit: nothing read; r0001 keeps its source text\n")
    assert store.current_regions(paths)[0].text == "안녕"
    monkeypatch.setattr(edit_cli, "read_again", failing_reader(OSError("model file missing")))
    down = runner.invoke(edit_cli.edit_app, ["ocr", "S", "Chapter 1", "r0001"])
    assert down.exit_code == 1 and "the OCR could not run: model file missing" in down.output

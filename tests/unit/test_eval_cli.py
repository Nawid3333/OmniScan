"""Tests for the `omniscan eval` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.core.schemas import (
    BBox,
    FinalArtifact,
    FinalLine,
    IngestArtifact,
    Region,
    RegionsArtifact,
    SourceFile,
)

SERIES = "S"

runner = CliRunner()

SVG_KR = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="3000">'
    '<flowRoot><flowRegion><rect x="100" y="100" width="400" height="200"/></flowRegion>'
    "<flowPara>안녕하세요</flowPara></flowRoot>"
    '<flowRoot><flowRegion><rect x="600" y="100" width="300" height="200"/></flowRegion>'
    "<flowPara>잘 가</flowPara></flowRoot>"
    '<flowRoot><flowRegion><rect x="100" y="1000" width="400" height="200"/></flowRegion>'
    "<flowPara>미안해요</flowPara></flowRoot></svg>"
)

SVG_EN = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="3000">'
    '<flowRoot><flowRegion><rect x="100" y="100" width="400" height="200"/></flowRegion>'
    "<flowPara>hello</flowPara><flowPara>world</flowPara></flowRoot>"
    '<flowRoot><flowRegion><rect x="600" y="100" width="300" height="200"/></flowRegion>'
    "<flowPara>goodbye</flowPara></flowRoot></svg>"
)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Config with all paths under tmp_path (translated-check next to the library root)."""
    return Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
    )


def write_chapter(cfg: Config, chapter: str, *, with_ocr: bool = True, with_final: bool = True) -> None:
    """Hand-write ingest.json, ocr.json and final.json for one chapter."""
    (cfg.paths.library_root / SERIES / chapter).mkdir(parents=True, exist_ok=True)
    work = cfg.paths.work_root / SERIES / chapter
    work.mkdir(parents=True, exist_ok=True)
    IngestArtifact(
        series=SERIES,
        chapter=chapter,
        strip_width=1000,
        strip_height=3000,
        files=[SourceFile(index=0, name="01.jpg", sha256="x", width=1000, height=3000, y0=0, y1=3000)],
    ).save(work / "ingest.json")
    if with_ocr:
        RegionsArtifact(
            regions=[
                Region(
                    id="r0001",
                    slice_index=0,
                    kind="bubble_text",
                    bbox=BBox(x0=150, y0=150, x1=450, y1=250),
                    text="안녕하세여",
                ),
                Region(
                    id="r0002",
                    slice_index=0,
                    kind="bubble_text",
                    bbox=BBox(x0=620, y0=120, x1=880, y1=280),
                    text="잘가",
                ),
                Region(
                    id="r0003",
                    slice_index=0,
                    kind="bubble_text",
                    bbox=BBox(x0=700, y0=2000, x1=800, y1=2100),
                    text="junk",
                ),
            ]
        ).save(work / "ocr.json")
    if with_final:
        FinalArtifact(
            judge_model="judge",
            lines=[
                FinalLine(region_id="r0001", text="Hello, world!", decision="pick"),
                FinalLine(region_id="r0002", text="Goodbye", decision="pick"),
            ],
        ).save(work / "final.json")


def write_truth(cfg: Config, chapter: str, lang: str = "kr") -> None:
    """Write the ground-truth SVGs of one chapter next to the library root."""
    lang_dir = (
        cfg.paths.library_root.parent / "translated-check" / SERIES / chapter / "truth" / lang
    )
    lang_dir.mkdir(parents=True, exist_ok=True)
    (lang_dir / "E01P01.svg").write_text(SVG_KR if lang == "kr" else SVG_EN, encoding="utf-8")


def test_cli_eval_prints_the_block_and_writes_eval_json(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_chapter(cfg, "Chapter 1")
    write_truth(cfg, "Chapter 1", "kr")
    write_truth(cfg, "Chapter 1", "en")
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["eval", SERIES])
    assert result.exit_code == 0
    assert "S/Chapter 1: 1 pages, 3 truth boxes (0 ignored, 0 dropped)" in result.output
    assert "recall 0.667 (2/3)" in result.output
    assert "precision 0.667 (2/3 regions)" in result.output
    assert "CER macro 0.100" in result.output
    assert "micro 0.143" in result.output
    assert "(2 boxes)" in result.output
    assert "chrF 1.000 (1 pages)" in result.output
    assert 'missed: p01 [100,1000,500,1200] "미안해요"' in result.output
    assert 'worst CER: p01 0.20 "안녕하세요"' in result.output
    report = json.loads(
        (cfg.paths.work_root / SERIES / "Chapter 1" / "eval.json").read_text(encoding="utf-8")
    )
    assert report["recall"] == 2 / 3
    assert report["chrf_mean"] == 1.0


def test_cli_eval_json_prints_one_object_per_chapter(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_chapter(cfg, "Chapter 1")
    write_truth(cfg, "Chapter 1", "kr")
    write_truth(cfg, "Chapter 1", "en")
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["eval", SERIES, "--json"])
    assert result.exit_code == 0
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 1
    report = json.loads(lines[0])
    assert report["chapter"] == "Chapter 1"
    assert report["detected_boxes"] == 2


def test_cli_eval_missing_ocr_exits_1(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    write_chapter(cfg, "Chapter 1", with_ocr=False)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["eval", SERIES])
    assert result.exit_code == 1
    assert "eval: Chapter 1: ocr.json missing — run the ocr stage first" in result.output


def test_cli_eval_missing_ingest_exits_1(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    write_chapter(cfg, "Chapter 1")
    (cfg.paths.work_root / SERIES / "Chapter 1" / "ingest.json").unlink()
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["eval", SERIES])
    assert result.exit_code == 1
    assert "eval: Chapter 1: ingest.json missing — run the ingest stage first" in result.output


def test_cli_eval_missing_truth_dir_exits_2(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    write_chapter(cfg, "Chapter 1")
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["eval", SERIES])
    assert result.exit_code == 2
    assert "no ground truth at" in result.output
    assert str(cfg.paths.library_root.parent / "translated-check" / SERIES / "Chapter 1" / "truth" / "kr") in result.output


def test_cli_eval_selects_a_chapter(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    for chapter in ("Chapter 1", "Chapter 2"):
        write_chapter(cfg, chapter)
        write_truth(cfg, chapter, "kr")
        write_truth(cfg, chapter, "en")
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["eval", SERIES, "--chapter", "Chapter 2"])
    assert result.exit_code == 0
    assert "Chapter 1" not in result.output
    assert "S/Chapter 2: 1 pages, 3 truth boxes" in result.output


def test_cli_eval_unknown_series_exits_2(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["eval", "NoSuchSeries"])
    assert result.exit_code == 2
    assert "no chapters found" in result.output
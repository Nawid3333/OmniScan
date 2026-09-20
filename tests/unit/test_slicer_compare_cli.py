"""CLI tests for `omniscan slice-compare` (card S2): the strip loader is faked, so no GPU is needed."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import omniscan.cli
import omniscan.ingest.strip
from omniscan.cli import app
from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.core.schemas import IngestArtifact
from omniscan.core.stage import ChapterContext
from tests.fixtures.images import plain_jpeg

SERIES = "S"
CHAPTER = "Chapter 1"
PAGES = [(400, 1200), (400, 1200), (400, 1200)]  # strip 400 x 3600
SUMMARY_FIELDS = {
    "strategy",
    "slices",
    "min_height",
    "median_height",
    "max_height",
    "forced",
    "blank",
    "cuts",
}
STRATEGIES = ("smart", "page", "fixed", "simple_gutter")

runner = CliRunner()


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Config with all paths under tmp_path."""
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


@pytest.fixture
def fake_strip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the strip loader with deterministic noise of the ingest's strip size."""
    from tests.fixtures.strip_layouts import art_strip

    def load(ctx: ChapterContext, ingest: IngestArtifact):
        return art_strip(ingest.strip_height, ingest.strip_width, 0)

    monkeypatch.setattr(omniscan.ingest.strip, "load_strip", load)


def write_chapter(cfg: Config, chapter: str = CHAPTER) -> None:
    raw = cfg.paths.library_root / SERIES / chapter
    raw.mkdir(parents=True, exist_ok=True)
    for i, size in enumerate(PAGES, start=1):
        plain_jpeg(raw / f"{i:03d}.jpg", size=size, color=(200, 60, 60))


def by_name(payload: dict) -> dict[str, dict]:
    return {entry["strategy"]: entry for entry in payload["strategies"]}


def test_table_has_one_row_per_strategy(
    cfg: Config, fake_strip: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_chapter(cfg)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["slice-compare", SERIES])
    assert result.exit_code == 0
    assert "strategy" in result.output
    for name in STRATEGIES:
        assert result.output.count(name) == 1


def test_json_output(cfg: Config, fake_strip: None, monkeypatch: pytest.MonkeyPatch) -> None:
    write_chapter(cfg)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["slice-compare", SERIES, "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert set(payload) == {"series", "chapter", "strip_height", "strategies"}
    assert payload["series"] == SERIES
    assert payload["chapter"] == CHAPTER
    assert payload["strip_height"] == 3600
    assert len(payload["strategies"]) == 4
    for entry in payload["strategies"]:
        assert set(entry) == SUMMARY_FIELDS
    summary = by_name(payload)
    assert summary["page"]["slices"] == 3
    assert summary["page"]["cuts"] == [1200, 2400]  # JSON has no tuples
    # 3600 under the default target 3000: one forced cut whose 600-row tail merges back into it
    assert summary["fixed"]["slices"] == 1
    assert summary["fixed"]["cuts"] == []
    assert summary["fixed"]["forced"] == 1
    assert summary["smart"]["slices"] == 1
    assert summary["simple_gutter"]["slices"] == 1


def test_default_chapter_is_the_first(cfg: Config, fake_strip: None, monkeypatch: pytest.MonkeyPatch) -> None:
    write_chapter(cfg, "Chapter 2")
    write_chapter(cfg, "Chapter 1")
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["slice-compare", SERIES, "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["chapter"] == "Chapter 1"


def test_series_toml_target_height_is_honoured(
    cfg: Config, fake_strip: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_chapter(cfg)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["slice-compare", SERIES, "--json"])
    assert result.exit_code == 0
    assert by_name(json.loads(result.output))["fixed"]["slices"] == 1

    toml = cfg.paths.library_root / SERIES / "series.toml"
    toml.write_text("[slicer]\ntarget_height = 2000\n", encoding="utf-8")
    result = runner.invoke(app, ["slice-compare", SERIES, "--json"])
    assert result.exit_code == 0
    assert by_name(json.loads(result.output))["fixed"]["slices"] == 2  # [0,2000) + [2000,3600)


def test_writes_nothing_to_work_dir(cfg: Config, fake_strip: None, monkeypatch: pytest.MonkeyPatch) -> None:
    write_chapter(cfg)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    assert runner.invoke(app, ["ingest", SERIES]).exit_code == 0  # real ingest first (CPU codec)

    work = cfg.paths.work_root / SERIES / CHAPTER
    before = {p.name for p in work.iterdir()}
    result = runner.invoke(app, ["slice-compare", SERIES])
    assert result.exit_code == 0
    assert {p.name for p in work.iterdir()} == before  # no slices.json, no manifest rewrite


def test_unknown_series_and_chapter_exit_1(
    cfg: Config, fake_strip: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_chapter(cfg)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["slice-compare", "NoSuchSeries"])
    assert result.exit_code == 1
    assert "no chapters found" in result.output

    result = runner.invoke(app, ["slice-compare", SERIES, "--chapter", "Chapter 9"])
    assert result.exit_code == 1
    assert "unknown chapter" in result.output

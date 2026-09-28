"""Find missed text on one page: which boxes count as covered, the page rows, the web route and `omniscan edit
find` (torch-free: the search itself is faked; tests/unit/test_detect_on_demand.py runs it with fake models)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
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
from omniscan.detect.on_demand import Found, covered, page_rows
from omniscan.edits import store
from omniscan.web.app import create_app

BASE = "/api/series/S/chapters/Chapter%201"
MISSED = Found(
    kind="free_text",
    bbox=BBox(x0=40, y0=450, x1=160, y1=500),
    bubble_bbox=None,
    score=0.42,
    text="쾅",
    confidence=0.9,
)


def box(x0: int, y0: int, x1: int, y1: int) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )


@pytest.fixture
def paths(cfg: Config) -> ChapterPaths:
    """Two 300x400 pages (the second promo-filtered away is not listed), one slice each, one region on page 0."""
    paths = SeriesPaths.from_config(cfg, "S").chapter("Chapter 1")
    paths.raw_dir.mkdir(parents=True)
    IngestArtifact(
        series="S",
        chapter="Chapter 1",
        strip_width=300,
        strip_height=800,
        files=[
            SourceFile(index=0, name="001.png", sha256="0" * 64, width=300, height=400, y0=0, y1=400),
            SourceFile(index=2, name="003.png", sha256="2" * 64, width=300, height=400, y0=400, y1=800),
        ],
    ).save(paths.artifact("ingest.json"))
    SlicesArtifact(
        strip_width=300,
        strip_height=800,
        bands=[],
        slices=[Slice(index=0, y0=0, y1=400), Slice(index=1, y0=400, y1=800)],
    ).save(paths.artifact("slices.json"))
    region = Region(id="r0001", slice_index=0, kind="bubble_text", bbox=box(50, 50, 150, 120), text="안녕")
    RegionsArtifact(regions=[region]).save(paths.artifact("ocr.json"))
    return paths


def test_a_box_a_region_already_has_is_covered() -> None:
    regions = [Region(id="r0001", slice_index=0, kind="bubble_text", bbox=box(50, 50, 150, 120))]
    assert covered(box(52, 50, 150, 118), regions)  # the same box
    assert covered(box(60, 60, 100, 90), regions)  # text inside the region's box
    assert covered(box(30, 30, 200, 170), regions)  # a bubble found around it
    assert not covered(box(140, 100, 260, 200), regions)  # a corner overlaps, the text is elsewhere
    assert not covered(box(50, 300, 150, 370), regions)


def test_page_rows_follow_the_ingest_index(paths: ChapterPaths) -> None:
    ingest = IngestArtifact.load(paths.artifact("ingest.json"))
    assert page_rows(ingest, 2) == box(0, 400, 300, 800)  # indices keep the promo page's gap
    with pytest.raises(ValueError, match="no page with index 1"):
        page_rows(ingest, 1)
    promo = SourceFile(
        index=1, name="002.png", sha256="1" * 64, width=300, height=400, y0=400, y1=400, filtered=True
    )
    with pytest.raises(ValueError, match="no page with index 1"):  # a promo page is not part of the strip
        page_rows(ingest.model_copy(update={"files": [*ingest.files, promo]}), 1)


def test_the_web_route_returns_suggestions_and_writes_nothing(paths: ChapterPaths, cfg: Config) -> None:
    calls: list[tuple[int, float | None]] = []
    failure: list[Exception] = []

    def finder(given: ChapterPaths, page: int, scfg: Config, threshold: float | None) -> list[Found]:
        if failure:
            raise failure[0]
        calls.append((page, threshold))
        return [MISSED]

    client = TestClient(create_app(cfg, page_finder=finder))
    response = client.post(f"{BASE}/pages/2/find", json={"threshold": 0.2})
    assert response.status_code == 200
    assert response.json() == {
        "found": [
            {
                "kind": "free_text",
                "bbox": {"x0": 40, "y0": 450, "x1": 160, "y1": 500},
                "bubble_bbox": None,
                "score": 0.42,
                "text": "쾅",
                "confidence": 0.9,
            }
        ]
    }
    assert client.post(f"{BASE}/pages/0/find", json={}).status_code == 200
    assert calls == [(2, 0.2), (0, None)]
    assert [r.id for r in store.current_regions(paths)] == ["r0001"] and not paths.artifact(
        store.EDITS_FILE
    ).exists()
    assert client.post(f"{BASE}/pages/0/find", json={"threshold": 1.5}).status_code == 422
    for exc, status in (
        (FileNotFoundError("slices.json missing"), 404),
        (ValueError("the chapter has no page with index 9"), 422),
        (RuntimeError("no GPU"), 503),
        (ImportError("No module named 'torch'"), 503),
    ):
        failure[:] = [exc]
        assert client.post(f"{BASE}/pages/9/find", json={}).status_code == status


def test_edit_find_on_the_command_line(
    paths: ChapterPaths, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(edit_cli, "get_config", lambda: cfg)
    asked: list[int] = []

    def finder(given: ChapterPaths, page: int, scfg: Config, threshold: float | None) -> list[Found]:
        asked.append(page)
        return [MISSED, MISSED] if page == 2 else []

    monkeypatch.setattr(edit_cli, "find_missed", finder)
    runner = CliRunner()
    listed = runner.invoke(edit_cli.edit_app, ["find", "S", "Chapter 1", "2"])
    assert listed.exit_code == 0 and asked == [2]  # page 2 as the Studio counts is ingest index 2
    assert listed.output.count("edit: free_text at 40,450,160,500 (0.42): 쾅") == 2
    assert (
        runner.invoke(edit_cli.edit_app, ["find", "S", "Chapter 1", "1"]).output
        == "edit: nothing missed on page 1\n"
    )
    steps = store.history_steps(paths)
    added = runner.invoke(edit_cli.edit_app, ["find", "S", "Chapter 1", "2", "--add", "--threshold", "0.2"])
    assert added.exit_code == 0 and added.output.endswith("edit: added m0001, m0002\n")
    new = [r for r in store.current_regions(paths) if r.id.startswith("m")]
    assert [(r.text, r.kind, r.slice_index) for r in new] == [("쾅", "free_text", 1)] * 2
    assert store.history_steps(paths)[0] == steps[0] + 1  # --add is one undo step
    missing = runner.invoke(edit_cli.edit_app, ["find", "S", "Chapter 1", "3"])
    assert missing.exit_code == 2 and "the chapter has 2 page(s), not 3" in missing.output

    def broken(*_: object) -> list[Found]:
        raise OSError("model file missing")

    monkeypatch.setattr(edit_cli, "find_missed", broken)
    failed = runner.invoke(edit_cli.edit_app, ["find", "S", "Chapter 1", "1"])
    assert failed.exit_code == 1 and "the detector could not run: model file missing" in failed.output

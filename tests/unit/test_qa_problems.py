"""Tests for a chapter's problem list (qa/problems.py): the Studio's checks and the finished-page re-read in one
list, `omniscan edit problems` and the web API's GET …/problems."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import omniscan.edits.cli as edit_cli
from omniscan.core.config import Config, PathsConfig
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import (
    BBox,
    FinalArtifact,
    FinalLine,
    LayoutArtifact,
    LayoutItem,
    QaArtifact,
    QaIssue,
    Region,
    RegionKind,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
)
from omniscan.edits import store
from omniscan.qa.leftover import QA_FILE
from omniscan.qa.problems import Problem, chapter_problems
from omniscan.web.app import create_app

ARGS = ["S", "Chapter 1"]


def region(rid: str, y0: int, text: str, kind: RegionKind = "bubble_text") -> Region:
    return Region(id=rid, slice_index=0, kind=kind, bbox=BBox(x0=10, y0=y0, x1=110, y1=y0 + 50), text=text)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )


@pytest.fixture
def paths(cfg: Config) -> ChapterPaths:
    """Five regions with one kind of problem each (r0005, a sound effect left untranslated, has none), and the
    lettering of r0001 overflowing."""
    paths = SeriesPaths.from_config(cfg, "S").chapter("Chapter 1")
    paths.raw_dir.mkdir(parents=True)
    SlicesArtifact(strip_width=200, strip_height=600, bands=[], slices=[Slice(index=0, y0=0, y1=600)]).save(
        paths.artifact("slices.json")
    )
    regions = [
        region("r0001", 10, "안녕"),
        region("r0002", 100, "반가워"),
        region("r0003", 200, "가자"),
        region("r0004", 300, "말했지"),
        region("r0005", 400, "쾅", kind="sfx"),
    ]
    RegionsArtifact(regions=regions).save(paths.artifact("ocr.json"))
    lines = {"r0001": "Hi", "r0003": "Let's 가자", "r0004": "I sayy it again."}
    FinalArtifact(
        judge_model="judge",
        lines=[
            FinalLine(
                region_id=rid,
                text=text,
                decision="pick",
                flags=["uncertain"] if rid == "r0001" else [],
            )
            for rid, text in lines.items()
        ],
    ).save(paths.artifact("final.json"))
    box = BBox(x0=10, y0=10, x1=110, y1=60)
    LayoutArtifact(
        items=[
            LayoutItem(
                region_id="r0001",
                font_role="dialogue",
                font="f.ttf",
                size_px=12,
                lines=["Hi"],
                box=box,
                overflow=True,
            )
        ]
    ).save(paths.artifact("layout.json"))
    return paths


def rerun_qa(paths: ChapterPaths) -> None:
    """A finished-page re-read that still read r0003's source text, and text of a region deleted since."""
    QaArtifact(
        checked=3,
        issues=[
            QaIssue(region_id="r0099", kind="source_left", message="the original text is still readable"),
            QaIssue(
                region_id="r0003",
                kind="source_left",
                message="the original text is still readable",
                read="가자",
            ),
        ],
    ).save(paths.artifact(QA_FILE))


def test_the_problems_of_a_chapter_in_reading_order(paths: ChapterPaths) -> None:
    store.set_checked(paths, ["r0003"])
    rerun_qa(paths)
    problems = chapter_problems(paths)
    assert [(p.region_id, p.kind, p.finished_page) for p in problems] == [
        ("r0001", "overflow", False),
        ("r0001", "uncertain", False),
        ("r0002", "untranslated", False),
        ("r0003", "source_left", False),
        ("r0003", "source_left", True),
        ("r0004", "typo", False),
        ("r0099", "source_left", True),
    ]
    assert [p.status for p in problems] == ["todo", "todo", "todo", "checked", "checked", "todo", "todo"]
    typo = problems[5]
    assert typo.word == "sayy" and "sayy" in typo.message
    assert problems[4].read == "가자" and problems[3].read == ""
    assert problems[2] == Problem("r0002", "untranslated", "no English line", "todo")


def test_hand_edits_count_and_a_damaged_qa_file_is_no_re_read(paths: ChapterPaths) -> None:
    store.set_translation(paths, "r0002", "Glad to see you", direction="ltr")
    store.update_region(paths, "r0004", direction="ltr", kind="sfx")  # effects are not spell-checked
    paths.artifact(QA_FILE).write_text('{"checked": ', encoding="utf-8")
    problems = chapter_problems(paths)
    assert [(p.region_id, p.kind) for p in problems] == [
        ("r0001", "overflow"),
        ("r0001", "uncertain"),
        ("r0003", "source_left"),
    ]
    assert [p.status for p in problems] == ["todo", "todo", "todo"]


def test_no_problems_before_detection(cfg: Config) -> None:
    paths = SeriesPaths.from_config(cfg, "S").chapter("Chapter 2")
    paths.raw_dir.mkdir(parents=True)
    assert chapter_problems(paths) == []


def test_edit_problems_on_the_command_line(
    paths: ChapterPaths, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(edit_cli, "get_config", lambda: cfg)
    store.set_checked(paths, ["r0003"])
    rerun_qa(paths)
    runner = CliRunner()
    shown = runner.invoke(edit_cli.edit_app, ["problems", *ARGS])
    assert shown.exit_code == 0
    lines = shown.output.splitlines()
    assert lines[0] == "r0001  overflow: the lettering does not fit its balloon"
    assert lines[3] == "r0003  source_left: source-language characters left in the English [checked]"
    assert lines[4] == "r0003  source_left (finished page): the original text is still readable [checked]"
    assert lines[-1] == "edit: 7 problem(s) in S / Chapter 1"
    left = runner.invoke(edit_cli.edit_app, ["problems", *ARGS, "--unchecked", "--json"])
    rows = json.loads(left.output)
    assert [(r["region_id"], r["kind"]) for r in rows] == [
        ("r0001", "overflow"),
        ("r0001", "uncertain"),
        ("r0002", "untranslated"),
        ("r0004", "typo"),
        ("r0099", "source_left"),
    ]
    assert rows[3]["word"] == "sayy" and rows[4]["finished_page"] is True


def test_problems_over_the_web_api(paths: ChapterPaths, cfg: Config) -> None:
    rerun_qa(paths)
    client = TestClient(create_app(cfg))
    response = client.get("/api/series/S/chapters/Chapter%201/problems")
    assert response.status_code == 200
    problems = response.json()["problems"]
    assert len(problems) == 7
    assert problems[4] == {
        "region_id": "r0003",
        "kind": "source_left",
        "message": "the original text is still readable",
        "status": "todo",
        "finished_page": True,
        "word": "",
        "read": "가자",
    }
    empty = client.get("/api/series/S/chapters/Chapter%209/problems")
    assert empty.status_code == 200 and empty.json() == {"problems": []}
    assert client.get("/api/series/S/chapters/..%2F..%2Fx/problems").status_code == 404

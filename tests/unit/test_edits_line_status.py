"""Tests for per-line status (todo / edited / checked): edits.store, the desktop session, `omniscan edit check`
and the web API (torch-free)."""

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
    Region,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
)
from omniscan.edits import store
from omniscan.edits.session import StudioSession
from omniscan.web.app import create_app

ARGS = ["S", "Chapter 1"]


def region(rid: str, y0: int, text: str) -> Region:
    return Region(
        id=rid, slice_index=0, kind="bubble_text", bbox=BBox(x0=10, y0=y0, x1=110, y1=y0 + 50), text=text
    )


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )


@pytest.fixture
def paths(cfg: Config) -> ChapterPaths:
    """Three read and judged regions."""
    paths = SeriesPaths.from_config(cfg, "S").chapter("Chapter 1")
    paths.raw_dir.mkdir(parents=True)
    SlicesArtifact(strip_width=200, strip_height=600, bands=[], slices=[Slice(index=0, y0=0, y1=600)]).save(
        paths.artifact("slices.json")
    )
    regions = [region("r0001", 10, "안녕"), region("r0002", 200, "반가워"), region("r0003", 400, "가자")]
    RegionsArtifact(regions=regions).save(paths.artifact("ocr.json"))
    FinalArtifact(
        judge_model="judge",
        lines=[
            FinalLine(region_id=rid, text=text, decision="pick")
            for rid, text in (("r0001", "Hi"), ("r0002", "Glad"), ("r0003", "Let's go"))
        ],
    ).save(paths.artifact("final.json"))
    return paths


def rejudge(paths: ChapterPaths, lines: dict[str, str]) -> None:
    """A judge re-run that writes new machine lines (hand lines are re-applied on top, as the stage does)."""
    judged = FinalArtifact(
        judge_model="judge",
        lines=[FinalLine(region_id=rid, text=text, decision="pick") for rid, text in lines.items()],
    )
    store.write_final(paths, judged, store.current_regions(paths))


def test_a_check_holds_until_the_source_or_english_changes(paths: ChapterPaths) -> None:
    assert store.line_statuses(paths) == {"r0001": "todo", "r0002": "todo", "r0003": "todo"}
    store.set_translation(paths, "r0002", "Nice to see you", direction="ltr")
    assert store.line_statuses(paths)["r0002"] == "edited"
    store.set_checked(paths, ["r0001", "r0002", "r0003"])
    assert set(store.line_statuses(paths).values()) == {"checked"}
    store.update_region(paths, "r0001", direction="ltr", text="안녕!")  # the source changed after the check
    store.set_translation(paths, "r0002", "Good to see you", direction="ltr")  # and a checked line
    rejudge(
        paths, {"r0001": "Hi", "r0002": "Glad", "r0003": "Go!"}
    )  # the machine's line changed under a check
    assert store.line_statuses(paths) == {"r0001": "edited", "r0002": "edited", "r0003": "todo"}
    store.set_checked(paths, ["r0003"])
    store.set_checked(paths, ["r0003"])  # checking again replaces the check, never duplicates it
    assert len(store.load_edits(paths).checked) == 3 and store.line_statuses(paths)["r0003"] == "checked"
    store.set_checked(paths, ["r0003"], checked=False)
    assert store.line_statuses(paths)["r0003"] == "todo" and len(store.load_edits(paths).checked) == 2
    store.undo(paths, direction="ltr")  # a check is one undo step
    assert store.line_statuses(paths)["r0003"] == "checked"
    with pytest.raises(store.EditNotFoundError):
        store.set_checked(paths, ["r0009"])


def test_a_check_follows_its_region_through_a_re_detection(paths: ChapterPaths) -> None:
    store.set_checked(paths, ["r0002"])
    read = store.current_regions(paths)  # the pipeline renumbers its regions: r0002 is now r0005 (same box)
    renamed = [r.model_copy(update={"id": "r0005"}) if r.id == "r0002" else r for r in read]
    RegionsArtifact(regions=renamed).save(paths.artifact(store.OCR_AUTO_FILE))
    store.rebuild(paths, store.load_edits(paths), direction="ltr")
    rejudge(paths, {"r0001": "Hi", "r0005": "Glad", "r0003": "Let's go"})
    assert store.line_statuses(paths) == {"r0001": "todo", "r0005": "checked", "r0003": "todo"}


def test_the_desktop_session_holds_checks_until_save(paths: ChapterPaths) -> None:
    session = StudioSession(paths, direction="ltr")
    session.set_translation("r0001", "Hello there")
    session.set_checked("r0001")  # checked together with the line it approves
    session.set_checked("r0002")
    assert [row.status for row in session.rows()] == ["checked", "checked", "todo"] and session.dirty
    assert store.line_statuses(paths)["r0001"] == "todo"  # nothing saved yet
    assert session.save() == 3
    assert store.line_statuses(paths) == {"r0001": "checked", "r0002": "checked", "r0003": "todo"}
    assert store.load_edits(paths).checked[0].english == "Hello there"
    session.set_checked("r0002", False)
    session.set_checked("r0003")
    session.set_checked("r0003", False)  # back to what is saved: no change left
    assert [row.status for row in session.rows()] == ["checked", "todo", "todo"]
    assert session.save() == 1 and store.line_statuses(paths)["r0002"] == "todo"
    assert session.undo() and [row.status for row in session.rows()] == ["checked", "checked", "todo"]
    assert session.undo() and store.line_statuses(paths)["r0001"] == "todo"  # the first save was one step
    session.set_source("r0003", "가자!")
    assert session.rows()[2].status == "edited"  # an unsaved edit
    session.remove_region("r0003")
    with pytest.raises(KeyError):
        session.set_checked("r0003")
    with pytest.raises(KeyError):
        session.set_checked("r0009")


def test_edit_check_on_the_command_line(
    paths: ChapterPaths, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(edit_cli, "get_config", lambda: cfg)
    runner = CliRunner()

    def edit(*args: str) -> str:
        result = runner.invoke(edit_cli.edit_app, list(args))
        return f"{result.exit_code}:{result.output}"

    assert edit("check", *ARGS, "r0001", "r0003") == "0:edit: 2 line(s) checked; 2 of 3 checked\n"
    shown = json.loads(runner.invoke(edit_cli.edit_app, ["show", *ARGS, "--json"]).output)
    assert [r["status"] for r in shown["regions"]] == ["checked", "todo", "checked"]
    assert "checked" in runner.invoke(edit_cli.edit_app, ["show", *ARGS]).output
    assert edit("check", *ARGS, "r0003", "--uncheck") == "0:edit: 1 line(s) unchecked; 1 of 3 checked\n"
    assert edit("check", *ARGS, "--all") == "0:edit: 3 line(s) checked; 3 of 3 checked\n"
    assert edit("check", *ARGS).startswith("2:edit: name the regions to check, or pass --all")
    assert edit("check", *ARGS, "r0001", "--all").startswith("2:edit: name regions or pass --all, not both")
    assert edit("check", *ARGS, "r0009").startswith("2:edit: region 'r0009' not found")


def test_checked_lines_over_the_web_api(paths: ChapterPaths, cfg: Config) -> None:
    client = TestClient(create_app(cfg))
    url = "/api/series/S/chapters/Chapter%201"
    assert client.get(f"{url}/edits").json()["checked_region_ids"] == []
    done = client.post(f"{url}/checked", json={"region_ids": ["r0002", "r0003"]})
    assert done.status_code == 200 and done.json()["checked_region_ids"] == ["r0002", "r0003"]
    assert done.json()["history"] == {"undo": 1, "redo": 0}
    undone = client.post(f"{url}/checked", json={"region_ids": ["r0003"], "checked": False})
    assert undone.json()["checked_region_ids"] == ["r0002"]
    assert client.post(f"{url}/checked", json={"region_ids": ["r0009"]}).status_code == 404
    assert client.post(f"{url}/checked", json={"region_ids": []}).status_code == 422

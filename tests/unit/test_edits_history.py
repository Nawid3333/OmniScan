"""Tests for undo and redo of hand edits (edits/store.py history) through the store, the Studio session, the CLI
and the web API (torch-free)."""

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
    ChapterEdits,
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


def box(x0: int, y0: int, x1: int, y1: int) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def region(rid: str, y0: int, text: str, order: int) -> Region:
    return Region(
        id=rid,
        slice_index=0,
        kind="bubble_text",
        bbox=box(10, y0, 110, y0 + 50),
        text=text,
        reading_order=order,
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
    paths = SeriesPaths.from_config(cfg, "S").chapter("Chapter 1")
    paths.raw_dir.mkdir(parents=True)
    SlicesArtifact(strip_width=200, strip_height=600, bands=[], slices=[Slice(index=0, y0=0, y1=600)]).save(
        paths.artifact("slices.json")
    )
    RegionsArtifact(regions=[region("r0001", 10, "안녕", 0), region("r0002", 200, "반가워", 1)]).save(
        paths.artifact("ocr.json")
    )
    FinalArtifact(judge_model="judge", lines=[FinalLine(region_id="r0001", text="Hi", decision="pick")]).save(
        paths.artifact("final.json")
    )
    return paths


def texts(paths: ChapterPaths) -> dict[str, str]:
    return {r.id: r.text for r in store.current_regions(paths)}


def english(paths: ChapterPaths) -> dict[str, str]:
    return {line.region_id: line.text for line in FinalArtifact.load(paths.artifact("final.json")).lines}


# ---------------------------------------------------------------- the store


def test_undo_and_redo_walk_the_edits_back_and_forth(paths: ChapterPaths) -> None:
    assert store.history_steps(paths) == (0, 0)
    store.update_region(paths, "r0001", direction="ltr", text="안녕하세요")
    store.set_translation(paths, "r0002", "Nice to see you", direction="ltr")
    store.delete_region(paths, "r0001", direction="ltr")
    assert store.history_steps(paths) == (3, 0)
    store.undo(paths, direction="ltr")  # the deletion
    assert texts(paths) == {"r0001": "안녕하세요", "r0002": "반가워"}
    store.undo(paths, direction="ltr")  # the English line
    assert english(paths) == {"r0001": "Hi"}
    store.undo(paths, direction="ltr")  # the text fix: back to the pipeline's reading
    assert texts(paths) == {"r0001": "안녕", "r0002": "반가워"} and store.load_edits(paths) == ChapterEdits()
    assert store.history_steps(paths) == (0, 3)
    with pytest.raises(store.EditNotFoundError, match="nothing to undo"):
        store.undo(paths, direction="ltr")
    store.redo(paths, direction="ltr")
    store.redo(paths, direction="ltr")
    assert texts(paths)["r0001"] == "안녕하세요" and english(paths)["r0002"] == "Nice to see you"
    assert store.history_steps(paths) == (2, 1)


def test_a_new_edit_forgets_the_redo_steps_and_a_no_op_records_nothing(paths: ChapterPaths) -> None:
    store.set_cuts(paths, [300])
    store.undo(paths, direction="ltr")
    assert store.load_edits(paths).cuts is None and store.history_steps(paths) == (0, 1)
    store.set_cuts(paths, [250])
    assert store.history_steps(paths) == (1, 0)
    with pytest.raises(store.EditNotFoundError, match="nothing to redo"):
        store.redo(paths, direction="ltr")
    store.set_cuts(paths, [250])  # the same cuts again: edits.json does not change
    assert store.history_steps(paths) == (1, 0)


def test_the_history_keeps_the_last_steps_only(paths: ChapterPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "HISTORY_DEPTH", 3)
    for row in (100, 200, 300, 400, 500):
        store.set_cuts(paths, [row])
    assert store.history_steps(paths) == (3, 0)
    for _ in range(3):
        store.undo(paths, direction="ltr")
    assert store.load_edits(paths).cuts == [200]  # the two oldest steps are gone


def test_an_edit_group_is_one_step(paths: ChapterPaths) -> None:
    store.set_translation(paths, "r0001", "Hello", direction="ltr")
    with store.edit_group(paths):
        store.update_region(paths, "r0002", direction="ltr", text="반가워요")
        store.set_translation(paths, "r0002", "Good to see you", direction="ltr")
        with store.edit_group(paths):  # nested: still the same step
            store.add_region(paths, box(10, 400, 110, 450), direction="ltr", text="쾅")
    assert store.history_steps(paths) == (2, 0)
    store.undo(paths, direction="ltr")
    assert texts(paths) == {"r0001": "안녕", "r0002": "반가워"} and english(paths) == {"r0001": "Hello"}
    with store.edit_group(paths):
        pass  # nothing changed: nothing recorded
    assert store.history_steps(paths) == (1, 1)


def test_an_undo_that_cannot_rebuild_changes_nothing(paths: ChapterPaths) -> None:
    store.update_region(paths, "r0001", direction="ltr", text="안녕하세요")
    store.update_region(paths, "r0002", direction="ltr", text="반가워요")
    paths.artifact("slices.json").unlink()
    before = store.load_edits(paths)
    with pytest.raises(store.EditNotFoundError, match=r"slices\.json"):
        store.undo(paths, direction="ltr")
    assert store.load_edits(paths) == before and store.history_steps(paths) == (2, 0)


# ---------------------------------------------------------------- the Studio session


def test_a_session_save_is_one_undo_step(paths: ChapterPaths) -> None:
    session = StudioSession(paths, direction="ltr")
    session.set_source("r0002", "반가워요")
    session.set_translation("r0002", "Nice to see you")
    session.remove_region("r0001")
    assert session.save() == 3 and store.history_steps(paths) == (1, 0)
    assert session.undo() and [row.region_id for row in session.rows()] == ["r0001", "r0002"]
    assert session.rows()[1].source == "반가워" and session.translations() == {"r0001": "Hi"}
    session.set_source("r0002", "unsaved")
    assert session.redo() and not session.dirty  # the unsaved change is dropped
    assert [row.region_id for row in session.rows()] == ["r0002"]
    assert not session.redo() and session.undo() and session.undo() is False


# ---------------------------------------------------------------- the CLI and the web API


def test_edit_undo_and_redo_on_the_command_line(
    paths: ChapterPaths, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(edit_cli, "get_config", lambda: cfg)
    runner = CliRunner()
    runner.invoke(edit_cli.edit_app, ["english", "S", "Chapter 1", "r0002", "Nice to see you"])
    undone = runner.invoke(edit_cli.edit_app, ["undo", "S", "Chapter 1"])
    assert undone.output == "edit: undone (0 more to undo, 1 to redo)\n" and english(paths) == {"r0001": "Hi"}
    redone = runner.invoke(edit_cli.edit_app, ["redo", "S", "Chapter 1"])
    assert redone.output == "edit: redone (1 more to undo, 0 to redo)\n"
    nothing = runner.invoke(edit_cli.edit_app, ["redo", "S", "Chapter 1"])
    assert nothing.exit_code == 2 and "nothing to redo" in nothing.output


def test_undo_and_redo_over_the_web_api(paths: ChapterPaths, cfg: Config) -> None:
    client = TestClient(create_app(cfg))
    base = "/api/series/S/chapters/Chapter%201"
    assert client.get(f"{base}/edits").json()["history"] == {"undo": 0, "redo": 0}
    assert client.post(f"{base}/edits/undo", json={}).status_code == 409
    client.patch(f"{base}/regions/r0002", json={"text": "반가워요"})
    after = client.post(f"{base}/edits/undo", json={})
    assert after.status_code == 200 and after.json()["history"] == {"undo": 0, "redo": 1}
    assert after.json()["regions"] == [] and texts(paths)["r0002"] == "반가워"
    again = client.post(f"{base}/edits/redo", json={}).json()
    assert again["edited_region_ids"] == ["r0002"] and texts(paths)["r0002"] == "반가워요"
    assert client.post(f"{base}/edits/sideways", json={}).status_code == 422

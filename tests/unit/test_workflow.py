"""A group's workflow on a chapter (#38): status and hand-overs, notes on regions, each role's to-do list, signed
edits, and the `omniscan workflow` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from omniscan.core.config import Config, PathsConfig, UserConfig
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
    RegionsArtifact,
    Slice,
    SlicesArtifact,
)
from omniscan.edits import store
from omniscan.edits.session import StudioSession
from omniscan.workflow import cli as workflow_cli
from omniscan.workflow.notes import NOTES_FILE, add_note, load_notes, notes_by_region, resolve_note
from omniscan.workflow.status import STATUS_FILE, describe, hand_over, load_status, mark_step, next_step
from omniscan.workflow.todo import todo

runner = CliRunner()


def _region(rid: str, y0: int, text: str, order: int) -> Region:
    box = BBox(x0=10, y0=y0, x1=110, y1=y0 + 50)
    return Region(id=rid, slice_index=0, kind="bubble_text", bbox=box, text=text, reading_order=order)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )


@pytest.fixture
def paths(cfg: Config) -> ChapterPaths:
    """Chapter 1 of series S: three read regions, two with an English line."""
    series = SeriesPaths.from_config(cfg, "S")
    (series.library_dir / "Chapter 1").mkdir(parents=True)
    paths = series.chapter("Chapter 1")
    SlicesArtifact(strip_width=200, strip_height=600, bands=[], slices=[Slice(index=0, y0=0, y1=600)]).save(
        paths.artifact("slices.json")
    )
    regions = [
        _region("r0001", 10, "안녕", 0),
        _region("r0002", 200, "반가워", 1),
        _region("r0003", 400, "가자", 2),
    ]
    RegionsArtifact(regions=regions).save(paths.artifact("ocr.json"))
    FinalArtifact(
        judge_model="judge",
        lines=[
            FinalLine(region_id="r0001", text="Hi", decision="pick"),
            FinalLine(region_id="r0002", text="Nice to meet you", decision="pick"),
        ],
    ).save(paths.artifact("final.json"))
    return paths


# ---------------------------------------------------------------- status


def test_steps_are_marked_done_in_order_and_kept_with_who_did_them(paths: ChapterPaths) -> None:
    assert load_status(paths).done == [] and describe(load_status(paths)) == "not started"
    mark_step(paths, "proofread", by="Ana")
    status = mark_step(paths, "translated", by="Ben", note="first pass")
    assert status.done == ["translated", "proofread"] and next_step(status) == "cleaned"
    assert mark_step(paths, "translated", by="Ben") == status  # already done: nothing changes
    status = mark_step(paths, "proofread", done=False)
    assert status.done == ["translated"] and describe(status) == "translated (1/5)"
    status = hand_over(paths, "Cleo", by="Ben", note="cleaning next")
    assert status.holder == "Cleo" and describe(status) == "translated (1/5) · with Cleo"
    assert [(e.kind, e.step, e.to, e.by, e.note) for e in status.events] == [
        ("done", "proofread", None, "Ana", None),
        ("done", "translated", None, "Ben", "first pass"),
        ("undone", "proofread", None, None, None),
        ("handed", None, "Cleo", "Ben", "cleaning next"),
    ]
    assert load_status(paths) == status
    for step in ("proofread", "cleaned", "lettered", "qc_passed"):
        status = mark_step(paths, step)  # type: ignore[arg-type]
    assert next_step(status) is None and describe(hand_over(paths, "")) == "QC passed (5/5)"
    with pytest.raises(ValueError, match="unknown step"):
        mark_step(paths, "printed")  # type: ignore[arg-type]
    paths.artifact(STATUS_FILE).write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        load_status(paths)


# ---------------------------------------------------------------- notes


def test_notes_sit_on_regions_follow_them_and_can_be_resolved(paths: ChapterPaths) -> None:
    first = add_note(paths, "r0002", "  Is this polite enough?  ", by="Ana")
    second = add_note(paths, "r0002", "Keep the honorific")
    assert (first.id, first.text, first.by, second.id, second.by) == (
        "n0001",
        "Is this polite enough?",
        "Ana",
        "n0002",
        None,
    )
    with pytest.raises(KeyError):
        add_note(paths, "r0099", "nowhere")
    with pytest.raises(ValueError, match="needs some text"):
        add_note(paths, "r0001", "   ")
    assert resolve_note(paths, "n0001").resolved and not load_notes(paths)[1].resolved
    with pytest.raises(KeyError):
        resolve_note(paths, "n0042")
    # a re-run renumbers the regions: the notes follow the box they were written on
    renumbered = [_region("r0001", 10, "안녕", 0), _region("r0005", 200, "반가워", 1)]
    assert [n.id for n in notes_by_region(paths, renumbered)["r0005"]] == ["n0001", "n0002"]
    assert notes_by_region(paths, [_region("r0001", 10, "안녕", 0)]) == {}  # its region is gone
    assert paths.artifact(NOTES_FILE).is_file()


# ---------------------------------------------------------------- to-do per role


def test_each_role_sees_what_it_has_left(paths: ChapterPaths) -> None:
    assert [(t.region_id, t.what) for t in todo(paths, "translator")] == [("r0003", "no translation yet")]
    assert [t.region_id for t in todo(paths, "proofreader")] == ["r0001", "r0002"]
    store.set_checked(paths, ["r0001"])
    assert [t.region_id for t in todo(paths, "proofreader")] == ["r0002"]
    assert todo(paths, "cleaner") == [] and todo(paths, "typesetter") == []  # nothing re-read or lettered yet
    QaArtifact(
        checked=2,
        issues=[
            QaIssue(region_id="r0002", kind="source_left", message="the original text still shows"),
            QaIssue(region_id="r0001", kind="other", message="something else"),
        ],
    ).save(paths.artifact("qa.json"))
    LayoutArtifact(
        items=[
            LayoutItem(
                region_id="r0001",
                font_role="dialogue",
                font="Mali-SemiBold.ttf",
                size_px=14,
                lines=["Hi"],
                box=BBox(x0=10, y0=10, x1=60, y1=30),
                color=(0, 0, 0),
                overflow=True,
            )
        ]
    ).save(paths.artifact("layout.json"))
    add_note(paths, "r0003", "check the name")
    assert [(t.region_id, t.what) for t in todo(paths, "cleaner")] == [
        ("r0002", "the original text still shows"),
        ("r0003", "note n0001: check the name"),  # open notes show for every role
    ]
    typeset = todo(paths, "typesetter")
    assert (typeset[0].region_id, typeset[0].what) == ("r0001", "lettering does not fit")
    assert [t.what for t in todo(paths, "qc")] == [
        "lettering does not fit",
        "something else",
        "the original text still shows",
        "note n0001: check the name",
    ]


# ---------------------------------------------------------------- signed edits


def test_edits_are_signed_by_who_made_them_and_keep_their_signature(paths: ChapterPaths) -> None:
    store.set_translation(paths, "r0003", "Let's go", direction="ltr")  # no author set: unsigned
    with store.edit_author("Ana"):
        store.set_translation(paths, "r0001", "Hello", direction="ltr")
        store.set_checked(paths, ["r0002"])
    with store.edit_author("Ben"):
        store.set_translation(paths, "r0002", "Good to see you", direction="ltr")
    edits = store.load_edits(paths)
    assert {t.region_id: t.by for t in edits.translations} == {"r0003": None, "r0001": "Ana", "r0002": "Ben"}
    assert [c.by for c in edits.checked] in (
        [],
        ["Ana"],
    )  # Ben's new line unchecked it, or the check stayed Ana's
    store.undo(paths, direction="ltr")  # an undo brings back the earlier signatures as they were
    assert {t.region_id: t.by for t in store.load_edits(paths).translations} == {
        "r0003": None,
        "r0001": "Ana",
    }

    session = StudioSession(paths, direction="ltr", author="Cleo")
    session.set_translation("r0003", "Come on")
    session.save()
    assert {t.region_id: t.by for t in store.load_edits(paths).translations}["r0003"] == "Cleo"


# ---------------------------------------------------------------- the commands


def _workflow(cfg: Config, monkeypatch: pytest.MonkeyPatch, *args: str) -> Any:
    monkeypatch.setattr(workflow_cli, "get_config", lambda: cfg)
    return runner.invoke(workflow_cli.workflow_app, list(args))


def test_the_workflow_commands(cfg: Config, paths: ChapterPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    me = cfg.model_copy(update={"user": UserConfig(name="Ana")})
    done = _workflow(me, monkeypatch, "done", "S", "Chapter 1", "translated", "--note", "all lines in")
    assert done.exit_code == 0 and done.stdout.strip() == "Chapter 1: translated (1/5)"
    assert load_status(paths).events[-1].by == "Ana"
    handed = _workflow(me, monkeypatch, "hand", "S", "Chapter 1", "Ben")
    assert handed.stdout.strip() == "Chapter 1: translated (1/5) · with Ben"
    shown = _workflow(me, monkeypatch, "status", "S")
    assert shown.stdout.strip() == "Chapter 1: translated (1/5) · with Ben; next: proofread"
    history = _workflow(me, monkeypatch, "status", "S", "Chapter 1")
    assert (
        "done translated by Ana — all lines in" in history.stdout and "handed to Ben by Ana" in history.stdout
    )
    noted = _workflow(me, monkeypatch, "note", "S", "Chapter 1", "r0002", "Too formal?")
    assert noted.stdout.strip() == "n0001 on r0002"
    assert (
        _workflow(me, monkeypatch, "notes", "S", "Chapter 1").stdout.strip()
        == "n0001\tr0002\tToo formal? (Ana)"
    )
    left = _workflow(me, monkeypatch, "todo", "S", "Chapter 1", "--role", "translator")
    assert left.stdout.splitlines() == [
        "r0002\tpage 1\tnote n0001: Too formal?",  # an open note shows for every role
        "r0003\tpage 1\tno translation yet",
    ]
    assert (
        _workflow(me, monkeypatch, "resolve", "S", "Chapter 1", "n0001").stdout.strip() == "n0001: resolved"
    )
    assert _workflow(me, monkeypatch, "notes", "S", "Chapter 1").stdout.strip() == ""
    for bad in (
        ["done", "S", "Chapter 1", "printed"],
        ["todo", "S", "Chapter 1", "--role", "editor"],
        ["note", "S", "Chapter 1", "r0099", "x"],
        ["status", "S", "Chapter 9"],
    ):
        assert _workflow(me, monkeypatch, *bad).exit_code == 2, bad

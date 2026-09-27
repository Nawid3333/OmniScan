"""Tests for omniscan.edits.session — the desktop Studio's session on the edits.json layer (torch-free)."""

from __future__ import annotations

from pathlib import Path

import pytest

import omniscan.edits.session as session_module
from omniscan.core.config import Config, DetectConfig, PathsConfig
from omniscan.core.paths import ChapterPaths
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
from omniscan.edits.session import StudioRow, StudioSession


def region(rid: str, y0: int, text: str, order: int, slice_index: int = 0) -> Region:
    box = BBox(x0=10, y0=y0, x1=110, y1=y0 + 50)
    return Region(
        id=rid, slice_index=slice_index, kind="bubble_text", bbox=box, text=text, reading_order=order
    )


REGIONS = [
    region("r0001", 10, "안녕", 0),
    region("r0002", 200, "반가워", 1),
    region("r0003", 700, "가자", 0, 1),
]


@pytest.fixture
def paths(tmp_path: Path) -> ChapterPaths:
    paths = ChapterPaths(
        series="S",
        chapter="Chapter 1",
        raw_dir=tmp_path / "lib" / "S" / "Chapter 1",
        work_dir=tmp_path / "work" / "S" / "Chapter 1",
        output_dir=tmp_path / "out" / "S" / "Chapter 1",
        filtered_dir=tmp_path / "out" / "S" / "_filtered" / "Chapter 1",
    )
    SlicesArtifact(
        strip_width=200,
        strip_height=1000,
        bands=[],
        slices=[Slice(index=0, y0=0, y1=600), Slice(index=1, y0=600, y1=1000)],
    ).save(paths.artifact("slices.json"))
    RegionsArtifact(regions=REGIONS).save(paths.artifact("ocr.json"))
    FinalArtifact(
        judge_model="judge",
        lines=[
            FinalLine(region_id="r0001", text="Hi", decision="pick"),
            FinalLine(region_id="r0002", text="Glad", decision="pick"),
        ],
    ).save(paths.artifact("final.json"))
    return paths


def open_session(paths: ChapterPaths) -> StudioSession:
    return StudioSession(paths, direction="ltr")


def test_rows_show_the_machine_line_and_the_hand_line(paths: ChapterPaths) -> None:
    store.set_translation(paths, "r0002", "Nice to see you", direction="ltr")  # an earlier hand edit
    rows = open_session(paths).rows()
    assert rows == [
        StudioRow("r0001", 0, "bubble_text", "안녕", "Hi", "Hi", False),
        StudioRow("r0002", 0, "bubble_text", "반가워", "Glad", "Nice to see you", True),
        StudioRow("r0003", 1, "bubble_text", "가자", "", "", False),
    ]


def test_changes_stay_in_memory_until_save(paths: ChapterPaths) -> None:
    before = paths.artifact("ocr.json").read_bytes()
    session = open_session(paths)
    assert not session.dirty and session.save() == 0
    session.set_source("r0001", "안녕하세요")
    session.set_translation("r0001", "Hello there")
    session.set_translation("r0003", "Let's go")
    session.remove_region("r0002")
    assert session.dirty
    assert [(r.id, r.text) for r in session.regions()] == [("r0001", "안녕하세요"), ("r0003", "가자")]
    assert session.translations()["r0001"] == "Hello there"
    assert [row.edited for row in session.rows()] == [True, True]
    assert paths.artifact("ocr.json").read_bytes() == before and not paths.artifact("edits.json").exists()

    assert session.save() == 4 and not session.dirty
    ocr = {r.id: r for r in store.current_regions(paths)}
    assert set(ocr) == {"r0001", "r0003"} and ocr["r0001"].text == "안녕하세요"
    final = {line.region_id: line for line in FinalArtifact.load(paths.artifact("final.json")).lines}
    assert (final["r0001"].text, final["r0001"].decision, final["r0003"].text) == (
        "Hello there",
        "manual",
        "Let's go",
    )
    edits = store.load_edits(paths)
    assert [(e.region_id, e.text, e.deleted, e.auto_text) for e in edits.regions] == [
        ("r0001", "안녕하세요", False, "안녕"),
        ("r0002", None, True, "반가워"),
    ]
    again = open_session(paths)
    assert not again.dirty and [row.edited for row in again.rows()] == [True, True]


def test_setting_the_machine_line_again_clears_the_hand_line(paths: ChapterPaths) -> None:
    session = open_session(paths)
    session.set_translation("r0001", "Hey")
    session.set_translation("r0001", "Hi")  # back to the machine's line before saving: nothing to save
    session.set_source("r0002", "반가워")  # unchanged text
    assert not session.dirty
    session.set_translation("r0001", "Hey")
    session.save()
    reopened = open_session(paths)
    reopened.set_translation("r0001", "Hey")  # the saved hand line: no change
    assert not reopened.dirty
    reopened.set_translation("r0001", "Hi")
    assert reopened.translations()["r0001"] == "Hi" and reopened.save() == 1
    assert store.load_edits(paths).translations == []
    assert open_session(paths).rows()[0] == StudioRow("r0001", 0, "bubble_text", "안녕", "Hi", "Hi", False)


def test_unknown_regions_raise_key_error(paths: ChapterPaths) -> None:
    session = open_session(paths)
    for change in (
        lambda: session.set_source("r0009", "x"),
        lambda: session.set_translation("r0009", "x"),
        lambda: session.remove_region("r0009"),
    ):
        with pytest.raises(KeyError):
            change()


def test_a_chapter_before_ocr_is_edited_from_its_detected_regions(paths: ChapterPaths) -> None:
    detected = [r.model_copy(update={"text": ""}) for r in REGIONS]
    paths.artifact("ocr.json").unlink()
    paths.artifact("final.json").unlink()
    RegionsArtifact(regions=detected).save(paths.artifact("regions.json"))
    session = open_session(paths)
    assert session.has_regions and [row.source for row in session.rows()] == ["", "", ""]
    session.set_source("r0003", "가자")
    assert session.save() == 1
    assert [r.text for r in store.current_regions(paths)] == ["", "", "가자"]
    edit = store.load_edits(paths).regions[0]
    assert (edit.region_id, edit.text, edit.auto_text) == ("r0003", "가자", "")


def test_a_chapter_without_artifacts_has_nothing_to_edit(tmp_path: Path) -> None:
    empty = ChapterPaths("S", "C", tmp_path / "r", tmp_path / "w", tmp_path / "o", tmp_path / "f")
    session = StudioSession(empty, direction="ltr")
    assert not session.has_regions and session.rows() == [] and session.translations() == {}


def test_the_reading_direction_comes_from_the_series(
    paths: ChapterPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Config(
        paths=PathsConfig(library_root=paths.raw_dir.parents[1], work_root=paths.work_dir.parents[1]),
        detect=DetectConfig(reading_direction="rtl"),
    )
    monkeypatch.setattr(session_module, "get_config", lambda: cfg)
    seen: list[str] = []
    real = store.update_region

    def spy(*args: object, direction: store.Direction, **kwargs: object) -> Region:
        seen.append(direction)
        return real(*args, direction=direction, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(store, "update_region", spy)
    session = StudioSession(paths)
    session.set_source("r0001", "안녕!")
    session.save()
    assert seen == ["rtl"]


def test_issues_use_the_desktop_studios_checks(paths: ChapterPaths) -> None:
    pytest.importorskip("omniscan.studio.qa")  # the desktop Studio's QA module (PR #18)
    issues = open_session(paths).issues()
    assert [(issue.region_id, issue.kind) for issue in issues] == [("r0003", "untranslated")]

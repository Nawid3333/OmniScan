"""LibraryView tests (offscreen): series list, per-chapter stage states, double-click, refresh, and the chapter
file buttons (open / send a chapter project, export a contribution, export for and import from other tools)."""

from __future__ import annotations

import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QListWidgetItem, QTableWidget, QTableWidgetItem

from omniscan.core.config import Config, PathsConfig, ShareConfig, UserConfig
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import BBox, FinalArtifact, FinalLine, Region, RegionsArtifact
from omniscan.edits import store
from omniscan.gui.library_view import WORKFLOW_COLUMN, LibraryView
from omniscan.interchange import ballons
from omniscan.interchange.labelplus import LabelFile, write
from omniscan.share.contribution import export_log, read_archive
from omniscan.translate.on_demand import english_lines
from omniscan.workflow.status import load_status
from tests.fixtures.gui_library import CHAPTERS, SERIES, _artifacts, build_library
from tests.unit.test_share_contribution import make_chapter


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """Config over the synthetic fixture library."""
    return build_library(tmp_path / "lib")


def _cell(table: QTableWidget, row: int, column: int) -> QTableWidgetItem:
    """One chapter-table cell (asserted present)."""
    item = table.item(row, column)
    assert item is not None
    return item


def _series_item(view: LibraryView) -> QListWidgetItem:
    """The first series-list entry (asserted present)."""
    item = view.series_list.item(0)
    assert item is not None
    return item


def test_series_list_shows_counts(qapp: QApplication, cfg: Config) -> None:
    """The list holds one entry per series with its chapter count."""
    view = LibraryView(cfg)
    qapp.processEvents()

    assert view.series_list.count() == 1
    assert "FixtureSeries" in _series_item(view).text()
    assert view.series() == SERIES


def test_chapter_table_states(qapp: QApplication, cfg: Config) -> None:
    """Each chapter row shows its ten stage states (column 0 is the chapter name)."""
    view = LibraryView(cfg)
    qapp.processEvents()

    assert view.table.rowCount() == 4
    assert [_cell(view.table, 0, column + 1).text() for column in range(10)] == ["done"] * 10
    # Episode 02: translate failed (col 5), typeset stale (col 9), export not run (col 10)
    assert _cell(view.table, 1, 5).text() == "failed"
    assert _cell(view.table, 1, 9).text() == "stale"
    assert _cell(view.table, 1, 10).text() == "not run"
    # Episode 03: nothing ran; Episode 04: detect stale, inpaint failed
    assert all(_cell(view.table, 2, column + 1).text() == "not run" for column in range(10))
    assert _cell(view.table, 3, 3).text() == "stale"
    assert _cell(view.table, 3, 7).text() == "failed"
    assert _cell(view.table, 0, 0).text() == "Episode 01"
    assert ": 4 chapter(s)" in view.status_label.text()


def test_double_click_emits_chapter_opened(qapp: QApplication, cfg: Config) -> None:
    """Double-clicking a chapter row reports (series, chapter)."""
    opened: list[tuple[str, str]] = []
    view = LibraryView(cfg)
    qapp.processEvents()
    view.chapter_opened.connect(lambda series, chapter: opened.append((series, chapter)))

    view._on_chapter_double_click(1, 0)  # Episode 02's row
    assert opened == [(SERIES, "Episode 02")]


def test_empty_library_shows_a_hint(qapp: QApplication) -> None:
    """No series: the table stays empty and the status label explains."""
    empty = Config(
        paths=PathsConfig(library_root=Path("nowhere"), work_root=Path("w"), output_root=Path("o"))
    )
    view = LibraryView(empty)
    qapp.processEvents()

    assert view.series() is None
    assert view.table.rowCount() == 0
    assert "no series" in view.status_label.text()


def test_reconfigure_swaps_the_source(qapp: QApplication, tmp_path: Path) -> None:
    """A settings change reloads from the new config."""
    view = LibraryView(build_library(tmp_path / "a"))
    qapp.processEvents()
    second = build_library(tmp_path / "b")
    view.reconfigure(second)
    qapp.processEvents()

    assert view.series_list.count() == 1
    assert "FixtureSeries" in _series_item(view).text()


# ---------------------------------------------------------------------- chapter files


def _settle(qapp: QApplication, view: LibraryView, timeout: float = 10.0) -> None:
    """Pump the event loop until the view's chapter-file task is over (worker signals arrive queued)."""
    deadline = time.monotonic() + timeout
    qapp.processEvents()
    while view._busy:
        if time.monotonic() > deadline:
            pytest.fail("timed out waiting for the chapter-file task")
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()


def _select(view: LibraryView, chapter: str) -> None:
    """Select the chapter row named `chapter`."""
    row = next(r for r in range(view.table.rowCount()) if _cell(view.table, r, 0).text() == chapter)
    view.table.selectRow(row)


def _asking(answers: dict[str, Path | None]) -> Callable[..., Path | None]:
    """A file-dialog hook answering by dialog title; records the suggested name under "<title> name"."""

    def ask(title: str, *rest: str) -> Path | None:
        if len(rest) == 2:  # a save dialog: (suggested name, filter)
            answers[f"{title} name"] = Path(rest[0])
        return answers[title]

    return ask


def test_send_chapter_packs_the_selected_chapter(qapp: QApplication, cfg: Config, tmp_path: Path) -> None:
    """Send is off until a chapter is selected; it suggests the CLI's file name and packs that chapter."""
    answers: dict[str, Path | None] = {"Send chapter": tmp_path / "sent.omniscan"}
    view = LibraryView(cfg, ask_save=_asking(answers))
    qapp.processEvents()
    assert not view.send_button.isEnabled() and view.contribute_button.isEnabled()

    _select(view, CHAPTERS[1])
    assert view.chapter() == CHAPTERS[1] and view.send_button.isEnabled()
    view.send_button.click()
    _settle(qapp, view)

    assert answers["Send chapter name"] == Path(f"{SERIES} - {CHAPTERS[1]}.omniscan")
    assert view.status_label.text().startswith(f"Sent {SERIES} — {CHAPTERS[1]}: ")
    with zipfile.ZipFile(tmp_path / "sent.omniscan") as archive:
        assert "project.json" in archive.namelist()
        assert any(name.startswith("raw/") for name in archive.namelist())


def test_open_chapter_project_adds_it_and_asks_before_replacing(
    qapp: QApplication, cfg: Config, tmp_path: Path
) -> None:
    """A project opens into another library as a new chapter; opening it again replaces it only after a yes."""
    view = LibraryView(cfg, ask_save=_asking({"Send chapter": tmp_path / "ep1.omniscan"}))
    qapp.processEvents()
    _select(view, CHAPTERS[0])
    view.send_button.click()
    _settle(qapp, view)

    empty = Config(  # another user's install, with an empty library
        paths=PathsConfig(
            library_root=tmp_path / "b" / "lib",
            work_root=tmp_path / "b" / "work",
            output_root=tmp_path / "b" / "out",
        )
    )
    questions: list[str] = []
    replies = [False, True]
    receiver = LibraryView(
        empty,
        ask_open=_asking({"Open chapter project": tmp_path / "ep1.omniscan"}),
        confirm=lambda title, text: questions.append(title) is None and replies.pop(0),
    )
    qapp.processEvents()
    assert receiver.series() is None and not receiver.send_button.isEnabled()
    receiver.open_project_button.click()
    _settle(qapp, receiver)
    assert receiver.status_label.text() == f"Opened {SERIES} — {CHAPTERS[0]}"
    assert receiver.series() == SERIES and receiver.table.rowCount() == 1  # the library was re-read

    receiver.open_project_button.click()  # again: the chapter exists now
    _settle(qapp, receiver)
    assert questions == ["Replace chapter?"] and "exists already" in receiver.status_label.text()
    receiver.open_project_button.click()
    _settle(qapp, receiver)
    assert questions == ["Replace chapter?"] * 2
    assert receiver.status_label.text() == f"Replaced {SERIES} — {CHAPTERS[0]}"


def test_a_damaged_project_is_reported(qapp: QApplication, cfg: Config, tmp_path: Path) -> None:
    """A file that is not a chapter project leaves the library alone and says why."""
    bad = tmp_path / "bad.omniscan"
    bad.write_bytes(b"not a zip")
    view = LibraryView(cfg, ask_open=_asking({"Open chapter project": bad}))
    qapp.processEvents()
    view.open_project_button.click()
    _settle(qapp, view)
    assert view.status_label.text() == "bad.omniscan is not an OmniScan chapter project (not a zip file)"
    assert view.open_project_button.isEnabled()


def test_cancelled_dialogs_do_nothing(qapp: QApplication, cfg: Config) -> None:
    """Cancelling a file dialog starts no task."""
    answers: dict[str, Path | None] = {
        "Open chapter project": None,
        "Send chapter": None,
        "Export contribution": None,
    }
    view = LibraryView(
        cfg, ask_open=_asking(answers), ask_save=_asking(answers), confirm=lambda title, text: True
    )
    qapp.processEvents()
    before = view.status_label.text()
    _select(view, CHAPTERS[0])
    for button in (view.open_project_button, view.send_button, view.contribute_button):
        button.click()
        assert not view._busy
    assert view.status_label.text() == before


def test_export_contribution_writes_the_series_archive(qapp: QApplication, tmp_path: Path) -> None:
    """A series with a checked line exports it; one without corrections or checks writes nothing; an opted-out
    machine is told so."""
    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )
    chapter = make_chapter(cfg, "Chapter 1")
    answers: dict[str, Path | None] = {"Export contribution": tmp_path / "c.zip"}
    asked: list[str] = []
    replies = [False, True, True, True]  # the first export is declined at the consent question

    def consent(title: str, text: str) -> bool:
        asked.append(text)
        return replies.pop(0)

    view = LibraryView(cfg, ask_save=_asking(answers), confirm=consent)
    qapp.processEvents()

    view.contribute_button.click()
    _settle(qapp, view)
    assert "CC BY 4.0" in asked[0] and "receipt id" in asked[0] and "Export contribution name" not in answers

    view.contribute_button.click()
    _settle(qapp, view)
    assert "nothing exported" in view.status_label.text() and not (tmp_path / "c.zip").exists()

    store.set_checked(chapter, ["r0001"])
    view.contribute_button.click()
    _settle(qapp, view)
    assert str(answers["Export contribution name"]).startswith("omniscan-contribution-")
    status = view.status_label.text()
    assert (
        status.startswith("Exported 1 chapter(s), 1 page(s)")
        and "1 checked line to c.zip; receipt " in status
    )
    assert status.endswith("keep it until the project says where to send it")
    contribution = read_archive(tmp_path / "c.zip")
    assert contribution.chapters[0].pages[0].regions[0].checked and contribution.receipt in status
    assert [record.receipt for record in export_log(cfg.paths.work_root)] == [contribution.receipt]

    view.reconfigure(cfg.model_copy(update={"share": ShareConfig(enabled=False)}))
    view.contribute_button.click()
    _settle(qapp, view)
    assert "opted out of sharing" in view.status_label.text()


# ---------------------------------------------------------------------- other tools


def _translatable(cfg: Config) -> ChapterPaths:
    """Episode 03 with page geometry, two regions and one English line (not cleaned, not lettered)."""
    paths = SeriesPaths.from_config(cfg, SERIES).chapter(CHAPTERS[2])
    _artifacts(paths, {"ingest", "slice"})
    for index in range(3):  # the pages where a real library keeps them (the fixture's own sit in raw/)
        Image.new("RGB", (40, 100), (200, 200, 200)).save(paths.raw_dir / f"{index + 1:04d}.jpg", "JPEG")
    RegionsArtifact(
        regions=[
            Region(
                id="r0001",
                slice_index=0,
                kind="bubble_text",
                bbox=BBox(x0=5, y0=10, x1=35, y1=40),
                text="안녕",
            ),
            Region(
                id="r0002",
                slice_index=2,
                kind="bubble_text",
                bbox=BBox(x0=5, y0=220, x1=35, y1=260),
                text="뭐",
            ),
        ]
    ).save(paths.artifact("ocr.json"))
    FinalArtifact(
        judge_model="fake", lines=[FinalLine(region_id="r0001", text="Hello", decision="pick")]
    ).save(paths.artifact("final.json"))
    return paths


def test_a_chapter_goes_out_to_other_tools_and_their_lines_come_back(
    qapp: QApplication, cfg: Config, tmp_path: Path
) -> None:
    """LabelPlus, PSD and BallonsTranslator exports write the CLI's files; the imports take the lines back."""
    paths = _translatable(cfg)
    label_file = tmp_path / "ep3.txt"
    answers: dict[str, Path | None] = {
        "Export for LabelPlus": label_file,
        "Import a LabelPlus file": label_file,
    }
    folders = {
        "Export layered PSD pages": tmp_path / "psd",
        "Export a BallonsTranslator project": tmp_path / "bt",
    }
    suggested: list[Path] = []
    view = LibraryView(
        cfg,
        ask_save=_asking(answers),
        ask_open=_asking(answers),
        ask_folder=lambda title, start: suggested.append(start) or folders[title],
    )
    qapp.processEvents()
    assert not view.export_button.isEnabled() and not view.import_button.isEnabled()
    _select(view, CHAPTERS[2])
    assert view.export_button.isEnabled() and view.import_button.isEnabled()

    view.export_labelplus()
    _settle(qapp, view)
    output = cfg.paths.output_root / SERIES
    assert answers["Export for LabelPlus name"] == output / "_labelplus" / f"{CHAPTERS[2]}.txt"
    assert view.status_label.text() == "Wrote 1 label(s) on 3 page(s) to ep3.txt"  # r0002 has no English yet
    label_file.write_text(label_file.read_text("utf-8-sig").replace("Hello", "Hi there"), "utf-8-sig")
    view.import_labelplus()
    _settle(qapp, view)
    assert view.status_label.text().startswith(f"Imported 1 English line(s) into {CHAPTERS[2]} (")
    assert english_lines(paths)["r0001"] == "Hi there"

    view.export_psd()
    _settle(qapp, view)
    assert suggested[-1] == output / "_psd" / CHAPTERS[2]
    assert sorted(p.name for p in (tmp_path / "psd").iterdir()) == ["0001.psd", "0002.psd", "0003.psd"]
    assert view.status_label.text().endswith("(no lettering yet: run typeset for the text layers)")

    view.export_ballons()
    _settle(qapp, view)
    assert suggested[-1] == output / "_ballons" / CHAPTERS[2]
    assert view.status_label.text().startswith("Wrote 2 text block(s) on 3 page(s)")
    answers["Import a BallonsTranslator project"] = ballons.project_file(tmp_path / "bt")
    view.import_ballons()
    _settle(qapp, view)
    assert view.status_label.text().startswith(
        f"Imported 0 English line(s) into {CHAPTERS[2]} (1 already the same"
    )


def test_an_import_before_ingest_and_cancelled_tool_dialogs_are_harmless(
    qapp: QApplication, cfg: Config, tmp_path: Path
) -> None:
    """A chapter with no page geometry refuses with the CLI's reason; cancelled dialogs start nothing."""
    label_file = tmp_path / "x.txt"
    label_file.write_text(write(LabelFile(pages={"0001.jpg": []})), "utf-8-sig")  # valid, with no labels
    answers: dict[str, Path | None] = {
        "Import a LabelPlus file": label_file,
        "Export for LabelPlus": None,
        "Import a BallonsTranslator project": None,
    }
    view = LibraryView(
        cfg,
        ask_open=_asking(answers),
        ask_save=_asking(answers),
        ask_open_many=lambda *_: [],
        ask_folder=lambda *_: None,
    )
    qapp.processEvents()
    _select(view, CHAPTERS[2])
    before = view.status_label.text()
    for action in (
        view.export_labelplus,
        view.export_psd,
        view.export_ballons,
        view.import_ballons,
        view.import_mit,
    ):
        action()
        assert not view._busy
    assert view.status_label.text() == before
    view.import_labelplus()
    _settle(qapp, view)
    assert view.status_label.text() == "ingest.json not found — run the ingest stage first"


# ---------------------------------------------------------------------- group workflow (#38)


def test_the_workflow_column_and_menu(qapp: QApplication, cfg: Config) -> None:
    """Mark a chapter's steps done and hand it over from the Library; the Workflow column follows."""
    me = cfg.model_copy(update={"user": UserConfig(name="Ana")})
    texts = iter(["Ben", "please clean page 2"])
    view = LibraryView(me, ask_text=lambda title, label, preset: next(texts))
    qapp.processEvents()
    chapter = CHAPTERS[0]
    _select(view, chapter)
    row = next(r for r in range(view.table.rowCount()) if _cell(view.table, r, 0).text() == chapter)
    assert (
        _cell(view.table, row, WORKFLOW_COLUMN).text() == "not started" and view.workflow_button.isEnabled()
    )

    view.step_actions["translated"].trigger()
    view.step_actions["proofread"].trigger()
    assert _cell(view.table, row, WORKFLOW_COLUMN).text() == "proofread (2/5)"
    view.workflow_menu.aboutToShow.emit()
    assert view.step_actions["proofread"].isChecked() and not view.step_actions["cleaned"].isChecked()
    view.step_actions["proofread"].trigger()  # untick: not done any more
    assert _cell(view.table, row, WORKFLOW_COLUMN).text() == "translated (1/5)"

    view.hand_over()
    assert _cell(view.table, row, WORKFLOW_COLUMN).text() == "translated (1/5) · with Ben"
    assert view.status_label.text().endswith("send it with Send chapter…")
    status = load_status(SeriesPaths.from_config(cfg, SERIES).chapter(chapter))
    assert (status.events[-1].to, status.events[-1].by, status.events[-1].note) == (
        "Ben",
        "Ana",
        "please clean page 2",
    )

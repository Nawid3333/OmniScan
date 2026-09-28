"""LibraryView tests (offscreen): series list, per-chapter stage states, double-click, refresh, and the chapter
file buttons (open / send a chapter project, export a contribution)."""

from __future__ import annotations

import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QListWidgetItem, QTableWidget, QTableWidgetItem

from omniscan.core.config import Config, PathsConfig, ShareConfig
from omniscan.edits import store
from omniscan.gui.library_view import LibraryView
from omniscan.share.contribution import read_archive
from tests.fixtures.gui_library import CHAPTERS, SERIES, build_library
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
    view = LibraryView(cfg, ask_open=_asking(answers), ask_save=_asking(answers))
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
    view = LibraryView(cfg, ask_save=_asking(answers))
    qapp.processEvents()

    view.contribute_button.click()
    _settle(qapp, view)
    assert "nothing exported" in view.status_label.text() and not (tmp_path / "c.zip").exists()

    store.set_checked(chapter, ["r0001"])
    view.contribute_button.click()
    _settle(qapp, view)
    assert str(answers["Export contribution name"]).startswith("omniscan-contribution-")
    assert view.status_label.text().startswith("Exported 1 chapter(s), 1 page(s)")
    assert view.status_label.text().endswith("1 checked line to c.zip")
    assert read_archive(tmp_path / "c.zip").chapters[0].pages[0].regions[0].checked

    view.reconfigure(cfg.model_copy(update={"share": ShareConfig(enabled=False)}))
    view.contribute_button.click()
    _settle(qapp, view)
    assert "opted out of sharing" in view.status_label.text()

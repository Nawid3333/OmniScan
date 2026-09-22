"""LibraryView tests (offscreen): series list, per-chapter stage states, double-click, refresh."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QListWidgetItem, QTableWidget, QTableWidgetItem

from omniscan.core.config import Config, PathsConfig
from omniscan.gui.library_view import LibraryView
from tests.fixtures.gui_library import SERIES, build_library


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

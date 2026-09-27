"""ReaderView tests (offscreen): chapter navigation, jump-to-slice, zoom, side selector."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget

from omniscan.core.config import Config
from omniscan.gui.reader_view import ReaderView
from tests.fixtures.gui_library import SERIES, build_library


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """Config over the synthetic fixture library."""
    return build_library(tmp_path / "lib")


def _shown(qapp: QApplication, cfg: Config) -> ReaderView:
    """A reader shown and sized so the strip views have real viewports."""
    view = ReaderView(cfg)
    view.resize(600, 500)
    view.show()
    qapp.processEvents()
    return view


def _pane(view: ReaderView, index: int) -> QWidget:
    """One compare-view pane (asserted present)."""
    widget = view.compare.splitter.widget(index)
    assert widget is not None
    return widget


def test_open_chapter_loads_the_compare_view(qapp: QApplication, cfg: Config) -> None:
    """Opening a chapter fills both sides and the switcher, and reports the current chapter."""
    view = _shown(qapp, cfg)

    assert view.open_chapter(SERIES, "Episode 01")
    assert view.current() == (SERIES, "Episode 01")
    assert view.compare.left.tiles() != ()
    assert view.compare.right.tiles() != ()  # Episode 01 has output
    assert view.chapter_combo.currentText() == "Episode 01"
    assert view.chapter_combo.count() == 4
    assert "Episode 01" in view.status_label.text()
    assert view.compare.right_caption.text() == "Output"


def test_prev_next_navigation(qapp: QApplication, cfg: Config) -> None:
    """Prev/next step through the series' chapters and enable/disable at the ends."""
    view = _shown(qapp, cfg)
    view.open_chapter(SERIES, "Episode 01")
    assert not view.prev_button.isEnabled()
    assert view.next_button.isEnabled()

    view.next_button.click()
    assert view.current() == (SERIES, "Episode 02")
    assert view.prev_button.isEnabled() and view.next_button.isEnabled()

    view.chapter_combo.setCurrentIndex(3)  # direct switcher jump (activated is for user clicks)
    view._open_at(3)
    assert view.current() == (SERIES, "Episode 04")
    assert not view.next_button.isEnabled()

    view.prev_button.click()
    assert view.current() == (SERIES, "Episode 03")


def test_open_chapter_unknown_returns_false(qapp: QApplication, cfg: Config) -> None:
    """An unknown chapter (or series) returns False and keeps the current chapter."""
    view = _shown(qapp, cfg)
    view.open_chapter(SERIES, "Episode 01")

    assert not view.open_chapter(SERIES, "Episode 99")
    assert view.current() == (SERIES, "Episode 01")
    assert not view.open_chapter("Other", "Episode 01")


def test_jump_to_slice_scrolls_both_sides(qapp: QApplication, cfg: Config) -> None:
    """Picking a jump target scrolls raw and output to that tile's strip y."""
    view = _shown(qapp, cfg)
    view.open_chapter(SERIES, "Episode 01")

    assert view.jump_combo.count() == 1 + 3  # placeholder + three output slices
    view._on_jump_activated(2)  # the second tile starts at strip y 100
    assert view.compare.left.strip_y() == pytest.approx(100.0)
    assert view.compare.right.strip_y() == pytest.approx(100.0)


def test_zoom_buttons_scale_the_master(qapp: QApplication, cfg: Config) -> None:
    """Zoom in/out multiply the current zoom of the compare view's master."""
    view = _shown(qapp, cfg)
    view.open_chapter(SERIES, "Episode 01")
    view.compare.left.set_zoom(1.0)  # fit width can already hit the 4.0 cap; start small
    view.compare.right.set_zoom(1.0)

    view.zoom_in_button.click()
    assert view.compare.left.zoom() == pytest.approx(1.25)
    view.zoom_out_button.click()
    assert view.compare.left.zoom() == pytest.approx(1.0, rel=1e-6)


def test_sides_combo_hides_a_pane(qapp: QApplication, cfg: Config) -> None:
    """Raw only / Output only collapse the other pane; Both shows both again."""
    view = _shown(qapp, cfg)
    view.open_chapter(SERIES, "Episode 01")

    view.sides_combo.setCurrentIndex(1)  # Raw only
    assert _pane(view, 0).isVisible()
    assert not _pane(view, 1).isVisible()

    view.sides_combo.setCurrentIndex(2)  # Output only
    assert not _pane(view, 0).isVisible()
    assert _pane(view, 1).isVisible()

    view.sides_combo.setCurrentIndex(0)  # Both
    assert _pane(view, 0).isVisible()
    assert _pane(view, 1).isVisible()


# ---------------------------------------------------------------------- reading mode


def test_reading_mode_shows_the_output_column_and_hides_the_chrome(
    qapp: QApplication, cfg: Config, tmp_path: Path
) -> None:
    """Read: output pane only, toolbars hidden, a capped column; Esc leaves and restores the view."""
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtTest import QTest

    from omniscan.gui.reader_view import READING_WIDTH_PX

    qsettings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    view = ReaderView(cfg, qsettings=qsettings)
    view.resize(1400, 500)
    view.show()
    changes: list[bool] = []
    view.reading_changed.connect(changes.append)
    view.set_reading(True)  # nothing open yet: stays off
    assert not view.is_reading()
    assert view.open_chapter(SERIES, "Episode 01")
    qapp.processEvents()

    view.read_button.click()
    qapp.processEvents()
    assert view.is_reading() and changes == [True]
    assert view.toolbar.isHidden() and view.compare.sync_checkbox.isHidden()
    assert _pane(view, 0).isHidden() and not _pane(view, 1).isHidden()
    strip = view.reading_strip()
    assert strip is view.compare.right
    assert strip.zoom() * 40 <= READING_WIDTH_PX + 0.5  # 40 px wide fixture strip: the column is capped

    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert not view.is_reading() and changes == [True, False]
    assert not view.toolbar.isHidden() and not _pane(view, 0).isHidden()


def test_reading_keys_page_through_and_the_position_is_remembered(
    qapp: QApplication, cfg: Config, tmp_path: Path
) -> None:
    """Page Down scrolls, N opens the next chapter, and reopening a chapter resumes where it was left."""
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtTest import QTest

    qsettings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    view = ReaderView(cfg, qsettings=qsettings)
    view.resize(300, 200)  # a small window so the strip scrolls
    view.show()
    assert view.open_chapter(SERIES, "Episode 01")
    view.set_reading(True)
    qapp.processEvents()
    strip = view.reading_strip()
    assert strip.max_strip_y() > 0

    QTest.keyClick(view, Qt.Key.Key_PageDown)
    left_at = strip.strip_y()
    assert left_at > 0
    QTest.keyClick(view, Qt.Key.Key_N)
    assert view.current() == (SERIES, "Episode 02")
    QTest.keyClick(view, Qt.Key.Key_P)
    assert view.current() == (SERIES, "Episode 01")
    assert abs(view.reading_strip().strip_y() - left_at) <= 1.0

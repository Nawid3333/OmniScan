"""Tests for the desktop app's appearance: OLED/light themes, accent colour, quick/standard/pro modes."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from omniscan.core.config import Config
from omniscan.gui.main_window import PAGES, MainWindow
from omniscan.gui.settings_view import SettingsView
from omniscan.gui.theme import (
    DEFAULT_ACCENT,
    Appearance,
    apply_theme,
    build_palette,
    load_appearance,
    save_appearance,
)


def _qsettings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)


def test_default_is_oled_black_with_the_default_accent(tmp_path: Path) -> None:
    appearance = load_appearance(_qsettings(tmp_path))
    assert appearance == Appearance("oled", DEFAULT_ACCENT, "standard")
    palette = build_palette(appearance)
    assert palette.color(QPalette.ColorRole.Window) == QColor("#000000")
    assert palette.color(QPalette.ColorRole.Base) == QColor("#000000")
    assert palette.color(QPalette.ColorRole.Highlight) == QColor(DEFAULT_ACCENT)


def test_light_theme_and_accent_round_trip(tmp_path: Path) -> None:
    qsettings = _qsettings(tmp_path)
    save_appearance(qsettings, Appearance("light", "#ff8800", "pro"))
    appearance = load_appearance(qsettings)
    assert appearance == Appearance("light", "#ff8800", "pro")
    palette = build_palette(appearance)
    assert palette.color(QPalette.ColorRole.Window).lightness() > 200
    assert palette.color(QPalette.ColorRole.Highlight) == QColor("#ff8800")


def test_invalid_saved_values_fall_back(tmp_path: Path) -> None:
    qsettings = _qsettings(tmp_path)
    qsettings.setValue("appearance/theme", "neon")
    qsettings.setValue("appearance/accent", "not a colour")
    qsettings.setValue("appearance/mode", "expert")
    assert load_appearance(qsettings) == Appearance()


def test_apply_theme_restyles_the_application(qapp: QApplication) -> None:
    apply_theme(qapp, Appearance("oled", "#00c853"))
    assert qapp.palette().color(QPalette.ColorRole.Highlight) == QColor("#00c853")
    assert "#00c853" in qapp.styleSheet()
    apply_theme(qapp, Appearance())


def test_appearance_tab_saves_applies_and_emits(qapp: QApplication, tmp_path: Path) -> None:
    qsettings = _qsettings(tmp_path)
    view = SettingsView(Config(), qsettings=qsettings)
    seen: list[Appearance] = []
    view.appearance_changed.connect(seen.append)
    view.theme_combo.setCurrentText("light")
    view.set_accent("#2979ff")
    assert load_appearance(qsettings) == Appearance("light", "#2979ff", "standard")
    assert seen[-1] == Appearance("light", "#2979ff", "standard")
    assert qapp.palette().color(QPalette.ColorRole.Highlight) == QColor("#2979ff")
    apply_theme(qapp, Appearance())


def test_quick_mode_hides_the_models_page(qapp: QApplication, tmp_path: Path) -> None:
    qsettings = _qsettings(tmp_path)
    window = MainWindow(Config(), qsettings=qsettings, config_loader=Config)
    models = PAGES.index("Models")
    assert not window.sidebar.isRowHidden(models)
    window.show_page(models)
    window.settings_view.mode_combo.setCurrentText("quick")
    assert window.sidebar.isRowHidden(models)
    assert window.sidebar.currentRow() == PAGES.index("Library")  # left the hidden page
    window.settings_view.mode_combo.setCurrentText("pro")
    assert not window.sidebar.isRowHidden(models)
    apply_theme(qapp, Appearance())

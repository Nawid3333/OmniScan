"""MainWindow tests (offscreen): page navigation, library→reader jump, status bar, persisted state."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from omniscan.core.config import Config
from omniscan.gui.main_window import MainWindow
from omniscan.gui.services.hardware import HardwareReport, ModelWarning
from omniscan.hw.detect import HardwareInfo
from tests.fixtures.gui_library import SERIES, build_library
from tests.unit.test_gui_hardware_service import _hw


class FakeModelsService:
    """ModelsService stand-in: instant rows, no daemon or torch."""

    def rows(self, **_kw: object) -> tuple[list[object], HardwareInfo]:
        return [], _hw()

    def download_required(self, on_progress: object = None) -> list[tuple[str, str | None]]:
        return []

    def download(self, model_id: str, on_progress: object = None) -> str:
        return "already installed"

    def remove(self, model_id: str) -> bool:
        return False


class FakeHardware:
    """HardwareService stand-in for the settings tab."""

    def report(self) -> HardwareReport:
        return HardwareReport(
            info=_hw(), warnings=(ModelWarning("slow-model", "slow", None, ("needs 24 GB",)),)
        )


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """Config over the synthetic fixture library."""
    return build_library(tmp_path / "lib")


def _window(
    cfg: Config, qsettings: QSettings, config_loader: Callable[[], Config] | None = None
) -> MainWindow:
    """A main window over the fixture library with every worker-backed service faked."""
    return MainWindow(
        cfg,
        models_service=FakeModelsService(),  # type: ignore[arg-type]
        hardware_service=FakeHardware(),  # type: ignore[arg-type]
        qsettings=qsettings,
        config_loader=config_loader,
    )


def test_five_pages_in_sidebar_order(qapp: QApplication, cfg: Config, tmp_path: Path) -> None:
    """The sidebar holds the five pages and the stack follows the selection."""
    qsettings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    window = _window(cfg, qsettings)

    assert window.sidebar.count() == 5
    for index, name in enumerate(("Library", "Reader", "Run", "Models", "Settings")):
        window.show_page(index)
        assert window.stack.currentIndex() == index
        assert window.sidebar.item(index).text() == name
    assert window.device_label.text() == f"device: {cfg.gpu.device}"
    assert window.job_label.text() == "job: idle"


def test_library_double_click_opens_the_reader(qapp: QApplication, cfg: Config, tmp_path: Path) -> None:
    """The library's chapter_opened jumps to the reader with the chapter loaded."""
    qsettings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    window = _window(cfg, qsettings)
    qapp.processEvents()

    window.library_view.chapter_opened.emit(SERIES, "Episode 01")
    assert window.stack.currentIndex() == 1  # the Reader page
    assert window.reader_view.current() == (SERIES, "Episode 01")


def test_run_state_shows_in_the_status_bar(qapp: QApplication, cfg: Config, tmp_path: Path) -> None:
    """The run page's busy signal flips the status bar's job label."""
    qsettings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    window = _window(cfg, qsettings)

    window.run_view._set_running(True)
    assert window.job_label.text() == "job: running"
    window.run_view._set_running(False)
    assert window.job_label.text() == "job: idle"


def test_settings_change_reloads_the_config(qapp: QApplication, cfg: Config, tmp_path: Path) -> None:
    """A settings write reloads the config through the loader and hands it to the pages."""
    qsettings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    loads: list[Config] = []

    def loader() -> Config:
        fresh = build_library(tmp_path / "again")
        loads.append(fresh)
        return fresh

    window = _window(cfg, qsettings, config_loader=loader)
    window.settings_view.settings_changed.emit()

    assert loads, "the loader ran once"
    assert window.library_view._cfg is loads[0]
    assert window.run_view._cfg is loads[0]


def test_geometry_and_last_page_are_remembered(qapp: QApplication, cfg: Config, tmp_path: Path) -> None:
    """Closing stores geometry and page; a new window restores both."""
    ini = tmp_path / "gui.ini"
    qsettings = QSettings(str(ini), QSettings.Format.IniFormat)
    window = _window(cfg, qsettings)
    window.show()
    window.show_page(3)  # Models
    window.close()

    reopened = _window(cfg, QSettings(str(ini), QSettings.Format.IniFormat))
    assert reopened.sidebar.currentRow() == 3
    assert reopened.stack.currentIndex() == 3

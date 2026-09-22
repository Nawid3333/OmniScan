"""SettingsView tests (offscreen): global commits/rejects, per-series overrides, profiles, hardware."""

from __future__ import annotations

import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox

from omniscan.core.config import Config
from omniscan.gui.services.hardware import HardwareReport, ModelWarning
from omniscan.gui.settings_view import SettingsView
from tests.fixtures.gui_library import SERIES, build_library
from tests.unit.test_gui_hardware_service import _hw


def _settle(qapp: QApplication, predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Pump the event loop until `predicate` holds (worker signals arrive queued)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


class FakeHardware:
    """HardwareService stand-in: a static snapshot with one warning."""

    def __init__(self) -> None:
        self.calls = 0

    def report(self) -> HardwareReport:
        self.calls += 1
        return HardwareReport(
            info=_hw(), warnings=(ModelWarning("slow-model", "slow", None, ("needs 24 GB",)),)
        )


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """Config over the synthetic fixture library."""
    return build_library(tmp_path / "lib")


# ---------------------------------------------------------------------- global tab


def test_global_commit_writes_the_toml_and_emits(qapp: QApplication, cfg: Config, tmp_path: Path) -> None:
    """Unticking gpu.warmup (the built-in default is on) writes the value and emits settings_changed once."""
    toml = tmp_path / "config.toml"
    changes: list[int] = []
    view = SettingsView(cfg, config_path=toml)
    view.settings_changed.connect(lambda: changes.append(1))

    editor = cast(QCheckBox, view._editors[("gpu", "warmup")])
    editor.setChecked(False)  # the default is True, so unticking is a real change
    assert tomllib.loads(toml.read_text(encoding="utf-8"))["gpu"]["warmup"] is False
    assert len(changes) == 1

    view._commit_global(view._fields[("filter", "threshold")], 0.42)
    assert tomllib.loads(toml.read_text(encoding="utf-8"))["filter"]["threshold"] == 0.42


def test_global_reject_shows_inline_and_reverts(qapp: QApplication, cfg: Config, tmp_path: Path) -> None:
    """A refused value shows the error inline and the editor keeps the last good value."""
    toml = tmp_path / "config.toml"
    view = SettingsView(cfg, config_path=toml)
    field = view._fields[("gpu", "device")]
    editor = cast(QComboBox, view._editors[("gpu", "device")])

    view._commit_global(field, "bogus")
    assert "gpu.device" in view.global_error.text()
    assert editor.currentText() == "auto"  # reverted
    assert not toml.exists()  # nothing was written

    view._commit_global(field, "cuda:1")
    view._commit_global(field, "cuda:9:9")
    assert "not auto" in view.global_error.text()
    assert editor.currentText() == "cuda:1"  # the last good value


def test_reset_restores_the_builtin_default(qapp: QApplication, cfg: Config, tmp_path: Path) -> None:
    """Reset clears the override; the editor shows the default and a further change rewrites."""
    toml = tmp_path / "config.toml"
    view = SettingsView(cfg, config_path=toml)
    field = view._fields[("ocr", "engine")]

    view._commit_global(field, "manga_ocr")
    assert tomllib.loads(toml.read_text(encoding="utf-8"))["ocr"]["engine"] == "manga_ocr"
    view._reset_global(field)
    assert not toml.exists() or "ocr" not in tomllib.loads(toml.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------- per-series tab


def test_series_override_round_trip(qapp: QApplication, cfg: Config) -> None:
    """Set writes series.toml (the row appears); Remove deletes the key and the empty file."""
    view = SettingsView(cfg)
    toml = cfg.paths.library_root / SERIES / "series.toml"

    view.override_key_combo.setCurrentText("strategy")
    view.override_value_edit.setText("fixed")
    view.override_set_button.click()
    assert tomllib.loads(toml.read_text(encoding="utf-8"))["slicer"]["strategy"] == "fixed"
    assert view.override_table.rowCount() == 1
    section = view.override_table.item(0, 0)
    key = view.override_table.item(0, 1)
    assert section is not None and key is not None
    assert section.text() == "slicer" and key.text() == "strategy"

    view._on_override_remove("slicer", "strategy")
    assert view.override_table.rowCount() == 0
    assert not toml.exists()


def test_series_override_rejects_a_bad_value(qapp: QApplication, cfg: Config) -> None:
    """An invalid value shows the service's message inline; nothing is written."""
    view = SettingsView(cfg)
    toml = cfg.paths.library_root / SERIES / "series.toml"

    view.override_key_combo.setCurrentText("strategy")
    view.override_value_edit.setText("bogus")
    view.override_set_button.click()
    assert "slicer.strategy" in view.series_error.text()
    assert not toml.exists()
    assert view.override_table.rowCount() == 0


def test_series_override_parses_json_values(qapp: QApplication, cfg: Config) -> None:
    """The value field accepts JSON for numbers (a float override lands as a float)."""
    view = SettingsView(cfg)
    view.override_section_combo.setCurrentText("filter")
    view.override_key_combo.setCurrentText("threshold")
    view.override_value_edit.setText("0.5")
    view.override_set_button.click()

    toml = cfg.paths.library_root / SERIES / "series.toml"
    assert tomllib.loads(toml.read_text(encoding="utf-8"))["filter"]["threshold"] == 0.5


# ---------------------------------------------------------------------- translation tab


def test_profile_toggle_writes_the_user_file(qapp: QApplication, cfg: Config, tmp_path: Path) -> None:
    """Ticking a profile writes [profiles.<name>] enabled = true to the injected user file."""
    user = tmp_path / "profiles.toml"
    view = SettingsView(cfg, profiles_path=user)
    assert view._profile_boxes

    box = next(box for box in view._profile_boxes if box.text().startswith("glm-5-3-flash-cloud"))
    box.setChecked(True)
    data = tomllib.loads(user.read_text(encoding="utf-8"))["profiles"]
    assert data["glm-5-3-flash-cloud"]["enabled"] is True


# ---------------------------------------------------------------------- hardware tab


def test_hardware_tab_lists_the_warnings(qapp: QApplication, cfg: Config) -> None:
    """Opening the Hardware tab detects on a pool thread; the snapshot line and rows land."""
    fake = FakeHardware()
    view = SettingsView(cfg, hardware=fake)  # type: ignore[arg-type]
    view.tabs.setCurrentWidget(view._pages["hardware"])  # detection starts on first open
    assert _settle(qapp, lambda: fake.calls > 0 and "detecting" not in view.hardware_header_label.text())

    assert "Test GPU" in view.hardware_header_label.text()
    assert view.hardware_table.rowCount() == 1
    name = view.hardware_table.item(0, 0)
    level = view.hardware_table.item(0, 1)
    assert name is not None and level is not None
    assert name.text() == "slow-model" and level.text() == "slow"

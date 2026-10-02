"""SettingsView tests (offscreen): global commits/rejects, per-series overrides, profiles, hardware."""

from __future__ import annotations

import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
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


def test_share_corrections_toggle_writes_the_machine_opt_out(
    qapp: QApplication, cfg: Config, tmp_path: Path
) -> None:
    """The Sharing section's Share corrections box writes `[share] enabled` (on by default)."""
    toml = tmp_path / "config.toml"
    view = SettingsView(cfg, config_path=toml)
    editor = cast(QCheckBox, view._editors[("share", "enabled")])
    assert editor.isChecked()
    editor.setChecked(False)
    assert tomllib.loads(toml.read_text(encoding="utf-8"))["share"]["enabled"] is False


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


# ---------------------------------------------------------------------- search


def _qsettings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)


def test_search_shows_only_matching_settings_and_tabs(
    qapp: QApplication, cfg: Config, tmp_path: Path
) -> None:
    """Typing filters rows by plain name, help, key or choices; tabs without a match are hidden."""
    view = SettingsView(cfg, hardware=FakeHardware(), qsettings=_qsettings(tmp_path))  # type: ignore[arg-type]
    device = view._editors[("gpu", "device")]
    library = view._editors[("paths", "library_root")]

    assert view.filter_settings("graphics") >= 1
    assert device.isVisibleTo(view) and not library.isVisibleTo(view)
    visible_tabs = [view.tabs.tabText(i) for i in range(view.tabs.count()) if view.tabs.isTabVisible(i)]
    assert visible_tabs == ["Global", "Hardware"]

    view.search_edit.setText("accent")  # only on the Appearance tab: it becomes the current one
    assert view.tabs.currentWidget() is view._pages["appearance"]
    assert view.accent_button.isVisibleTo(view)

    assert view.filter_settings("no such setting anywhere") == 0
    assert not view.search_empty.isHidden() and view.tabs.isHidden()

    view.search_edit.setText("")
    assert library.isVisibleTo(view) and not view.tabs.isHidden()
    assert all(view.tabs.isTabVisible(i) for i in range(view.tabs.count()))


def test_global_rows_use_plain_names_with_the_key_as_tooltip(qapp: QApplication, cfg: Config) -> None:
    """Each global setting reads as a plain name; the config key stays one hover away."""
    from PySide6.QtWidgets import QLabel

    view = SettingsView(cfg)
    labels = {label.text(): label.toolTip() for label in view._pages["global"].findChildren(QLabel)}
    assert labels["Graphics card"] == "gpu.device"
    assert labels["Hardware usage"] == "gpu.usage"


def test_optimise_for_this_pc_writes_the_plan(qapp: QApplication, cfg: Config, tmp_path: Path) -> None:
    """The Hardware tab shows the tuning plan; the button writes its settings and emits settings_changed once."""
    from omniscan.hw.profiles import profile
    from omniscan.hw.tune import plan_for

    class FakePlannedHardware(FakeHardware):
        def report(self) -> HardwareReport:
            self.calls += 1
            info = profile("gtx1650-laptop")
            return HardwareReport(info=info, warnings=(), plan=plan_for(info))

    toml = tmp_path / "config.toml"
    fake = FakePlannedHardware()
    view = SettingsView(cfg, hardware=fake, config_path=toml)  # type: ignore[arg-type]
    emitted: list[int] = []
    view.settings_changed.connect(lambda: emitted.append(1))
    assert not view.optimise_button.isEnabled()
    view.tabs.setCurrentWidget(view._pages["hardware"])
    assert _settle(qapp, lambda: view.optimise_button.isEnabled())
    assert "tier: light" in view.plan_label.text() and "GTX 1650" in view.hardware_header_label.text()
    assert view.apply_plan() == 7
    data = tomllib.loads(toml.read_text(encoding="utf-8"))
    assert data["gpu"]["device"] == "cuda:0" and data["gpu"]["vram_budget_gib"] == 2.5
    assert data["ocr"]["engine"] == "ppocr" and data["inpaint"]["lama"] is True
    assert emitted == [1] and view.plan_status_label.text().startswith("wrote 7 setting(s)")

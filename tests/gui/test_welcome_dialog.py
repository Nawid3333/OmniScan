"""WelcomeDialog tests (offscreen): hardware plan from a fake service, Ollama line, applying the plan."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from omniscan.gui.services.hardware import HardwareReport
from omniscan.gui.welcome_dialog import WelcomeDialog
from omniscan.hw.profiles import profile
from omniscan.hw.tune import plan_for


class FakeHardware:
    """HardwareService stand-in: the rtx3060 profile with its plan."""

    def report(self) -> HardwareReport:
        info = profile("rtx3060")
        return HardwareReport(info=info, warnings=(), plan=plan_for(info))


def _settle(qapp: QApplication, done: Callable[[], bool], seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not done() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert done()


def test_shows_the_plan_and_applies_it(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_toml = tmp_path / "config.toml"
    monkeypatch.setattr("omniscan.core.config.USER_TOML", user_toml)
    dialog = WelcomeDialog(FakeHardware(), ollama_check=lambda: "Ollama answers (fake)")  # type: ignore[arg-type]
    applied: list[int] = []
    dialog.plan_applied.connect(lambda: applied.append(1))
    assert dialog.apply_plan() == 0  # nothing detected yet
    _settle(qapp, lambda: dialog.optimise_button.isEnabled() and "fake" in dialog.ollama_label.text())
    assert "RTX 3060" in dialog.hardware_label.text() and "tier: gaming" in dialog.plan_label.text()
    assert dialog.apply_plan() == 7 and applied == [1]
    assert 'device = "cuda:0"' in user_toml.read_text(encoding="utf-8")
    assert not dialog.optimise_button.isEnabled() and dialog.status_label.text().startswith("wrote 7")


def test_a_failed_check_is_shown_not_raised(qapp: QApplication) -> None:
    def boom() -> str:
        raise RuntimeError("no network")

    dialog = WelcomeDialog(FakeHardware(), ollama_check=boom)  # type: ignore[arg-type]
    _settle(qapp, lambda: "check failed" in dialog.ollama_label.text())
    assert "RuntimeError: no network" in dialog.ollama_label.text()

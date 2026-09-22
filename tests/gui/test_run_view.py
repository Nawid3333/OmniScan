"""RunView tests (offscreen): progress from a fake pipeline, Cancel, and the step-mode gate."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from omniscan.core.config import Config
from omniscan.core.stage import StageOutcome
from omniscan.gui.run_view import RunView
from omniscan.pipeline.runner import GateEvent, PipelineResult
from tests.fixtures.gui_library import SERIES, build_library
from tests.unit.test_gui_runs_service import FakeRunner


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """Config over the synthetic fixture library."""
    return build_library(tmp_path / "lib")


def _settle(qapp: QApplication, predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Pump the event loop until `predicate` holds (worker signals arrive queued)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _slow_runner(
    cfg: Any,
    series: str,
    chapters: list[str] | None = None,
    *,
    stages: list[str],
    after_stage: Callable[[Any, StageOutcome], bool] | None = None,
    gate: Callable[[GateEvent], bool] | None = None,
    **_kw: object,
) -> PipelineResult:
    """Many short stages: gives the GUI time to press Cancel mid-run."""
    outcomes: dict[str, list[StageOutcome]] = {}
    for chapter in chapters or []:
        listed: list[StageOutcome] = []
        for i in range(40):
            outcome = StageOutcome(stage=f"stage{i}", status="done", seconds=0.01)
            assert after_stage is not None
            if not after_stage(SimpleNamespace(paths=SimpleNamespace(chapter=chapter)), outcome):
                return PipelineResult(outcomes=outcomes, failed={}, aborted="stopped")
            listed.append(outcome)
            outcomes[chapter] = listed
            time.sleep(0.02)  # let the GUI thread keep up (and get its Cancel in)
    return PipelineResult(outcomes=outcomes, failed={}, aborted=None)


def _gate_runner(
    cfg: Any,
    series: str,
    chapters: list[str] | None = None,
    *,
    stages: list[str],
    after_stage: Callable[[Any, StageOutcome], bool] | None = None,
    gate: Callable[[GateEvent], bool] | None = None,
    **_kw: object,
) -> PipelineResult:
    """Step-mode stand-in: offers the gate after every stage, `preview failed` on False."""
    outcomes: dict[str, list[StageOutcome]] = {}
    for chapter in chapters or []:
        listed: list[StageOutcome] = []
        for position, stage in enumerate(stages, start=1):
            outcome = StageOutcome(stage=stage, status="done", seconds=0.01)
            if after_stage is not None:
                after_stage(SimpleNamespace(paths=SimpleNamespace(chapter=chapter)), outcome)
            listed.append(outcome)
            outcomes[chapter] = listed
            assert gate is not None
            if not gate(
                GateEvent(
                    series=series,
                    chapter=chapter,
                    stage=stage,
                    outcome=outcome,
                    position=position,
                    total=len(stages),
                )
            ):
                return PipelineResult(outcomes=outcomes, failed={}, aborted="preview failed")
    return PipelineResult(outcomes=outcomes, failed={}, aborted=None)


def test_full_run_reports_progress_and_finishes(qapp: QApplication, cfg: Config) -> None:
    """Start runs the fake pipeline: live stage lines, a full progress bar, the form re-enabled."""
    runner = FakeRunner()
    view = RunView(cfg, run_fn=runner)  # type: ignore[arg-type]
    qapp.processEvents()

    view.start_button.click()
    assert view.is_running()
    assert not view.start_button.isEnabled()
    assert view.cancel_button.isEnabled()

    assert _settle(qapp, lambda: not view.is_running())
    assert runner.calls[0]["series"] == SERIES  # type: ignore[index]
    assert runner.calls[0]["mode"] == "auto"  # type: ignore[index]
    assert len(runner.calls[0]["stages"]) == 10  # type: ignore[index]
    assert "Episode 01 — ingest: done" in view.log.toPlainText()
    assert view.progress.maximum() == 4 * 10
    assert view.progress.value() == view.progress.maximum()
    assert "finished: ok" in view.log.toPlainText()
    assert view.start_button.isEnabled() and not view.cancel_button.isEnabled()


def test_cancel_stops_after_the_current_stage(qapp: QApplication, cfg: Config) -> None:
    """Cancel mid-run: the run ends with the runner's `stopped` summary."""
    view = RunView(cfg, run_fn=_slow_runner)  # type: ignore[arg-type]
    qapp.processEvents()

    view.start_button.click()
    assert _settle(qapp, lambda: view.is_running())
    time.sleep(0.15)  # a few stages run first
    qapp.processEvents()
    view.cancel_button.click()

    assert _settle(qapp, lambda: not view.is_running())
    assert "cancel requested" in view.log.toPlainText()
    assert "aborted=stopped" in view.log.toPlainText()
    assert view.start_button.isEnabled()


def test_step_mode_gate_continue_then_abort(qapp: QApplication, cfg: Config) -> None:
    """Step mode: the GUI blocks on Continue/Abort; Continue advances, Abort ends the run."""
    view = RunView(cfg, run_fn=_gate_runner)  # type: ignore[arg-type]
    qapp.processEvents()
    view.mode_combo.setCurrentText("Step")
    view.start_button.click()

    assert _settle(qapp, lambda: view.continue_button.isEnabled())
    assert "Step 1/10" in view.preview_title.text() and "ingest" in view.preview_title.text()
    view.continue_button.click()

    assert _settle(qapp, lambda: view.continue_button.isEnabled())
    assert "Step 2/10" in view.preview_title.text()
    view.abort_button.click()

    assert _settle(qapp, lambda: not view.is_running())
    assert "aborted=preview failed" in view.log.toPlainText()


def test_subset_mode_sends_the_checked_stages(qapp: QApplication, cfg: Config) -> None:
    """Subset mode: only the checked stages and the explicitly selected chapter reach the runner."""
    runner = FakeRunner()
    view = RunView(cfg, run_fn=runner)
    qapp.processEvents()
    view.mode_combo.setCurrentText("Subset")
    for name, box in view.stage_boxes.items():
        box.setChecked(name in ("ingest", "export"))
    view.all_checkbox.setChecked(False)  # unchecks every item
    view.chapter_list.item(0).setCheckState(Qt.CheckState.Checked)  # Episode 01 only

    view.start_button.click()
    assert _settle(qapp, lambda: not view.is_running())
    assert runner.calls[0]["stages"] == ["ingest", "export"]  # type: ignore[index]
    assert runner.calls[0]["chapters"] == ["Episode 01"]  # type: ignore[index]


def test_start_refuses_an_empty_chapter_selection(qapp: QApplication, cfg: Config) -> None:
    """No chapter selected and All off: the inline error explains, nothing starts."""
    view = RunView(cfg, run_fn=FakeRunner())
    qapp.processEvents()
    view.all_checkbox.setChecked(False)
    view._on_all_toggled(False)

    view.start_button.click()
    assert "select at least one chapter" in view.error_label.text()
    assert not view.is_running()


def test_start_refuses_an_empty_stage_selection(qapp: QApplication, cfg: Config) -> None:
    """Subset mode with zero stages: inline error, nothing starts."""
    view = RunView(cfg, run_fn=FakeRunner())
    qapp.processEvents()
    view.mode_combo.setCurrentText("Subset")
    for box in view.stage_boxes.values():
        box.setChecked(False)

    view.start_button.click()
    assert "select at least one stage" in view.error_label.text()
    assert not view.is_running()


def test_run_failure_surfaces_inline(qapp: QApplication, cfg: Config) -> None:
    """A run_fn that raises shows the error inline and re-enables the form."""

    def broken(cfg: object, series: str, **_kw: object) -> None:
        raise RuntimeError("no ollama daemon")

    view = RunView(cfg, run_fn=broken)  # type: ignore[arg-type]
    qapp.processEvents()
    view.start_button.click()
    assert _settle(qapp, lambda: not view.is_running())

    assert "RuntimeError" in view.error_label.text()
    assert "no ollama daemon" in view.error_label.text()
    assert view.start_button.isEnabled()

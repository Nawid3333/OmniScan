"""QueueView tests (offscreen): adding jobs, the job actions, and draining with a fake executor."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from omniscan.core.config import Config
from omniscan.gui.queue_view import QueueView
from omniscan.queue.store import Job, QueueStore, queue_db_path
from omniscan.queue.worker import Executor, PermanentJobError
from tests.fixtures.gui_library import CHAPTERS, SERIES, build_library


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """Config over the synthetic fixture library."""
    return build_library(tmp_path / "lib")


def _settle(qapp: QApplication, done: Callable[[], bool], seconds: float = 5.0) -> None:
    """Pump the event loop until `done()` (worker signals arrive queued)."""
    deadline = time.monotonic() + seconds
    while not done() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert done()


def _view(qapp: QApplication, cfg: Config, executor: Executor | None = None) -> QueueView:
    view = QueueView(cfg, executor_factory=(lambda cfg: executor) if executor is not None else None)
    view.resize(900, 500)
    view.show()
    qapp.processEvents()
    return view


def test_adding_a_job_from_the_form(qapp: QApplication, cfg: Config) -> None:
    view = _view(qapp, cfg)
    assert view.jobs() == [] and view.status_label.text() == "queue is empty"
    assert not view.run_button.isEnabled()
    view.series_combo.setCurrentText(SERIES)
    for name, box in view.stage_boxes.items():
        box.setChecked(name in ("typeset", "export"))
    view.all_checkbox.setChecked(False)
    assert view.add_job() is None and view.error_label.text() == "select chapters, or tick All chapters"
    view.chapter_list.item(0).setSelected(True)  # type: ignore[union-attr]
    view.priority_spin.setValue(5)
    job = view.add_job()
    assert job is not None and (job.series, job.stages, job.chapters, job.priority) == (
        SERIES,
        ("typeset", "export"),
        (CHAPTERS[0],),
        5,
    )
    assert view.table.rowCount() == 1 and view.table.item(0, 1).text() == "queued"  # type: ignore[union-attr]
    assert view.table.item(0, 6).text() == CHAPTERS[0]  # type: ignore[union-attr]
    assert view.run_button.isEnabled()
    with QueueStore(queue_db_path(cfg)) as store:
        assert [j.id for j in store.list()] == [job.id]


def test_pause_resume_cancel_retry_and_clear(qapp: QApplication, cfg: Config) -> None:
    with QueueStore(queue_db_path(cfg)) as store:
        store.add(SERIES, ["slice"])
        store.add(SERIES, ["detect"])
    view = _view(qapp, cfg)
    assert [j.status for j in view.jobs()] == ["queued", "queued"]
    view.table.selectRow(0)
    assert view.pause_button.isEnabled() and not view.resume_button.isEnabled()
    assert view._job_action("pause").status == "paused"  # type: ignore[union-attr]
    assert view.resume_button.isEnabled() and view.cancel_button.isEnabled()
    assert view._job_action("cancel").status == "cancelled"  # type: ignore[union-attr]
    assert view.retry_button.isEnabled()
    assert view._job_action("retry").status == "queued"  # type: ignore[union-attr]
    assert view._job_action("cancel").status == "cancelled"  # type: ignore[union-attr]
    assert view.clear_finished() == 1 and [j.id for j in view.jobs()] == [2]


def test_running_the_queue_drains_it_with_the_executor(qapp: QApplication, cfg: Config) -> None:
    executed: list[int] = []

    def executor(job: Job) -> None:
        executed.append(job.id)
        if "ocr" in job.stages:
            raise PermanentJobError("no OCR model here")

    with QueueStore(queue_db_path(cfg)) as store:
        store.add(SERIES, ["slice"])
        store.add(SERIES, ["ocr"], priority=1)
    busy: list[bool] = []
    view = _view(qapp, cfg, executor)
    view.busy_changed.connect(busy.append)
    assert view.run() and view.is_running() and not view.run() and not view.run_button.isEnabled()
    _settle(qapp, lambda: not view.is_running())
    assert executed == [2, 1]  # higher priority first
    assert busy == [True, False]
    assert view.status_label.text() == "done=1 failed=1 retried=0"
    assert [(j.id, j.status) for j in view.jobs()] == [(1, "done"), (2, "failed")]
    assert view.table.item(1, 7).text() == "no OCR model here"  # type: ignore[union-attr]
    assert view.clear_button.isEnabled() and not view.run_button.isEnabled()

"""QueueView: the job queue page — batch runs over many series and chapters, drained one job at a time.

The same queue as `omniscan queue` (one SQLite file, `work_root/queue.db`): add a job (stages over a series,
all or some chapters, priority, force), see every job's state, pause / resume / cancel / retry / clear, and
`Run queue`, which drains the queue on a worker thread with the pipeline executor until it is empty. Exactly
one worker per queue is supported, so do not run `omniscan queue run` or `omniscan serve` on the same library
while the page is draining.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omniscan.core.config import Config
from omniscan.gui.services import library
from omniscan.gui.theme import set_role
from omniscan.gui.workers import WorkerSignals, run_task
from omniscan.queue.store import KNOWN_STAGES, Job, QueueStore, queue_db_path
from omniscan.queue.worker import Executor, QueueSummary, run_queue

COLUMNS = ("Id", "Status", "Priority", "Tries", "Series", "Stages", "Chapters", "Error")
_ID_COL = 0
REFRESH_MS = 1500  # how often the table re-reads the queue while a drain is going

ExecutorFactory = Callable[[Config], Executor]


def _default_executor(cfg: Config) -> Executor:
    """The pipeline executor (imported lazily: it pulls the LLM client and the pipeline runner)."""
    from omniscan.queue.executor import stage_executor

    return stage_executor(cfg)


class QueueView(QWidget):
    """Add jobs to the queue, watch them, and drain the queue."""

    busy_changed = Signal(bool)  # a drain started / ended

    def __init__(
        self, cfg: Config, *, executor_factory: ExecutorFactory | None = None, parent: QWidget | None = None
    ) -> None:
        """Build the page; `executor_factory` replaces the pipeline executor (tests inject a fake)."""
        super().__init__(parent)
        self._cfg = cfg
        self._executor_factory = executor_factory or _default_executor
        self._task: WorkerSignals | None = None
        self._jobs: list[Job] = []
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)

        # ---- the form
        self.series_combo = QComboBox(self)
        self.all_checkbox = QCheckBox("All chapters", self)
        self.all_checkbox.setChecked(True)
        self.chapter_list = QListWidget(self)
        self.chapter_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.chapter_list.setEnabled(False)
        self.stage_group = QGroupBox("Stages", self)
        self.stage_boxes: dict[str, QCheckBox] = {}
        grid = QGridLayout(self.stage_group)
        for row, name in enumerate(KNOWN_STAGES):
            box = QCheckBox(name, self)
            box.setChecked(True)
            self.stage_boxes[name] = box
            grid.addWidget(box, row % 5, row // 5)
        self.force_checkbox = QCheckBox("Force re-run (ignore up-to-date manifests)", self)
        self.priority_spin = QSpinBox(self)
        self.priority_spin.setRange(-100, 100)
        self.priority_spin.setToolTip("Higher runs first")
        self.add_button = QPushButton("Add job", self)
        set_role(self.add_button, "primary")
        self.error_label = QLabel("", self)
        set_role(self.error_label, "error")
        self.error_label.setWordWrap(True)

        form = QWidget(self)
        form_layout = QVBoxLayout(form)
        form_layout.addWidget(QLabel("Series", self))
        form_layout.addWidget(self.series_combo)
        form_layout.addWidget(self.all_checkbox)
        form_layout.addWidget(self.chapter_list, 1)
        form_layout.addWidget(self.stage_group)
        form_layout.addWidget(self.force_checkbox)
        priority = QHBoxLayout()
        priority.addWidget(QLabel("Priority", self))
        priority.addWidget(self.priority_spin)
        priority.addStretch(1)
        form_layout.addLayout(priority)
        form_layout.addWidget(self.add_button)
        form_layout.addWidget(self.error_label)

        # ---- the jobs
        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        self.run_button = QPushButton("Run queue", self)
        set_role(self.run_button, "primary")
        self.run_button.setToolTip("Run every queued job, one at a time, until the queue is empty")
        self.pause_button = QPushButton("Pause", self)
        self.resume_button = QPushButton("Resume", self)
        self.cancel_button = QPushButton("Cancel", self)
        self.retry_button = QPushButton("Retry", self)
        self.clear_button = QPushButton("Clear finished", self)
        self.refresh_button = QPushButton("Refresh", self)
        self.status_label = QLabel("", self)

        jobs = QWidget(self)
        jobs_layout = QVBoxLayout(jobs)
        buttons = QHBoxLayout()
        for widget in (
            self.run_button,
            self.pause_button,
            self.resume_button,
            self.cancel_button,
            self.retry_button,
            self.clear_button,
            self.refresh_button,
        ):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        jobs_layout.addLayout(buttons)
        jobs_layout.addWidget(self.table, 1)
        jobs_layout.addWidget(self.status_label)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(form)
        splitter.addWidget(jobs)
        splitter.setSizes([1, 3])
        root = QVBoxLayout(self)
        root.addWidget(splitter, 1)

        self.series_combo.currentTextChanged.connect(self._on_series)
        self.all_checkbox.toggled.connect(lambda checked: self.chapter_list.setEnabled(not checked))
        self.add_button.clicked.connect(self.add_job)
        self.run_button.clicked.connect(self.run)
        self.pause_button.clicked.connect(lambda: self._job_action("pause"))
        self.resume_button.clicked.connect(lambda: self._job_action("resume"))
        self.cancel_button.clicked.connect(lambda: self._job_action("cancel"))
        self.retry_button.clicked.connect(lambda: self._job_action("retry"))
        self.clear_button.clicked.connect(self.clear_finished)
        self.refresh_button.clicked.connect(self.refresh)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self._reload_series()
        self.refresh()

    # ------------------------------------------------------------------ public

    def reconfigure(self, cfg: Config) -> None:
        """Swap the config (a settings change): the series list and the queue file may differ."""
        self._cfg = cfg
        self._reload_series()
        self.refresh()

    def jobs(self) -> list[Job]:
        """The jobs as last read from the queue."""
        return list(self._jobs)

    def selected_job(self) -> Job | None:
        """The job of the selected row."""
        row = self.table.currentRow()
        return self._jobs[row] if 0 <= row < len(self._jobs) else None

    def is_running(self) -> bool:
        """Whether a drain is going."""
        return self._task is not None

    def refresh(self) -> None:
        """Read the queue again and redraw the table (the selection stays on the same job id)."""
        selected = self.selected_job()
        try:
            with self._store() as store:
                self._jobs = store.list()
        except OSError as error:
            self.error_label.setText(f"cannot read the queue: {error}")
            self._jobs = []
        self.table.setRowCount(len(self._jobs))
        for row, job in enumerate(self._jobs):
            cells = (
                str(job.id),
                job.status,
                str(job.priority),
                f"{job.attempts}/{job.max_attempts}",
                job.series,
                ", ".join(job.stages),
                "all" if job.chapters is None else ", ".join(job.chapters),
                job.error or "",
            )
            for column, text in enumerate(cells):
                self.table.setItem(row, column, QTableWidgetItem(text))
            if selected is not None and job.id == selected.id:
                self.table.selectRow(row)
        counts: dict[str, int] = {}
        for job in self._jobs:
            counts[job.status] = counts.get(job.status, 0) + 1
        summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
        if not self._task:
            self.status_label.setText(summary or "queue is empty")
        self._update_buttons()

    def add_job(self) -> Job | None:
        """Queue the form's job; None (with the reason shown) when the form is incomplete."""
        series = self.series_combo.currentText()
        stages = [name for name in KNOWN_STAGES if self.stage_boxes[name].isChecked()]
        chapters = None if self.all_checkbox.isChecked() else self._checked_chapters()
        self.error_label.setText("")
        if not series:
            self.error_label.setText("pick a series")
            return None
        if not stages:
            self.error_label.setText("tick at least one stage")
            return None
        if chapters is not None and not chapters:
            self.error_label.setText("select chapters, or tick All chapters")
            return None
        try:
            with self._store() as store:
                job = store.add(
                    series,
                    stages,
                    chapters=chapters,
                    priority=self.priority_spin.value(),
                    force=self.force_checkbox.isChecked(),
                )
        except (OSError, ValueError) as error:
            self.error_label.setText(str(error))
            return None
        self.refresh()
        self.status_label.setText(f"queued job {job.id}: {series}, {len(stages)} stage(s)")
        return job

    def run(self) -> bool:
        """Drain the queue on a worker thread; False when a drain is already going."""
        if self._task is not None:
            return False
        cfg = self._cfg
        db_path = queue_db_path(cfg)
        executor = self._executor_factory(cfg)

        def drain(_progress: Any) -> QueueSummary:
            with QueueStore(db_path) as store:  # its own connection: sqlite objects stay on one thread
                return run_queue(store, executor)

        signals = run_task(drain)
        signals.finished.connect(self._on_drained)
        signals.failed.connect(self._on_drain_failed)
        self._task = signals
        self.status_label.setText("running the queue...")
        self._timer.start()
        self._update_buttons()
        self.busy_changed.emit(True)
        return True

    def clear_finished(self) -> int:
        """Delete the done and cancelled jobs; how many."""
        with self._store() as store:
            count = store.clear_finished()
        self.refresh()
        self.status_label.setText(f"cleared {count} job(s)")
        return count

    # ------------------------------------------------------------------ slots

    def _on_series(self, series: str) -> None:
        """Fill the chapter list for the chosen series."""
        self.chapter_list.clear()
        if series:
            self.chapter_list.addItems(library.list_chapter_names(self._cfg, series))

    def _on_drained(self, result: object) -> None:
        """The drain ended: show its counts."""
        self._end_drain()
        if isinstance(result, QueueSummary):
            text = f"done={result.done} failed={result.failed} retried={result.retried}"
            self.status_label.setText(text if result.finished else "queue is empty")

    def _on_drain_failed(self, text: str) -> None:
        """The drain raised."""
        self._end_drain()
        self.status_label.setText(f"queue run failed: {text}")

    # ------------------------------------------------------------------ internals

    def _store(self) -> QueueStore:
        """A queue store on this thread (opened per call: sqlite connections are per thread)."""
        return QueueStore(queue_db_path(self._cfg))

    def _reload_series(self) -> None:
        """Fill the series combo; its change hook fills the chapters."""
        current = self.series_combo.currentText()
        self.series_combo.blockSignals(True)
        try:
            self.series_combo.clear()
            self.series_combo.addItems(library.list_series(self._cfg))
            if current and self.series_combo.findText(current) >= 0:
                self.series_combo.setCurrentText(current)
        finally:
            self.series_combo.blockSignals(False)
        self._on_series(self.series_combo.currentText())

    def _checked_chapters(self) -> list[str]:
        """The selected chapters, in list order."""
        return [item.text() for item in self.chapter_list.selectedItems()]

    def _job_action(self, action: str) -> Job | None:
        """Pause, resume, cancel or retry the selected job; the store's refusal is shown."""
        job = self.selected_job()
        if job is None:
            return None
        try:
            with self._store() as store:
                changed: Job = getattr(store, action)(job.id)
        except (LookupError, ValueError) as error:
            self.status_label.setText(str(error))
            return None
        self.refresh()
        self.status_label.setText(f"job {changed.id}: {changed.status}")
        return changed

    def _end_drain(self) -> None:
        """A drain finished (or failed)."""
        self._task = None
        self._timer.stop()
        self.refresh()
        self.busy_changed.emit(False)

    def _update_buttons(self) -> None:
        """Enable the actions that make sense for the selected job and the drain state."""
        job = self.selected_job()
        idle = self._task is None
        status = job.status if job is not None else None
        self.run_button.setEnabled(idle and any(j.status == "queued" for j in self._jobs))
        self.pause_button.setEnabled(status == "queued")
        self.resume_button.setEnabled(status == "paused")
        self.cancel_button.setEnabled(status in ("queued", "paused"))
        self.retry_button.setEnabled(status in ("failed", "cancelled"))
        self.clear_button.setEnabled(idle and any(j.status in ("done", "cancelled") for j in self._jobs))

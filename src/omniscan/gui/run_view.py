"""RunView: the Run page — build a run, watch per-chapter/stage progress, cancel, answer step gates.

One run at a time: `Start` builds a `RunSpec`, validates it and starts a `RunWorker` (QThread) on
`RunController`; the injected `run_fn` (tests) replaces the real `run_pipeline`. Cancel stops after
the current stage (the runner's `after_stage` hook); in Step mode the gate blocks the worker until
Continue/Abort is pressed here.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from omniscan.core.config import Config
from omniscan.gui.run_worker import RunWorker
from omniscan.gui.services import library
from omniscan.gui.services.runs import (
    RunController,
    RunFn,
    RunMode,
    RunOutcome,
    RunSpec,
    StageUpdate,
    StepPreview,
    validate_spec,
)
from omniscan.pipeline.stages import STAGE_ORDER

MODE_OF: dict[str, RunMode] = {"Full": "full", "Subset": "subset", "Step": "step", "Auto": "auto"}
MODES = tuple(MODE_OF)
MODE_HINTS = {
    "Full": "every stage over every selected chapter",
    "Subset": "pick the stages below; runs only those",
    "Step": "pauses after each stage of the preview chapter",
    "Auto": "full run (auto-mode hook; same as Full today)",
}


class RunView(QWidget):
    """The run form plus its live progress, step-preview panel and log."""

    busy_changed = Signal(bool)  # True after Start, False when the run ends or fails

    def __init__(self, cfg: Config, *, run_fn: RunFn | None = None, parent: QWidget | None = None) -> None:
        """Build the page; `run_fn` overrides the pipeline entry point (tests inject it)."""
        super().__init__(parent)
        self._cfg = cfg
        self._run_fn = run_fn
        self._worker: RunWorker | None = None

        # ---- form
        self.series_combo = QComboBox(self)
        self.chapter_list = QListWidget(self)
        self.all_checkbox = QCheckBox("All chapters", self)
        self.all_checkbox.setChecked(True)
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(MODES)
        self.mode_hint_label = QLabel(MODE_HINTS["Full"], self)
        self.stage_group = QGroupBox("Stages (Subset mode)", self)
        self.stage_boxes: dict[str, QCheckBox] = {}
        self.lama_checkbox = QCheckBox("LaMa inpainting (inpaint_lama)", self)
        self.lama_checkbox.setChecked(True)
        self.force_checkbox = QCheckBox("Force re-run (ignore up-to-date manifests)", self)
        self.preview_combo = QComboBox(self)
        self.error_label = QLabel("", self)
        self.error_label.setStyleSheet("color: darkred;")
        self.error_label.setWordWrap(True)
        self.start_button = QPushButton("Start", self)
        self.cancel_button = QPushButton("Cancel", self)

        grid = QGridLayout(self.stage_group)
        for row, name in enumerate(STAGE_ORDER):
            box = QCheckBox(name, self)
            box.setChecked(True)
            self.stage_boxes[name] = box
            grid.addWidget(box, row % 5, row // 5)

        form = QWidget()
        form_layout = QVBoxLayout(form)
        form_layout.addWidget(QLabel("Series", self))
        form_layout.addWidget(self.series_combo)
        form_layout.addWidget(self.all_checkbox)
        form_layout.addWidget(self.chapter_list, 1)
        form_layout.addWidget(QLabel("Mode", self))
        form_layout.addWidget(self.mode_combo)
        form_layout.addWidget(self.mode_hint_label)
        form_layout.addWidget(self.stage_group)
        form_layout.addWidget(self.lama_checkbox)
        form_layout.addWidget(self.force_checkbox)
        form_layout.addWidget(QLabel("Preview chapter (Step mode)", self))
        form_layout.addWidget(self.preview_combo)
        form_layout.addWidget(self.error_label)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        form_layout.addLayout(buttons)

        # ---- live output
        self.progress = QProgressBar(self)
        self.progress.setValue(0)
        self.progress_label = QLabel("", self)
        self.preview_title = QLabel("", self)
        self.preview_summary = QLabel("", self)
        self.preview_summary.setWordWrap(True)
        self.preview_details = QPlainTextEdit(self)
        self.preview_details.setReadOnly(True)
        self.preview_details.setMaximumHeight(120)
        self.continue_button = QPushButton("Continue", self)
        self.abort_button = QPushButton("Abort", self)
        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)

        preview_group = QGroupBox("Step preview", self)
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.addWidget(self.preview_title)
        preview_layout.addWidget(self.preview_summary)
        preview_layout.addWidget(self.preview_details)
        preview_buttons = QHBoxLayout()
        preview_buttons.addStretch(1)
        preview_buttons.addWidget(self.continue_button)
        preview_buttons.addWidget(self.abort_button)
        preview_layout.addLayout(preview_buttons)

        output = QWidget()
        output_layout = QVBoxLayout(output)
        output_layout.addWidget(self.progress)
        output_layout.addWidget(self.progress_label)
        output_layout.addWidget(preview_group)
        output_layout.addWidget(self.log, 1)

        layout = QHBoxLayout(self)
        layout.addWidget(form, 1)
        layout.addWidget(output, 2)

        self.series_combo.currentTextChanged.connect(self.set_series)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.all_checkbox.toggled.connect(self._on_all_toggled)
        self.start_button.clicked.connect(self._on_start)
        self.cancel_button.clicked.connect(self._on_cancel)
        self.continue_button.clicked.connect(lambda: self._respond_preview("continue"))
        self.abort_button.clicked.connect(lambda: self._respond_preview("abort"))

        self._on_mode_changed(self.mode_combo.currentText())
        self._set_running(False)
        self._reload_series()

    # ------------------------------------------------------------------ state

    def set_series(self, series: str) -> None:
        """Fill the chapter list and preview combo with this series' chapters."""
        self.chapter_list.clear()
        self.preview_combo.clear()
        if not series:
            return
        self.all_checkbox.setChecked(True)
        for name in library.list_chapter_names(self._cfg, series):
            item = QListWidgetItem(name)
            item.setCheckState(Qt.CheckState.Checked)
            self.chapter_list.addItem(item)
            self.preview_combo.addItem(name)

    def is_running(self) -> bool:
        """True while a run is on its worker thread (Start disabled, Cancel enabled)."""
        return self._worker is not None

    def reconfigure(self, cfg: Config) -> None:
        """Swap the config source (a settings change) and reload the series list."""
        self._cfg = cfg
        if not self.is_running():
            self._reload_series()

    # ------------------------------------------------------------------ slots

    def _on_mode_changed(self, mode: str) -> None:
        """Enable the stage checklist in Subset mode and the preview combo in Step mode."""
        self.mode_hint_label.setText(MODE_HINTS.get(mode, ""))
        self.stage_group.setEnabled(mode == "Subset")
        self.preview_combo.setEnabled(mode == "Step")

    def _on_all_toggled(self, checked: bool) -> None:
        """Reflect the all-chapters switch on the list's checkboxes."""
        for index in range(self.chapter_list.count()):
            item = self.chapter_list.item(index)
            if item is not None:
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def _on_start(self) -> None:
        """Validate the form and start one run on its worker thread."""
        if self._worker is not None or not self.series_combo.currentText():
            return
        self.error_label.setText("")
        mode = MODE_OF[self.mode_combo.currentText()]  # the combo only offers MODES
        if self.all_checkbox.isChecked():
            chapters: tuple[str, ...] = ()
        else:
            chapters = self._checked_chapters()
            if not chapters:
                self.error_label.setText("select at least one chapter (or tick All chapters)")
                return
        if mode == "subset":
            stages = tuple(name for name in STAGE_ORDER if self.stage_boxes[name].isChecked())
            if not stages:
                self.error_label.setText("select at least one stage")
                return
        else:
            stages = STAGE_ORDER
        preview_chapter = self.preview_combo.currentText() if mode == "step" else None
        spec = RunSpec(
            series=self.series_combo.currentText(),
            mode=mode,
            chapters=chapters,
            stages=stages,
            lama=self.lama_checkbox.isChecked(),
            force=self.force_checkbox.isChecked(),
            preview_chapter=preview_chapter,
        )
        try:
            validate_spec(self._cfg, spec)
        except ValueError as error:
            self.error_label.setText(str(error))
            return
        self._start_worker(RunController(self._cfg, spec, run_fn=self._run_fn))
        self.log.appendPlainText(
            f"== {spec.mode} run: {spec.series} chapters={spec.chapters or 'all'} "
            f"stages={spec.stages} lama={spec.lama} force={spec.force}"
        )

    def _on_cancel(self) -> None:
        """Ask the worker to stop after the current stage."""
        if self._worker is None:
            return
        self._worker.cancel()
        self.log.appendPlainText("== cancel requested: stopping after the current stage")

    def _respond_preview(self, answer: str) -> None:
        """Release the step-mode gate; the buttons stay off until the next preview arrives."""
        if self._worker is not None:
            self._worker.respond_preview(answer)
            self.continue_button.setEnabled(False)
            self.abort_button.setEnabled(False)

    # ------------------------------------------------------------------ worker signals

    def _on_stage(self, update: StageUpdate) -> None:
        """One finished stage: advance the progress bar and append a log line."""
        self.progress.setMaximum(update.chapter_total * update.stage_total)
        self.progress.setValue((update.chapter_number - 1) * update.stage_total + update.stage_number)
        self.progress_label.setText(
            f"chapter {update.chapter_number}/{update.chapter_total} — "
            f"{update.stage}: {update.status} ({update.seconds:.1f}s)"
        )
        line = f"[{update.chapter_number}/{update.chapter_total}] {update.chapter} — {update.stage}: {update.status}"
        if update.error:
            line += f" — {update.error}"
        self.log.appendPlainText(line)

    def _on_preview(self, preview: StepPreview) -> None:
        """Show one step-mode preview and enable Continue/Abort until it is answered."""
        self.preview_title.setText(
            f"Step {preview.position}/{preview.total} — {preview.stage} ({preview.status})"
        )
        self.preview_summary.setText(preview.summary)
        self.preview_details.setPlainText("\n".join(preview.details))
        self.continue_button.setEnabled(True)
        self.abort_button.setEnabled(True)
        self.log.appendPlainText(f"[step {preview.position}/{preview.total}] {preview.stage} — waiting")

    def _on_finished(self, outcome: RunOutcome) -> None:
        """Run ended: log the summary and re-enable the form."""
        self._set_running(False)
        self.progress.setValue(self.progress.maximum())
        summary = f"== finished: {'ok' if outcome.ok else 'failed'}"
        if outcome.aborted:
            summary += f", aborted={outcome.aborted}"
        if outcome.failed:
            summary += ", failed: " + "; ".join(f"{chapter}: {error}" for chapter, error in outcome.failed)
        summary += f", chapters run={outcome.chapters_run}"
        self.log.appendPlainText(summary)

    def _on_failed(self, text: str) -> None:
        """The run raised instead of returning (config error, missing series...)."""
        self._set_running(False)
        self.error_label.setText(text)
        self.log.appendPlainText(f"!! {text}")

    # ------------------------------------------------------------------ internals

    def _checked_chapters(self) -> tuple[str, ...]:
        """The chapter-list items with a checked box, in display order."""
        checked: list[str] = []
        for index in range(self.chapter_list.count()):
            item = self.chapter_list.item(index)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                checked.append(item.text())
        return tuple(checked)

    def _start_worker(self, controller: RunController) -> None:
        """Create, wire and start the worker; freeze the form until the run ends."""
        worker = RunWorker(controller)
        worker.stage_reported.connect(self._on_stage)
        worker.preview_ready.connect(self._on_preview)
        worker.run_finished.connect(self._on_finished)
        worker.run_failed.connect(self._on_failed)
        worker.finished.connect(lambda: self._release_worker(worker))
        self._worker = worker
        self._set_running(True)
        worker.start()

    def _release_worker(self, worker: RunWorker) -> None:
        """Drop the finished worker (QThread.finished, after the last signal was queued)."""
        if self._worker is worker:
            self._worker = None

    def _set_running(self, running: bool) -> None:
        """Enable Start + the form when idle, Cancel when running (one run at a time)."""
        self.cancel_button.setEnabled(running)
        self.start_button.setEnabled(not running and self.series_combo.count() > 0)
        for widget in (
            self.series_combo,
            self.chapter_list,
            self.all_checkbox,
            self.mode_combo,
            self.stage_group,
            self.lama_checkbox,
            self.force_checkbox,
            self.preview_combo,
        ):
            widget.setEnabled(not running)
        self.continue_button.setEnabled(False)
        self.abort_button.setEnabled(False)
        self.busy_changed.emit(running)

    def _reload_series(self) -> None:
        """Fill the series combo once; the currentTextChanged hook fills the chapters."""
        self.series_combo.blockSignals(True)
        try:
            self.series_combo.clear()
            self.series_combo.addItems(library.list_series(self._cfg))
        finally:
            self.series_combo.blockSignals(False)
        if self.series_combo.currentText():
            self.set_series(self.series_combo.currentText())
        self.start_button.setEnabled(self.series_combo.count() > 0 and not self.is_running())

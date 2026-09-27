"""StudioView: the Translator Studio page — fix a chapter's source text and English by hand, then re-letter it.

Left: the raw strip with every text region outlined (click a box to select its row). Right: one row per region
(page, kind, source, English, issues), editable in place. Save writes the edits through
`omniscan.studio.session.StudioSession` (ocr.json, studio.json, corrections.jsonl); Check runs the automatic QA
pass; Re-letter runs the typeset and export stages for this chapter so the output shows the edits.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omniscan.core.config import Config
from omniscan.core.paths import SeriesPaths
from omniscan.gui.run_worker import RunWorker
from omniscan.gui.services import library
from omniscan.gui.services.runs import RunController, RunOutcome, RunSpec
from omniscan.gui.strip_view import StripView
from omniscan.gui.theme import set_role
from omniscan.studio.session import StudioRow, StudioSession

COLUMNS = ("Page", "Kind", "Source", "English", "Issues")
_SOURCE_COL, _ENGLISH_COL, _ISSUES_COL = 2, 3, 4
RELETTER_STAGES = ("typeset", "export")

ControllerFactory = Callable[[Config, RunSpec], Any]  # a RunController (tests pass a fake)


class StudioView(QWidget):
    """Edit one chapter's regions: source text, English line, removal; QA issues shown per row."""

    busy_changed = Signal(bool)  # a re-letter run started / ended

    def __init__(
        self,
        cfg: Config,
        *,
        controller_factory: ControllerFactory | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build the page; `controller_factory` replaces the real re-letter run (tests inject it)."""
        super().__init__(parent)
        self._cfg = cfg
        self._controller_factory = controller_factory or (lambda cfg, spec: RunController(cfg, spec))
        self._session: StudioSession | None = None
        self._rows: list[StudioRow] = []
        self._issues: dict[str, list[str]] = {}
        self._filling = False
        self._worker: RunWorker | None = None

        self.series_combo = QComboBox(self)
        self.chapter_combo = QComboBox(self)
        self.chapter_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.issues_only = QCheckBox("Only lines with issues", self)
        self.check_button = QPushButton("Check", self)
        self.check_button.setToolTip("Run the automatic quality check")
        self.remove_button = QPushButton("Remove box", self)
        self.remove_button.setToolTip("Delete the selected region (a false detection)")
        self.save_button = QPushButton("Save", self)
        self.reletter_button = QPushButton("Re-letter", self)
        set_role(self.reletter_button, "primary")
        self.reletter_button.setToolTip("Save, then redo lettering and export for this chapter")
        self.status_label = QLabel(self)

        self.strip = StripView(self)
        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_SOURCE_COL, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_ENGLISH_COL, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_ISSUES_COL, QHeaderView.ResizeMode.Stretch)

        bar = QHBoxLayout()
        for widget in (
            QLabel("Series", self),
            self.series_combo,
            QLabel("Chapter", self),
            self.chapter_combo,
            self.issues_only,
        ):
            bar.addWidget(widget)
        bar.addStretch(1)
        for widget in (self.check_button, self.remove_button, self.save_button, self.reletter_button):
            bar.addWidget(widget)

        splitter = QSplitter(self)
        splitter.addWidget(self.strip)
        splitter.addWidget(self.table)
        splitter.setSizes([2, 3])

        root = QVBoxLayout(self)
        root.addLayout(bar)
        root.addWidget(splitter, 1)
        root.addWidget(self.status_label)

        self.series_combo.currentTextChanged.connect(self._on_series)
        self.chapter_combo.currentTextChanged.connect(self._on_chapter)
        self.issues_only.toggled.connect(lambda _checked: self._apply_filter())
        self.check_button.clicked.connect(self.run_check)
        self.remove_button.clicked.connect(self.remove_selected)
        self.save_button.clicked.connect(self.save)
        self.reletter_button.clicked.connect(self.reletter)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.strip.overlay_clicked.connect(self.select_region)
        self._reload_series()

    # ------------------------------------------------------------------ public

    def reconfigure(self, cfg: Config) -> None:
        """Swap the config (a settings change) and reload the series list."""
        self._cfg = cfg
        self._reload_series()

    def open_chapter(self, series: str, chapter: str) -> bool:
        """Open one chapter for editing; False when the series or chapter is unknown."""
        if self.series_combo.findText(series) < 0:
            return False
        self.series_combo.setCurrentText(series)
        if self.chapter_combo.findText(chapter) < 0:
            return False
        self.chapter_combo.setCurrentText(chapter)
        self._open(series, chapter)
        return self._session is not None

    def session(self) -> StudioSession | None:
        """The open chapter's edit session."""
        return self._session

    def select_region(self, region_id: str) -> None:
        """Select a region's row and show it in the strip."""
        for row, item in enumerate(self._rows):
            if item.region_id == region_id:
                self.table.selectRow(row)
                return

    def run_check(self) -> int:
        """Run the QA pass, show each row's issues; returns how many issues were found."""
        if self._session is None:
            return 0
        issues = self._session.issues()
        self._issues = {}
        for issue in issues:
            self._issues.setdefault(issue.region_id, []).append(issue.message)
        self._fill_table()
        self.status_label.setText(f"{len(issues)} issue(s) in {len(self._issues)} line(s)")
        return len(issues)

    def remove_selected(self) -> None:
        """Remove the selected region from the chapter (applied on Save)."""
        row = self.table.currentRow()
        if self._session is None or not 0 <= row < len(self._rows):
            return
        self._session.remove_region(self._rows[row].region_id)
        self._refresh()

    def save(self) -> int:
        """Save the edits; returns how many corrections were logged."""
        if self._session is None:
            return 0
        count = self._session.save()
        self.status_label.setText(f"saved {count} correction(s)" if count else "nothing to save")
        self._update_buttons()
        return count

    def reletter(self) -> None:
        """Save, then run typeset + export for this chapter on a worker thread."""
        if self._session is None or self._worker is not None:
            return
        self.save()
        paths = self._session.paths
        spec = RunSpec(series=paths.series, mode="subset", chapters=(paths.chapter,), stages=RELETTER_STAGES)
        worker = RunWorker(self._controller_factory(self._cfg, spec))
        worker.run_finished.connect(self._on_run_finished)
        worker.run_failed.connect(self._on_run_failed)
        worker.finished.connect(lambda: self._release_worker(worker))
        self._worker = worker
        self.status_label.setText("re-lettering...")
        self._update_buttons()
        self.busy_changed.emit(True)
        worker.start()

    def is_running(self) -> bool:
        """Whether a re-letter run is going."""
        return self._worker is not None

    # ------------------------------------------------------------------ slots

    def _on_series(self, series: str) -> None:
        """Fill the chapter combo for the chosen series."""
        self.chapter_combo.blockSignals(True)
        try:
            self.chapter_combo.clear()
            if series:
                self.chapter_combo.addItems(library.list_chapter_names(self._cfg, series))
        finally:
            self.chapter_combo.blockSignals(False)
        self._on_chapter(self.chapter_combo.currentText())

    def _on_chapter(self, chapter: str) -> None:
        """Open the chosen chapter."""
        series = self.series_combo.currentText()
        if series and chapter:
            self._open(series, chapter)
        else:
            self._close()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """An edited cell: hand the new text to the session."""
        if self._filling or self._session is None or not 0 <= item.row() < len(self._rows):
            return
        region_id = self._rows[item.row()].region_id
        if item.column() == _SOURCE_COL:
            self._session.set_source(region_id, item.text())
        elif item.column() == _ENGLISH_COL:
            self._session.set_translation(region_id, item.text())
        self._rows = self._session.rows()
        self._update_buttons()

    def _on_selection(self) -> None:
        """Highlight the selected row's box and scroll the strip to it."""
        row = self.table.currentRow()
        selected = self._rows[row].region_id if 0 <= row < len(self._rows) else None
        self._draw_overlays(selected)
        if selected is not None:
            box = next(box for box in self.strip.overlays() if box[0] == selected)
            self.strip.set_strip_y(max(0.0, box[2] - 40.0))
        self._update_buttons()

    def _on_run_finished(self, outcome: RunOutcome) -> None:
        """Re-letter ended."""
        self.status_label.setText("re-lettered: output updated" if outcome.ok else "re-letter failed")

    def _on_run_failed(self, text: str) -> None:
        """Re-letter raised."""
        self.status_label.setText(f"re-letter failed: {text}")

    # ------------------------------------------------------------------ internals

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

    def _open(self, series: str, chapter: str) -> None:
        """Load a chapter's strip and its edit session."""
        paths = SeriesPaths.from_config(self._cfg, series).chapter(chapter)
        session = StudioSession(paths)
        if not session.has_regions:
            self._close()
            self.status_label.setText(f"{chapter}: no text regions yet; run detection and OCR first")
            return
        try:
            view = library.load_chapter_view(self._cfg, series, chapter)
        except (OSError, ValueError) as error:
            self._close()
            self.status_label.setText(f"cannot open {chapter}: {error}")
            return
        self.strip.set_tiles(view.raw, view.strip_width, view.strip_height)
        self._session = session
        self._issues = {}
        self._refresh()
        self.run_check()

    def _close(self) -> None:
        """No chapter open."""
        self._session = None
        self._rows = []
        self._issues = {}
        self.strip.set_overlays(())
        self._fill_table()

    def _refresh(self) -> None:
        """Re-read the rows from the session and redraw."""
        self._rows = self._session.rows() if self._session is not None else []
        self._fill_table()

    def _fill_table(self) -> None:
        """Rebuild the table from the rows and issues."""
        self._filling = True
        try:
            self.table.setRowCount(len(self._rows))
            for row, item in enumerate(self._rows):
                cells = (
                    str(item.page),
                    item.kind,
                    item.source,
                    item.english,
                    "; ".join(self._issues.get(item.region_id, [])),
                )
                for column, text in enumerate(cells):
                    cell = QTableWidgetItem(text)
                    if column not in (_SOURCE_COL, _ENGLISH_COL):
                        cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    if column == _ENGLISH_COL and item.machine and item.english != item.machine:
                        cell.setToolTip(f"machine: {item.machine}")
                    self.table.setItem(row, column, cell)
        finally:
            self._filling = False
        self._apply_filter()
        self._draw_overlays(None)
        self._update_buttons()

    def _apply_filter(self) -> None:
        """Hide rows without issues while "only issues" is ticked."""
        only = self.issues_only.isChecked()
        for row, item in enumerate(self._rows):
            self.table.setRowHidden(row, only and item.region_id not in self._issues)

    def _draw_overlays(self, selected: str | None) -> None:
        """Outline every remaining region on the strip."""
        if self._session is None:
            self.strip.set_overlays(())
            return
        boxes = [
            (region.id, region.bbox.x0, region.bbox.y0, region.bbox.x1, region.bbox.y1)
            for region in self._session.regions()
        ]
        self.strip.set_overlays(boxes, selected)

    def _update_buttons(self) -> None:
        """Enable the actions that make sense now."""
        is_open = self._session is not None
        idle = self._worker is None
        self.check_button.setEnabled(is_open)
        self.save_button.setEnabled(is_open and idle and self._session is not None and self._session.dirty)
        self.remove_button.setEnabled(is_open and idle and self.table.currentRow() >= 0)
        self.reletter_button.setEnabled(is_open and idle)

    def _release_worker(self, worker: RunWorker) -> None:
        """Drop the finished worker."""
        if self._worker is worker:
            self._worker = None
            self._update_buttons()
            self.busy_changed.emit(False)

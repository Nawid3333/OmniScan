"""LibraryView: the Library page — series with chapter counts and each chapter's stage states.

The data comes from `gui.services.library` (filesystem + manifests, no torch); this widget only
renders it and reports a double-clicked chapter back to the shell. Its file buttons pass chapters
between users and share corrections (`gui.services.chapter_files`): *Open chapter project…*,
*Send chapter…* and *Export contribution…*, each on a worker thread.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omniscan.core.config import Config
from omniscan.gui.services import chapter_files, library
from omniscan.gui.theme import TINTS
from omniscan.gui.workers import WorkerSignals, run_task

COLUMNS = ("Ingest", "Slice", "Detect", "OCR", "Translate", "Judge", "Inpaint", "LaMa", "Typeset", "Export")

# Stage-state cell backgrounds; "not run" keeps the default background.
STATE_COLORS: dict[str, QColor] = {
    "done": TINTS["positive"],
    "stale": TINTS["attention"],
    "failed": TINTS["negative"],
}

type AskOpen = Callable[[str, str], Path | None]  # (title, file filter) -> the chosen file, None = cancelled
type AskSave = Callable[[str, str, str], Path | None]  # (title, suggested name, file filter) -> the file
type Confirm = Callable[[str, str], bool]  # (title, question) -> yes


class LibraryView(QWidget):
    """Series list on the left, chapters-with-stages table on the right."""

    chapter_opened = Signal(str, str)  # (series, chapter) — double-click a chapter row

    def __init__(
        self,
        cfg: Config,
        parent: QWidget | None = None,
        *,
        ask_open: AskOpen | None = None,
        ask_save: AskSave | None = None,
        confirm: Confirm | None = None,
    ) -> None:
        """Build the page and load the series list once (`refresh` re-reads it); the file and question dialogs
        are injectable for tests."""
        super().__init__(parent)
        self._cfg = cfg
        self._series: str | None = None
        self._ask_open = ask_open or self._default_ask_open
        self._ask_save = ask_save or self._default_ask_save
        self._confirm = confirm or self._default_confirm
        self._task_signals: WorkerSignals | None = None  # keeps the running task's signals alive
        self._busy = False  # a chapter-file task is running

        self.series_list = QListWidget(self)
        self.table = QTableWidget(0, 1 + len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(("Chapter", *COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.status_label = QLabel(self)
        self.refresh_button = QPushButton("Refresh", self)
        self.open_project_button = QPushButton("Open chapter project…", self)
        self.open_project_button.setToolTip(
            "Add a chapter someone sent you (a .omniscan file) to your library"
        )
        self.send_button = QPushButton("Send chapter…", self)
        self.send_button.setToolTip("Pack the selected chapter, with all its work and edits, into one file")
        self.contribute_button = QPushButton("Export contribution…", self)
        self.contribute_button.setToolTip(
            "Write this series' corrections and checked lines to an archive you can share to improve OmniScan"
        )

        list_pane = QWidget()
        list_layout = QVBoxLayout(list_pane)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.addWidget(QLabel("Series", self))
        list_layout.addWidget(self.series_list, 1)

        table_layout = QVBoxLayout()
        table_layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        for button in (self.open_project_button, self.send_button, self.contribute_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self.refresh_button)
        table_layout.addLayout(buttons)
        table_pane = QWidget()
        table_pane.setLayout(table_layout)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(list_pane)
        self.splitter.addWidget(table_pane)
        self.splitter.setSizes([1, 3])

        root = QVBoxLayout(self)
        root.addWidget(self.splitter, 1)
        root.addWidget(self.status_label)

        self.series_list.currentItemChanged.connect(self._on_series)
        self.table.cellDoubleClicked.connect(self._on_chapter_double_click)
        self.refresh_button.clicked.connect(self.refresh)
        self.open_project_button.clicked.connect(self.open_project)
        self.send_button.clicked.connect(self.send_chapter)
        self.contribute_button.clicked.connect(self.export_contribution)
        self.table.itemSelectionChanged.connect(self._update_buttons)

        self.refresh()

    # ------------------------------------------------------------------ data

    def refresh(self) -> None:
        """Re-read the series list; keeps the current series selected when it still exists."""
        previous = self._series
        self.series_list.blockSignals(True)
        try:
            self.series_list.clear()
            for summary in library.series_summaries(self._cfg):
                item = QListWidgetItem(f"{summary.name}  ({summary.chapters} chapter(s))")
                item.setData(Qt.ItemDataRole.UserRole, summary.name)
                self.series_list.addItem(item)
        finally:
            self.series_list.blockSignals(False)

        self._series = None
        for row in range(self.series_list.count()):
            item = self.series_list.item(row)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == previous:
                self.series_list.setCurrentItem(item)  # fires _on_series when the row changed
                break
        if self._series is None and self.series_list.count():
            self.series_list.setCurrentRow(0)
        if self._series is None:  # no selection, or the selection did not change (no signal fired)
            if self.series_list.count():
                self._load_chapters()
            else:
                self.table.setRowCount(0)
                self.status_label.setText("no series found in the library root")
                self._update_buttons()

    def series(self) -> str | None:
        """The selected series (None when the library is empty)."""
        return self._series

    def chapter(self) -> str | None:
        """The selected chapter row's name (None when no row is selected)."""
        rows = self.table.selectionModel().selectedRows()
        item = self.table.item(rows[0].row(), 0) if rows else None
        return item.text() if item is not None else None

    # ------------------------------------------------------------------ chapter files

    def open_project(self) -> None:
        """Ask for a chapter project and unpack it into the library; an existing chapter is replaced only after
        a yes."""
        path = self._ask_open("Open chapter project", chapter_files.PROJECT_FILTER)
        if path is not None:
            self._run(f"Opening {path.name} …", lambda: chapter_files.open_project(self._cfg, path), path)

    def send_chapter(self) -> None:
        """Ask where to write the selected chapter's project and pack it there."""
        series, chapter = self._series, self.chapter()
        if series is None or chapter is None:
            return
        dest = self._ask_save(
            "Send chapter", chapter_files.project_name(series, chapter), chapter_files.PROJECT_FILTER
        )
        if dest is not None:
            self._run(
                f"Packing {chapter} …", lambda: chapter_files.send_chapter(self._cfg, series, chapter, dest)
            )

    def export_contribution(self) -> None:
        """Ask where to write the selected series' contribution archive and write it."""
        series = self._series
        if series is None:
            return
        try:
            name = chapter_files.contribution_name(self._cfg, series)
        except (OSError, ValueError) as exc:  # the install's salt cannot be made or is damaged
            self.status_label.setText(f"Export contribution: {exc}")
            return
        dest = self._ask_save("Export contribution", name, chapter_files.CONTRIBUTION_FILTER)
        if dest is not None:
            self._run(
                f"Exporting {series} …", lambda: chapter_files.export_contribution(self._cfg, series, dest)
            )

    def reconfigure(self, cfg: Config) -> None:
        """Reload from a new config (a settings change): swap the source and refresh."""
        self._cfg = cfg
        self.refresh()

    # ------------------------------------------------------------------ internals

    def _run(self, busy: str, task: Callable[[], str], project: Path | None = None) -> None:
        """Run a chapter-file task on a worker thread with the file buttons off; its line lands in the status
        label and the library is re-read. `project`: the file an Open is unpacking (asked to replace an existing
        chapter on FileExistsError)."""
        self._set_busy(True)
        self.status_label.setText(busy)
        signals = run_task(lambda _progress: task())
        self._task_signals = signals
        signals.finished.connect(self._on_task_done)
        signals.failed.connect(lambda error: self._on_task_failed(error, project))

    def _on_task_done(self, message: object) -> None:
        """A chapter-file task finished: show its line and re-read the library (a chapter may be new)."""
        self._set_busy(False)
        self.refresh()
        self.status_label.setText(str(message))

    def _on_task_failed(self, error: str, project: Path | None) -> None:
        """A chapter-file task failed: offer to replace an existing chapter, else show the error."""
        self._set_busy(False)
        kind, _, message = error.partition(": ")
        if (
            project is not None
            and kind == "FileExistsError"
            and self._confirm(
                "Replace chapter?", f"{message}.\nReplace its raw pages, work and finished pages?"
            )
        ):
            self._run(
                f"Replacing from {project.name} …",
                lambda: chapter_files.open_project(self._cfg, project, force=True),
            )
            return
        self.status_label.setText(message or error)

    def _set_busy(self, busy: bool) -> None:
        """Turn the file buttons off while a task runs (and back on per selection after)."""
        self._busy = busy
        self._update_buttons()

    def _update_buttons(self) -> None:
        """Enable each file button when it has what it needs: a series, a selected chapter, no running task."""
        idle = not self._busy
        self.open_project_button.setEnabled(idle)
        self.send_button.setEnabled(idle and self._series is not None and self.chapter() is not None)
        self.contribute_button.setEnabled(idle and self._series is not None)

    @staticmethod
    def _default_ask_open(title: str, file_filter: str) -> Path | None:
        """The stock open-file dialog; tests replace this hook."""
        chosen, _filter = QFileDialog.getOpenFileName(None, title, str(Path.home()), file_filter)
        return Path(chosen) if chosen else None

    @staticmethod
    def _default_ask_save(title: str, name: str, file_filter: str) -> Path | None:
        """The stock save-file dialog, starting at `name` in the user's home; tests replace this hook."""
        chosen, _filter = QFileDialog.getSaveFileName(None, title, str(Path.home() / name), file_filter)
        return Path(chosen) if chosen else None

    @staticmethod
    def _default_confirm(title: str, text: str) -> bool:
        """The stock question dialog; tests replace this hook."""
        return QMessageBox.question(None, title, text) == QMessageBox.StandardButton.Yes

    def _on_series(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None = None) -> None:
        """Load the chosen series' chapter table."""
        self._series = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        if self._series is not None:
            self._load_chapters()

    def _load_chapters(self) -> None:
        """Fill the table from `chapter_stage_states` for the current series."""
        if self._series is None:
            return
        states = library.chapter_stage_states(self._cfg, self._series)
        self.table.setRowCount(len(states))
        for row, entry in enumerate(states):
            self.table.setItem(row, 0, QTableWidgetItem(entry.chapter))
            for column, state in enumerate(entry.states, start=1):
                item = QTableWidgetItem(state)
                background = STATE_COLORS.get(state)
                if background is not None:
                    item.setBackground(QBrush(background))
                self.table.setItem(row, column, item)
        self._update_buttons()
        done = sum(1 for entry in states for state in entry.states if state == "done")
        self.status_label.setText(
            f"{self._series}: {len(states)} chapter(s), {done} of {len(states) * len(COLUMNS)} stage(s) done"
        )

    def _on_chapter_double_click(self, row: int, _column: int) -> None:
        """Emit (series, chapter) for the double-clicked row."""
        if self._series is None or not 0 <= row < self.table.rowCount():
            return
        name = self.table.item(row, 0)
        if name is not None:
            self.chapter_opened.emit(self._series, name.text())

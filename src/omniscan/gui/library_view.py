"""LibraryView: the Library page — series with chapter counts and each chapter's stage states.

The data comes from `gui.services.library` (filesystem + manifests, no torch); this widget only
renders it and reports a double-clicked chapter back to the shell.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omniscan.core.config import Config
from omniscan.gui.services import library

COLUMNS = ("Ingest", "Slice", "Detect", "OCR", "Translate", "Judge", "Inpaint", "LaMa", "Typeset", "Export")

# Stage-state cell backgrounds; "not run" keeps the default background.
STATE_COLORS: dict[str, QColor] = {
    "done": QColor(200, 230, 201),
    "stale": QColor(255, 249, 196),
    "failed": QColor(255, 205, 210),
}


class LibraryView(QWidget):
    """Series list on the left, chapters-with-stages table on the right."""

    chapter_opened = Signal(str, str)  # (series, chapter) — double-click a chapter row

    def __init__(self, cfg: Config, parent: QWidget | None = None) -> None:
        """Build the page and load the series list once (`refresh` re-reads it)."""
        super().__init__(parent)
        self._cfg = cfg
        self._series: str | None = None

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

        list_pane = QWidget()
        list_layout = QVBoxLayout(list_pane)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.addWidget(QLabel("Series", self))
        list_layout.addWidget(self.series_list, 1)

        table_layout = QVBoxLayout()
        table_layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
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

    def series(self) -> str | None:
        """The selected series (None when the library is empty)."""
        return self._series

    def reconfigure(self, cfg: Config) -> None:
        """Reload from a new config (a settings change): swap the source and refresh."""
        self._cfg = cfg
        self.refresh()

    # ------------------------------------------------------------------ internals

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

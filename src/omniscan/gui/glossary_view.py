"""GlossaryView: the Glossary page — one series' terms, edited by hand (add, correct, lock, reject, remove).

The same functions as `omniscan glossary add|set|lock|reject|remove` and the web Studio's Glossary tab
(`omniscan.glossary.edit`): words typed here become the user's (`origin = "user"`), which a later proposal pass
never overwrites; every change writes the series' glossary.yaml again, and the next translate run redoes only the
lines that hold a changed term. Source, English, aliases (comma-separated) and notes are edited in the table; the
status and type actions apply to every selected row.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import cast, get_args

from PySide6.QtCore import QItemSelectionModel, Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omniscan.core.config import Config
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import GlossaryEntry, TermType
from omniscan.glossary.edit import TermError, TermStatus, add_term, remove_term, set_status, update_term
from omniscan.glossary.store import GlossaryStore
from omniscan.gui.services import library
from omniscan.gui.theme import set_role

COLUMNS = ("Id", "Source", "English", "Type", "Status", "Origin", "Seen", "Aliases", "Notes")
(
    _ID_COL,
    _SOURCE_COL,
    _TARGET_COL,
    _TYPE_COL,
    _STATUS_COL,
    _ORIGIN_COL,
    _SEEN_COL,
    _ALIASES_COL,
    _NOTES_COL,
) = range(len(COLUMNS))
_EDITABLE = frozenset({_SOURCE_COL, _TARGET_COL, _ALIASES_COL, _NOTES_COL})
TERM_TYPES: tuple[TermType, ...] = get_args(TermType)
ALL_STATUSES = "All terms"
STATUSES: tuple[TermStatus, ...] = ("proposed", "locked", "rejected")
_TYPE_PLACEHOLDER = "Type…"


def _aliases(text: str) -> list[str]:
    """The aliases typed in one cell, separated by commas."""
    return [alias.strip() for alias in text.split(",") if alias.strip()]


class GlossaryView(QWidget):
    """List, filter and edit a series glossary by hand."""

    def __init__(self, cfg: Config, parent: QWidget | None = None) -> None:
        """Build the page over `cfg`'s library."""
        super().__init__(parent)
        self._cfg = cfg
        self._entries: list[GlossaryEntry] = []  # the whole glossary of the series, by id
        self._shown: list[GlossaryEntry] = []  # the table's rows (after the status filter and the search)
        self._filling = False

        # ---- series and filters
        self.series_combo = QComboBox(self)
        self.status_combo = QComboBox(self)
        self.status_combo.addItems([ALL_STATUSES, *STATUSES])
        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("Search source, English or aliases")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setMinimumWidth(260)
        self.count_label = QLabel(self)
        set_role(self.count_label, "muted")

        # ---- adding a term
        self.source_edit = QLineEdit(self)
        self.source_edit.setPlaceholderText("Source (e.g. 성진우)")
        self.target_edit = QLineEdit(self)
        self.target_edit.setPlaceholderText("English (e.g. Sung Jinwoo)")
        self.new_type_combo = QComboBox(self)
        self.new_type_combo.addItems(TERM_TYPES)
        self.new_type_combo.setCurrentText("other")
        self.add_button = QPushButton("Add term", self)
        set_role(self.add_button, "primary")
        self.add_button.setToolTip("Add the term, locked: every translation must use this English")

        # ---- actions on the selected rows
        self.lock_button = QPushButton("Lock", self)
        self.lock_button.setToolTip("Every translation must use these terms' English")
        self.reject_button = QPushButton("Reject", self)
        self.reject_button.setToolTip("Never use these terms, and never propose them again")
        self.propose_button = QPushButton("Back to proposed", self)
        self.propose_button.setToolTip("Only a suggestion to the translation model again")
        self.type_combo = QComboBox(self)
        self.type_combo.addItem(_TYPE_PLACEHOLDER)
        self.type_combo.addItems(TERM_TYPES)
        self.type_combo.setToolTip("Make the selected terms a person, place, skill, …")
        self.remove_button = QPushButton("Remove", self)
        self.remove_button.setToolTip(
            "Delete the selected terms (a later proposal pass may suggest them again)"
        )
        self.refresh_button = QPushButton("Refresh", self)
        self.status_label = QLabel(self)
        self.status_label.setWordWrap(True)

        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for column in (_SOURCE_COL, _TARGET_COL, _NOTES_COL):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

        top = QHBoxLayout()
        for widget in (QLabel("Series", self), self.series_combo, self.status_combo, self.search_edit):
            top.addWidget(widget)
        top.addStretch(1)
        top.addWidget(self.count_label)
        add_row = QHBoxLayout()
        add_row.addWidget(self.source_edit, 1)
        add_row.addWidget(self.target_edit, 1)
        add_row.addWidget(self.new_type_combo)
        add_row.addWidget(self.add_button)
        actions = QHBoxLayout()
        for widget in (
            self.lock_button,
            self.reject_button,
            self.propose_button,
            self.type_combo,
            self.remove_button,
        ):
            actions.addWidget(widget)
        actions.addStretch(1)
        actions.addWidget(self.refresh_button)
        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addLayout(add_row)
        root.addLayout(actions)
        root.addWidget(self.table, 1)
        root.addWidget(self.status_label)

        self.series_combo.currentTextChanged.connect(lambda _series: self.refresh())
        self.status_combo.currentTextChanged.connect(lambda _status: self._fill_table())
        self.search_edit.textChanged.connect(lambda _text: self._fill_table())
        self.add_button.clicked.connect(self.add)
        self.source_edit.returnPressed.connect(self.add)
        self.target_edit.returnPressed.connect(self.add)
        self.lock_button.clicked.connect(lambda: self.set_status("locked"))
        self.reject_button.clicked.connect(lambda: self.set_status("rejected"))
        self.propose_button.clicked.connect(lambda: self.set_status("proposed"))
        self.type_combo.activated.connect(self._on_type_picked)
        self.remove_button.clicked.connect(self.remove_selected)
        self.refresh_button.clicked.connect(self.refresh)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self._reload_series()

    # ------------------------------------------------------------------ public

    def reconfigure(self, cfg: Config) -> None:
        """Swap the config (a settings change): the library and its series may differ."""
        self._cfg = cfg
        self._reload_series()

    def entries(self) -> list[GlossaryEntry]:
        """The terms the table shows, in row order."""
        return list(self._shown)

    def selected_ids(self) -> list[int]:
        """The ids of the selected rows, in row order."""
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        return [cast(int, self._shown[row].id) for row in rows if row < len(self._shown)]

    def select_ids(self, ids: Sequence[int]) -> None:
        """Select the rows of these term ids (the others are deselected)."""
        wanted = set(ids)
        model = self.table.selectionModel()
        model.clearSelection()
        flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
        for row, entry in enumerate(self._shown):
            if entry.id in wanted:
                model.select(self.table.model().index(row, 0), flags)
        self._update_buttons()

    def refresh(self) -> None:
        """Read the series' glossary again and redraw the table (the selection stays on the same terms)."""
        keep = self.selected_ids()
        self._entries = []
        paths = self._paths()
        if paths is not None and paths.db.is_file():  # reading never creates the database
            try:
                with GlossaryStore(paths.db) as store:
                    self._entries = store.list()
            except (
                sqlite3.Error,
                OSError,
                ValueError,
            ) as error:  # a locked or broken database: shown, not raised
                self._show(f"cannot read the glossary: {error}", error=True)
        self._fill_table(keep)

    def add(self) -> GlossaryEntry | None:
        """Add the form's term (locked); None, with the reason shown, when it cannot be added."""
        paths = self._paths()
        if paths is None:
            self._show("pick a series first", error=True)
            return None
        term_type = cast(TermType, self.new_type_combo.currentText())
        try:
            entry = add_term(paths, self.source_edit.text(), self.target_edit.text(), type=term_type)
        except (TermError, FileNotFoundError) as error:
            self._show(str(error), error=True)
            return None
        self.source_edit.clear()
        self.target_edit.clear()
        self.source_edit.setFocus()
        if self.status_combo.currentText() not in (ALL_STATUSES, entry.status):
            self.status_combo.setCurrentText(ALL_STATUSES)  # never hide the term just added
        self.search_edit.clear()
        self.refresh()
        self.select_ids([cast(int, entry.id)])
        self._show(f"added {entry.source} → {entry.target} (locked)")
        return entry

    def set_status(self, status: TermStatus) -> int:
        """Lock, reject or re-propose the selected terms; how many changed."""
        ids = self.selected_ids()
        paths = self._paths()
        if not ids or paths is None:
            return 0
        try:
            changed = set_status(paths, ids, status)
        except KeyError as error:  # a term removed elsewhere (the CLI, the web Studio) meanwhile
            self._show(f"{error.args[0]} — refreshed", error=True)
            self.refresh()
            return 0
        self.refresh()
        self._show(f"{len(changed)} term(s) {status}")
        return len(changed)

    def set_type(self, term_type: TermType) -> int:
        """Give the selected terms one type (person, place, skill, …); how many changed."""
        ids = self.selected_ids()
        paths = self._paths()
        if not ids or paths is None:
            return 0
        changed = 0
        for entry_id in ids:
            try:
                update_term(paths, entry_id, type=term_type)
            except KeyError:
                continue  # removed elsewhere meanwhile
            changed += 1
        self.refresh()
        self._show(f"{changed} term(s) set to type {term_type}")
        return changed

    def remove_selected(self) -> int:
        """Delete the selected terms; how many were removed."""
        ids = self.selected_ids()
        paths = self._paths()
        if not ids or paths is None:
            return 0
        removed = 0
        for entry_id in ids:
            try:
                remove_term(paths, entry_id)
            except KeyError:
                continue  # already gone
            removed += 1
        self.refresh()
        self._show(f"removed {removed} term(s)")
        return removed

    # ------------------------------------------------------------------ slots

    def _on_type_picked(self, index: int) -> None:
        """A type was picked in the actions bar: apply it, then show the placeholder again."""
        if index > 0:
            self.set_type(cast(TermType, self.type_combo.itemText(index)))
        self.type_combo.setCurrentIndex(0)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """A cell was edited: store the new words (a refused change puts the old ones back).

        Only this row is redrawn: the edited item is still delivering its signal, so the table is not rebuilt here.
        """
        row, column = item.row(), item.column()
        if self._filling or column not in _EDITABLE or row >= len(self._shown):
            return
        entry = self._shown[row]
        paths = self._paths()
        if entry.id is None or paths is None:
            return
        text = item.text()
        try:
            if column == _SOURCE_COL:
                changed = update_term(paths, entry.id, source=text)
            elif column == _TARGET_COL:
                changed = update_term(paths, entry.id, target=text)
            elif column == _ALIASES_COL:
                changed = update_term(paths, entry.id, aliases=_aliases(text))
            else:
                changed = update_term(paths, entry.id, notes=text)
        except TermError as error:
            self._show(str(error), error=True)
            self._show_row(row)
            return
        except KeyError as error:  # removed elsewhere (the CLI, the web Studio) meanwhile
            self._show(f"{error.args[0]} — refreshed", error=True)
            QTimer.singleShot(0, self.refresh)
            return
        self._entries = [changed if e.id == changed.id else e for e in self._entries]
        self._shown[row] = changed
        self._show_row(row)
        self._show(f"saved {changed.source} → {changed.target}")

    # ------------------------------------------------------------------ internals

    def _paths(self) -> SeriesPaths | None:
        """The chosen series' paths, or None when the library has no series."""
        series = self.series_combo.currentText()
        return SeriesPaths.from_config(self._cfg, series) if series else None

    def _reload_series(self) -> None:
        """Fill the series combo (keeping the choice when it still exists), then the table."""
        current = self.series_combo.currentText()
        self.series_combo.blockSignals(True)
        try:
            self.series_combo.clear()
            self.series_combo.addItems(library.list_series(self._cfg))
            if current and self.series_combo.findText(current) >= 0:
                self.series_combo.setCurrentText(current)
        finally:
            self.series_combo.blockSignals(False)
        self.refresh()

    def _matches(self, entry: GlossaryEntry) -> bool:
        """Whether a term passes the status filter and the search text."""
        status = self.status_combo.currentText()
        if status != ALL_STATUSES and entry.status != status:
            return False
        needle = self.search_edit.text().strip().casefold()
        if not needle:
            return True
        return any(needle in text.casefold() for text in (entry.source, entry.target, *entry.aliases))

    def _fill_table(self, keep: Sequence[int] | None = None) -> None:
        """Redraw the rows from `_entries`, then select `keep` (default: the current selection)."""
        keep = self.selected_ids() if keep is None else keep
        self._shown = [entry for entry in self._entries if self._matches(entry)]
        self.table.clearContents()
        self.table.setRowCount(len(self._shown))
        for row in range(len(self._shown)):
            self._show_row(row)
        locked = sum(entry.status == "locked" for entry in self._entries)
        self.count_label.setText(
            f"{len(self._shown)} of {len(self._entries)} term(s), {locked} locked"
            if self._entries
            else "no terms yet"
        )
        self.select_ids(keep)

    def _show_row(self, row: int) -> None:
        """Write one term's cells (the id, type, status, origin and count are read-only)."""
        entry = self._shown[row]
        cells = (
            str(entry.id),
            entry.source,
            entry.target,
            entry.type,
            entry.status,
            entry.origin,
            str(entry.count),
            ", ".join(entry.aliases),
            entry.notes or "",
        )
        self._filling = True
        try:
            for column, text in enumerate(cells):
                item = self.table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    if column not in _EDITABLE:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(row, column, item)
                item.setText(text)
        finally:
            self._filling = False

    def _show(self, text: str, *, error: bool = False) -> None:
        """A message under the table (red for a refusal)."""
        self.status_label.setText(text)
        set_role(self.status_label, "error" if error else "")

    def _update_buttons(self) -> None:
        """Enable the row actions only while terms are selected."""
        selected = bool(self.table.selectionModel().selectedRows())
        for widget in (
            self.lock_button,
            self.reject_button,
            self.propose_button,
            self.type_combo,
            self.remove_button,
        ):
            widget.setEnabled(selected)
        self.add_button.setEnabled(self.series_combo.currentText() != "")

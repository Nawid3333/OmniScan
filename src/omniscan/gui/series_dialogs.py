"""The Studio's series dialogs: find and replace across chapters, and the consistency report.

ReplaceDialog previews what a find-and-replace rule changes — in the open chapter or the whole series, in the
English lines or the source texts — one ticked row per line, and records the ticked ones as hand edits (one undo
step per chapter; `gui.services.series_check`, the same as `omniscan edit replace`). ConsistencyDialog lists the
source lines translated in more than one way and the lines that miss a locked glossary term; double-clicking a row
asks the Studio to open that line (`open_line`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from omniscan.core.config import Config
from omniscan.edits.replace import Change, FindReplace, Target
from omniscan.gui.services import series_check
from omniscan.gui.services.series_check import Consistency
from omniscan.gui.theme import set_role

PlanFn = Callable[[Config, str, FindReplace, Target, Sequence[str] | None], list[Change]]
ApplyFn = Callable[[Config, str, Sequence[Change], Target], int]
TARGETS: tuple[tuple[str, Target], ...] = (("English lines", "english"), ("Source texts", "source"))
SCOPES = ("This chapter", "Whole series")
REPLACE_COLUMNS = ("Chapter", "Line", "Before", "After")
DIVERGENCE_COLUMNS = ("Source", "English", "Times", "Where")
MISS_COLUMNS = ("Chapter", "Line", "Term", "Should read", "English")


def _table(columns: Sequence[str], parent: QWidget, stretch: Sequence[int]) -> QTableWidget:
    """A read-only, row-selecting table with `columns`; the `stretch` columns take the spare width."""
    table = QTableWidget(0, len(columns), parent)
    table.setHorizontalHeaderLabels(list(columns))
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setWordWrap(True)
    table.verticalHeader().setVisible(False)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    for column in stretch:
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
    return table


class ReplaceDialog(QDialog):
    """Find and replace in a chapter or a whole series, previewed line by line before anything is written."""

    def __init__(
        self,
        cfg: Config,
        series: str,
        chapter: str,
        *,
        plan_fn: PlanFn | None = None,
        apply_fn: ApplyFn | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Search `chapter` of `series` (or the whole series); `plan_fn` / `apply_fn` replace the backends (tests)."""
        super().__init__(parent)
        self.setWindowTitle(f"Find and replace — {series}")
        self._cfg, self._series, self._chapter = cfg, series, chapter
        self._plan_fn: PlanFn = plan_fn or series_check.plan_replace
        self._apply_fn: ApplyFn = apply_fn or series_check.apply_replace
        self._changes: list[Change] = []
        self._target: Target = "english"
        self.applied = 0  # lines changed since the dialog opened

        self.find_edit = QLineEdit(self)
        self.find_edit.setPlaceholderText("Find")
        self.replace_edit = QLineEdit(self)
        self.replace_edit.setPlaceholderText("Replace with")
        self.regex_box = QCheckBox("Regular expression", self)
        self.word_box = QCheckBox("Whole words", self)
        self.case_box = QCheckBox("Match case", self)
        self.case_box.setChecked(True)
        self.case_box.setToolTip(
            "Off: any case, and the replacement takes each match's case (JINWOO → JIN-WOO)"
        )
        self.target_combo = QComboBox(self)
        self.target_combo.addItems([label for label, _ in TARGETS])
        self.scope_combo = QComboBox(self)
        self.scope_combo.addItems(SCOPES)
        self.preview_button = QPushButton("Preview", self)
        set_role(self.preview_button, "primary")
        self.table = _table(("", *REPLACE_COLUMNS), self, stretch=(3, 4))
        self.replace_button = QPushButton("Replace ticked", self)
        self.replace_button.setEnabled(False)
        self.close_button = QPushButton("Close", self)
        self.status_label = QLabel(self)
        self.status_label.setWordWrap(True)

        form = QGridLayout()
        form.addWidget(self.find_edit, 0, 0, 1, 3)
        form.addWidget(self.replace_edit, 1, 0, 1, 3)
        form.addWidget(self.regex_box, 2, 0)
        form.addWidget(self.word_box, 2, 1)
        form.addWidget(self.case_box, 2, 2)
        form.addWidget(self.target_combo, 3, 0)
        form.addWidget(self.scope_combo, 3, 1)
        form.addWidget(self.preview_button, 3, 2)
        buttons = QHBoxLayout()
        buttons.addWidget(self.status_label, 1)
        buttons.addWidget(self.replace_button)
        buttons.addWidget(self.close_button)
        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(self.table, 1)
        root.addLayout(buttons)
        self.resize(900, 520)

        self.preview_button.clicked.connect(self.preview)
        self.find_edit.returnPressed.connect(self.preview)
        self.replace_edit.returnPressed.connect(self.preview)
        self.replace_button.clicked.connect(self.replace_ticked)
        self.close_button.clicked.connect(self.accept)

    def preview(self) -> int:
        """List what the rule changes, every row ticked; how many lines. A bad rule is shown, not raised."""
        rule = FindReplace(
            find=self.find_edit.text(),
            replace=self.replace_edit.text(),
            regex=self.regex_box.isChecked(),
            whole_word=self.word_box.isChecked(),
            case_sensitive=self.case_box.isChecked(),
        )
        self._target = TARGETS[self.target_combo.currentIndex()][1]
        chapters = None if self.scope_combo.currentText() == SCOPES[1] else [self._chapter]
        try:
            self._changes = self._plan_fn(self._cfg, self._series, rule, self._target, chapters)
        except ValueError as error:
            self._changes = []
            self._show(str(error), error=True)
        else:
            chapters_hit = len({change.chapter for change in self._changes})
            self._show(
                f"{len(self._changes)} line(s) in {chapters_hit} chapter(s) would change"
                if self._changes
                else "no line would change"
            )
        self._fill()
        return len(self._changes)

    def ticked(self) -> list[Change]:
        """The previewed changes whose rows are ticked."""
        return [
            change
            for row, change in enumerate(self._changes)
            if (item := self.table.item(row, 0)) is not None and item.checkState() == Qt.CheckState.Checked
        ]

    def replace_ticked(self) -> int:
        """Record the ticked changes as hand edits; how many lines changed."""
        changes = self.ticked()
        if not changes:
            return 0
        try:
            count = self._apply_fn(self._cfg, self._series, changes, self._target)
        except ValueError as error:
            self._show(str(error), error=True)
            return 0
        self.applied += count
        self._changes = []
        self._fill()
        self._show(f"replaced {count} line(s); Undo takes back one chapter's changes at a time")
        return count

    def _fill(self) -> None:
        """One ticked row per previewed change."""
        self.table.setRowCount(len(self._changes))
        for row, change in enumerate(self._changes):
            tick = QTableWidgetItem()
            tick.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            tick.setCheckState(Qt.CheckState.Checked)
            self.table.setItem(row, 0, tick)
            for column, text in enumerate((change.chapter, change.region_id, change.before, change.after), 1):
                self.table.setItem(row, column, QTableWidgetItem(text))
        self.replace_button.setEnabled(bool(self._changes))

    def _show(self, text: str, *, error: bool = False) -> None:
        """A message next to the buttons (red for a refusal)."""
        self.status_label.setText(text)
        set_role(self.status_label, "error" if error else "")


class ConsistencyDialog(QDialog):
    """A series' consistency report: lines translated in several ways, and missed locked terms."""

    open_line = Signal(str, str)  # (chapter, region id) — a row was double-clicked

    def __init__(self, series: str, report: Consistency, parent: QWidget | None = None) -> None:
        """Show `report` of `series`."""
        super().__init__(parent)
        self.setWindowTitle(f"Consistency — {series}")
        self._places: list[tuple[str, str]] = []  # (chapter, region id) per divergence row
        self._misses = report.misses
        self.divergence_table = _table(DIVERGENCE_COLUMNS, self, stretch=(0, 1, 3))
        self.miss_table = _table(MISS_COLUMNS, self, stretch=(4,))
        rows = [(item, rendering) for item in report.divergences for rendering in item.renderings]
        self.divergence_table.setRowCount(len(rows))
        for row, (item, rendering) in enumerate(rows):
            where = ", ".join(f"{chapter} {region}" for chapter, region in rendering.places)
            cells = (item.source, rendering.english, str(len(rendering.places)), where)
            for column, text in enumerate(cells):
                self.divergence_table.setItem(row, column, QTableWidgetItem(text))
            self._places.append(rendering.places[0])
        self.miss_table.setRowCount(len(report.misses))
        for row, miss in enumerate(report.misses):
            for column, text in enumerate(
                (miss.chapter, miss.region_id, miss.term, miss.target, miss.english)
            ):
                self.miss_table.setItem(row, column, QTableWidgetItem(text))
        self.tabs = QTabWidget(self)
        self.tabs.addTab(self.divergence_table, f"Translated differently ({len(report.divergences)})")
        self.tabs.addTab(self.miss_table, f"Missed locked terms ({len(report.misses)})")
        hint = QLabel("Double-click a row to open that line in the Studio.", self)
        set_role(hint, "muted")
        close = QPushButton("Close", self)
        close.clicked.connect(self.accept)
        bottom = QHBoxLayout()
        bottom.addWidget(hint, 1)
        bottom.addWidget(close)
        root = QVBoxLayout(self)
        root.addWidget(self.tabs, 1)
        root.addLayout(bottom)
        self.resize(900, 480)
        self.divergence_table.cellDoubleClicked.connect(lambda row, _column: self._open(self._places, row))
        self.miss_table.cellDoubleClicked.connect(
            lambda row, _column: self._open([(m.chapter, m.region_id) for m in self._misses], row)
        )

    def _open(self, places: Sequence[tuple[str, str]], row: int) -> None:
        """Ask the Studio to open the line of `row`."""
        if 0 <= row < len(places):
            chapter, region_id = places[row]
            self.open_line.emit(chapter, region_id)

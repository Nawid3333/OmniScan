"""ModelsView: the settings screen's list of every model with hardware fit and action buttons.

Every service call except `rows` at construction/`refresh` runs on a worker (`gui.workers`), so
the GUI thread never blocks on downloads, removals or the Ollama daemon.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from omniscan.gui.services.models import ModelsService
from omniscan.gui.workers import WorkerSignals, run_task
from omniscan.hw.detect import HardwareInfo
from omniscan.models.rows import ModelRow

COLUMNS = ("Model", "Role", "Languages", "Size", "Fit", "Status", "")

# Fit cell backgrounds by level; a clean `ok` row keeps the default background.
FIT_BACKGROUNDS: dict[str, QColor] = {
    "slow": QColor(255, 249, 196),  # light yellow
    "warn": QColor(255, 224, 178),  # light orange
    "incompatible": QColor(255, 205, 210),  # light red
}

Confirm = Callable[[str, str], bool]


def size_text(row: ModelRow) -> str:
    """`"<n> MB"`, `"<x.y> GB"` from 1024 MB up, or `"cloud"` for a cloud model."""
    if row.format == "cloud":
        return "cloud"
    if row.size_mb >= 1024:
        return f"{row.size_mb / 1024:.1f} GB"
    return f"{row.size_mb} MB"


def hardware_header(hw: HardwareInfo) -> str:
    """The one-line machine summary for the header label."""
    if not hw.gpus:
        return f"No GPU found — CPU only — RAM {hw.ram_gb:g} GB — disk {hw.disk_free_gb:g} GB free — torch {hw.torch_build}"
    gpu = hw.gpus[0]
    return (
        f"{gpu.name} · {gpu.vram_gb:g} GB · {gpu.backend} — RAM {hw.ram_gb:g} GB"
        f" — disk {hw.disk_free_gb:g} GB free — torch {hw.torch_build}"
    )


class ModelsView(QWidget):
    """The settings list of every catalog model with fit, status and download/remove buttons."""

    def __init__(
        self,
        service: ModelsService | Any,
        parent: QWidget | None = None,
        *,
        confirm: Confirm | None = None,
    ) -> None:
        """Build the view and load the rows once; `confirm` overrides the question dialog."""
        super().__init__(parent)
        self._service = service
        self._confirm = confirm or self._default_confirm
        self._all_rows: list[ModelRow] = []
        self._view_rows: list[ModelRow] = []
        self._hardware: HardwareInfo | None = None
        self._selected_id: str | None = None
        self._busy = False
        self._task_signals: WorkerSignals | None = None  # keeps the running task's signals alive
        self._buttons: list[QPushButton] = []
        self._required_ids: list[str] = []
        self._failures: list[tuple[str, str]] = []
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------ widgets

    def _build_ui(self) -> None:
        self.hardware_label = QLabel(self)
        self.role_combo = QComboBox(self)
        self.lang_combo = QComboBox(self)
        self.installed_only = QCheckBox("Installed only", self)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search name, id or description")
        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.progress = QProgressBar(self)
        self.progress.setVisible(False)
        self.status_label = QLabel(self)
        self.required_button = QPushButton("Download required models", self)
        self.details = QTextBrowser(self)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Role", self))
        filters.addWidget(self.role_combo)
        filters.addWidget(QLabel("Language", self))
        filters.addWidget(self.lang_combo)
        filters.addWidget(self.installed_only)
        filters.addWidget(self.search, stretch=1)

        bottom = QHBoxLayout()
        bottom.addWidget(self.status_label, stretch=1)
        bottom.addWidget(self.required_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.hardware_label)
        layout.addLayout(filters)
        layout.addWidget(self.table, stretch=1)
        layout.addWidget(self.progress)
        layout.addLayout(bottom)
        layout.addWidget(self.details)

        self.role_combo.currentIndexChanged.connect(lambda _index: self._populate())
        self.lang_combo.currentIndexChanged.connect(lambda _index: self._populate())
        self.installed_only.toggled.connect(lambda _on: self._populate())
        self.search.textChanged.connect(lambda _text: self._populate())
        self.table.currentCellChanged.connect(self._on_current_changed)
        self.required_button.clicked.connect(self._start_required)

    @staticmethod
    def _default_confirm(title: str, text: str) -> bool:
        """The stock question dialog; tests replace this hook with a lambda."""
        return QMessageBox.question(None, title, text) == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------ data

    def refresh(self) -> None:
        """Re-read the rows from the service (also after a download/remove finished)."""
        try:
            rows, hw = self._service.rows()
        except Exception as exc:  # a broken service must surface in the status label, not crash
            self._set_status(f"{type(exc).__name__}: {exc}", error=True)
            return
        self._all_rows = list(rows)
        self._hardware = hw
        self.hardware_label.setText(hardware_header(hw))
        self._rebuild_filters()
        self._populate()

    def _rebuild_filters(self) -> None:
        """Refill role/lang combos from the loaded rows, keeping the chosen value if still there."""
        roles = sorted({row.role for row in self._all_rows if row.role})
        langs = sorted({lang for row in self._all_rows for lang in row.langs})
        self._refill_combo(self.role_combo, "All roles", roles)
        self._refill_combo(self.lang_combo, "All languages", langs)

    def _refill_combo(self, combo: QComboBox, all_text: str, values: list[str]) -> None:
        previous = combo.currentData()
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(all_text)
            for value in values:
                combo.addItem(value, value)
            index = combo.findData(previous)
            combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            combo.blockSignals(False)

    def _filtered_rows(self) -> list[ModelRow]:
        """Client-side filtering of the loaded rows; the service order is kept."""
        role = self.role_combo.currentData()
        lang = self.lang_combo.currentData()
        needle = self.search.text().strip().lower()
        rows = self._all_rows
        if role is not None:
            rows = [row for row in rows if row.role == role]
        if lang is not None:
            rows = [row for row in rows if lang in row.langs]
        if self.installed_only.isChecked():
            rows = [row for row in rows if row.status == "installed"]
        if needle:
            rows = [
                row
                for row in rows
                if needle in row.id.lower() or needle in row.name.lower() or needle in row.description.lower()
            ]
        return rows

    def _populate(self) -> None:
        """Rebuild the table from the filtered rows (catalog order) and restore the selection."""
        self._view_rows = self._filtered_rows()
        self._buttons = []
        self.table.setRowCount(0)
        for index, row in enumerate(self._view_rows):
            self.table.setRowCount(index + 1)
            self._fill_row(index, row)
        self._required_ids = [
            row.id for row in self._all_rows if row.required and row.status in ("missing", "corrupt")
        ]
        self.required_button.setEnabled(bool(self._required_ids) and not self._busy)
        self._restore_selection()
        self._refresh_details()

    def _fill_row(self, index: int, row: ModelRow) -> None:
        model_item = QTableWidgetItem(f"{row.name} (required)" if row.required else row.name)
        model_item.setToolTip(row.id)
        self.table.setItem(index, 0, model_item)
        self.table.setItem(index, 1, QTableWidgetItem(row.role or "-"))
        self.table.setItem(index, 2, QTableWidgetItem(", ".join(row.langs) or "-"))
        self.table.setItem(index, 3, QTableWidgetItem(size_text(row)))
        fit_item = QTableWidgetItem(row.fit_level)
        fit_item.setToolTip("\n".join(row.fit_messages))
        background = FIT_BACKGROUNDS.get(row.fit_level)
        if background is not None:
            fit_item.setBackground(QBrush(background))
        self.table.setItem(index, 4, fit_item)
        status_item = QTableWidgetItem(row.status)
        if row.status == "installed":
            font = QFont(status_item.font())
            font.setBold(True)
            status_item.setFont(font)
        self.table.setItem(index, 5, status_item)
        self._fill_action(index, row)

    def _fill_action(self, index: int, row: ModelRow) -> None:
        """The action cell: a Download/Remove button, nothing for a cloud model."""
        if row.format == "cloud":
            self.table.setItem(index, 6, QTableWidgetItem())
            return
        button = QPushButton("Remove" if row.status == "installed" else "Download", self.table)
        button.clicked.connect(lambda checked=False, i=index: self._on_action(i))
        self.table.setCellWidget(index, 6, button)
        self._buttons.append(button)
        button.setEnabled(not self._busy)

    def _restore_selection(self) -> None:
        """Re-select the previously selected model (first row on the first populate)."""
        if self._selected_id is not None:
            for index, row in enumerate(self._view_rows):
                if row.id == self._selected_id:
                    self.table.selectRow(index)
                    return
        if self._view_rows and self.table.currentRow() < 0:
            self.table.selectRow(0)

    def _on_current_changed(self, row: int, _column: int, _prev_row: int, _prev_column: int) -> None:
        self._selected_id = self._view_rows[row].id if 0 <= row < len(self._view_rows) else None
        self._refresh_details()

    def _refresh_details(self) -> None:
        """Fill the details panel from the current row, plus the last required-download failures."""
        current = self.table.currentRow()
        row = self._view_rows[current] if 0 <= current < len(self._view_rows) else None
        lines: list[str] = [] if row is None else _detail_lines(row)
        if self._failures:
            if lines:
                lines.append("")
            lines.append("Failed downloads:")
            lines.extend(f"{model_id}: {error}" for model_id, error in self._failures)
        self.details.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------ actions

    def _on_action(self, index: int) -> None:
        row = self._view_rows[index]
        if row.status == "installed":
            self._start_remove(row)
        else:
            self._start_download(row)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for button in self._buttons:
            button.setEnabled(enabled)
        self.required_button.setEnabled(enabled and bool(self._required_ids))

    def _set_status(self, text: str, *, error: bool = False) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet("color: red;" if error else "")

    def _start_download(self, row: ModelRow) -> None:
        """Download one model after a confirmation for incompatible ones only."""
        if row.fit_level == "incompatible" and not self._confirm(
            "Download anyway?",
            f"{row.name} is not compatible with this machine:\n" + "\n".join(row.fit_messages),
        ):
            return
        name = row.name
        self._busy = True
        self._set_buttons_enabled(False)
        self._set_status(f"Downloading {name} …")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # busy until the first determinate update arrives
        signals = run_task(lambda progress: self._service.download(row.id, progress))
        self._task_signals = signals  # hold it until the task's terminal signal arrives
        signals.progress.connect(self._on_download_progress)
        signals.finished.connect(lambda source: self._on_download_done(name, source))
        signals.failed.connect(lambda error: self._on_task_failed(name, error))

    def _on_download_progress(self, done: int, total: int | None) -> None:
        if total and total > 0:
            self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _on_download_done(self, name: str, source: Any) -> None:
        self._finish_task()
        self._set_status(f"{name}: installed ({source})")
        self.refresh()

    def _start_remove(self, row: ModelRow) -> None:
        if not self._confirm("Remove model", f"Delete {row.name} from disk?"):
            return
        name = row.name
        self._busy = True
        self._set_buttons_enabled(False)
        self._set_status(f"Removing {name} …")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        signals = run_task(lambda progress: self._service.remove(row.id))
        self._task_signals = signals  # keeps the running task's signals alive
        signals.finished.connect(lambda removed: self._on_remove_done(name, removed))
        signals.failed.connect(lambda error: self._on_task_failed(name, error))

    def _on_remove_done(self, name: str, removed: Any) -> None:
        self._finish_task()
        self._set_status(f"{name}: removed" if removed else f"{name}: nothing to remove")
        self.refresh()

    def _on_task_failed(self, name: str, error: str) -> None:
        self._finish_task()
        self._set_status(f"{name}: {error}", error=True)
        self.refresh()

    def _finish_task(self) -> None:
        self._busy = False
        self._task_signals = None  # terminal signal arrived: the anchor can go
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._set_buttons_enabled(True)

    def _start_required(self) -> None:
        """Download every required model that is missing/corrupt, one worker for all of them."""
        ids = list(self._required_ids)
        current: dict[str, str] = {}  # written on the worker thread, read on the GUI thread

        def task(progress: Callable[[int, int | None], None]) -> list[tuple[str, str | None]]:
            def on_required(model_id: str, done: int, total: int | None) -> None:
                current["id"] = model_id  # set before the progress emit the label reads
                progress(done, total)

            return self._service.download_required(on_required)

        self._busy = True
        self._set_buttons_enabled(False)
        self._failures = []
        self._set_status(f"Downloading {len(ids)} required model(s) …")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        signals = run_task(task)
        self._task_signals = signals  # keeps the running task's signals alive
        signals.progress.connect(lambda done, total: self._on_required_progress(ids, current, done, total))
        signals.finished.connect(self._on_required_done)
        signals.failed.connect(lambda error: self._on_task_failed("required models", error))

    def _on_required_progress(
        self, ids: list[str], current: dict[str, str], done: int, total: int | None
    ) -> None:
        model_id = current.get("id", "")
        position = ids.index(model_id) + 1 if model_id in ids else 0
        if position:
            self._set_status(f"Downloading {model_id} ({position} of {len(ids)}) …")
        else:
            self._set_status(f"Downloading {model_id} …")
        self._on_download_progress(done, total)

    def _on_required_done(self, results: Any) -> None:
        self._finish_task()
        self._failures = [(model_id, error) for model_id, error in results if error is not None]
        ok = len(results) - len(self._failures)
        self._set_status(f"{ok} installed, {len(self._failures)} failed", error=bool(self._failures))
        self.refresh()


def _detail_lines(row: ModelRow) -> list[str]:
    """The details panel's lines for one row (description first, fit reasons, install path)."""
    lines = [row.description]
    if row.license:
        lines.append(f"license: {row.license}")
    if row.family:
        lines.append(f"family: {row.family}")
    if row.size_class:
        lines.append(f"size class: {row.size_class}")
    if row.recommended_for:
        lines.append(f"recommended for: {', '.join(row.recommended_for)}")
    if row.notes:
        lines.append(f"notes: {row.notes}")
    lines.extend(row.fit_messages)
    if row.status == "installed" and row.installed_path:
        lines.append(f"installed at: {row.installed_path}")
    if row.fit_level == "incompatible":
        lines.append("This model cannot run on this machine.")
    return lines

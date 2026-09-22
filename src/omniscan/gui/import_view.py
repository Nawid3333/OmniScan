"""ImportView: pick a folder or .zip/.cbz archive, preview the plan, fix the grouping, commit.

The plan preview is editable before anything is written: pages can move between chapters, pages
can be reordered, chapters renamed, merged or split. Non-JPEG sources are flagged for conversion
(quality 95, on the CPU) before the commit. Every service call runs on a worker (`gui.workers`),
so the GUI thread never blocks on archives or disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QBrush, QColor, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omniscan.gui.services.importer import (
    ImporterService,
    conversion_text,
    merge_chapters,
    move_file,
    rename_chapter,
    set_series,
    split_chapter,
)
from omniscan.gui.workers import WorkerSignals, run_task
from omniscan.importer.plan import ARCHIVE_SUFFIXES, JPEG_SUFFIXES, ImportPlan
from omniscan.legal import NOTICE

CONVERT_BACKGROUND = QBrush(QColor(255, 249, 196))  # light yellow: this page will be re-encoded

Selected = tuple[int, int | None]  # (chapter index, page index or None for the chapter itself)
Kept = tuple[str, Path | None]  # what an edit re-selects: (chapter name, page or None for the chapter)


class ImportView(QWidget):
    """The import page: source picker, editable plan preview, and the commit with progress."""

    def __init__(self, service: ImporterService | Any, parent: QWidget | None = None) -> None:
        """Build the view; `service` is an ImporterService (or a test double with the same methods)."""
        super().__init__(parent)
        self._service = service
        self._plan: ImportPlan | None = None
        self._busy = False
        self._task_signals: WorkerSignals | None = None  # keeps the running task's signals alive
        self._chapter_items: list[QTreeWidgetItem] = []
        self._keep: Kept | None = None  # what the next tree rebuild re-selects
        self._build_ui()
        try:
            self.context_label.setText(self._service.hardware_line())
        except Exception as exc:  # a broken hardware probe must not take the page down
            self.context_label.setText(f"machine: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ widgets

    def _build_ui(self) -> None:
        self.notice_label = QLabel(NOTICE, self)
        self.notice_label.setWordWrap(True)
        self.context_label = QLabel(self)
        self.context_label.setWordWrap(True)

        self.source_edit = QLineEdit(self)
        self.source_edit.setReadOnly(True)
        self.source_edit.setPlaceholderText("Folder or .zip/.cbz — browse or drop it here")
        self.folder_button = QPushButton("Browse folder…", self)
        self.archive_button = QPushButton("Browse archive…", self)
        self.plan_button = QPushButton("Plan import", self)
        self.plan_button.setEnabled(False)
        self.series_edit = QLineEdit(self)
        self.series_edit.setPlaceholderText("Series name")

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source", self))
        source_row.addWidget(self.source_edit, stretch=1)
        source_row.addWidget(self.folder_button)
        source_row.addWidget(self.archive_button)
        source_row.addWidget(self.plan_button)

        self.conversion_label = QLabel(self)
        self.conversion_label.setWordWrap(True)
        self.conversion_label.setVisible(False)
        self.conversions_button = QPushButton("Which files?", self)
        self.conversions_button.setCheckable(True)
        self.conversions_button.setVisible(False)
        conversion_row = QHBoxLayout()
        conversion_row.addWidget(self.conversion_label, stretch=1)
        conversion_row.addWidget(self.conversions_button)

        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(("Chapter / page", "Format"))
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self.target_combo = QComboBox(self)
        self.move_button = QPushButton("Move here", self)
        self.move_button.setToolTip("Move the selected page (or merge the selected chapter) into the target")
        self.up_button = QPushButton("Move up", self)
        self.down_button = QPushButton("Move down", self)
        self.split_button = QPushButton("Split here", self)
        self.split_button.setToolTip("Start a new chapter at the selected page (it and the pages after it)")
        self.chapter_edit = QLineEdit(self)
        self.chapter_edit.setPlaceholderText("Selected chapter's name")
        edit_row = QHBoxLayout()
        edit_row.addWidget(QLabel("Target", self))
        edit_row.addWidget(self.target_combo)
        edit_row.addWidget(self.move_button)
        edit_row.addWidget(self.up_button)
        edit_row.addWidget(self.down_button)
        edit_row.addWidget(self.split_button)
        edit_row.addWidget(self.chapter_edit, stretch=1)

        self.details = QTextBrowser(self)
        self.progress = QProgressBar(self)
        self.progress.setVisible(False)
        self.move_toggle = QCheckBox("Move instead of copy", self)
        self.import_button = QPushButton("Import", self)
        self.import_button.setEnabled(False)
        self.status_label = QLabel(self)

        bottom = QHBoxLayout()
        bottom.addWidget(self.move_toggle)
        bottom.addWidget(self.import_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.notice_label)
        layout.addWidget(self.context_label)
        layout.addLayout(source_row)
        layout.addWidget(self.series_edit)
        layout.addLayout(conversion_row)
        layout.addWidget(self.tree, stretch=1)
        layout.addLayout(edit_row)
        layout.addWidget(self.details)
        layout.addWidget(self.progress)
        layout.addLayout(bottom)
        layout.addWidget(self.status_label)

        self.folder_button.clicked.connect(self._browse_folder)
        self.archive_button.clicked.connect(self._browse_archive)
        self.plan_button.clicked.connect(self._start_plan)
        self.conversions_button.toggled.connect(lambda _on: self._refresh_details())
        self.tree.currentItemChanged.connect(lambda *_: self._sync_selection())
        self.target_combo.currentIndexChanged.connect(lambda _index: self._sync_buttons())
        self.move_button.clicked.connect(self._move_selected)
        self.up_button.clicked.connect(lambda: self._shift_selected(-1))
        self.down_button.clicked.connect(lambda: self._shift_selected(1))
        self.split_button.clicked.connect(self._split_selected)
        self.series_edit.textChanged.connect(self._series_edited)
        self.chapter_edit.textChanged.connect(self._chapter_edited)
        self.import_button.clicked.connect(self._start_import)
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------ source picking

    def set_source(self, source: Path) -> None:
        """Show `source` in the picker and plan it (the same thing browsing or dropping does)."""
        if self._busy:
            return
        self.source_edit.setText(str(source))
        self.plan_button.setEnabled(True)
        self._start_plan()

    def _browse_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose a source folder", self._browse_start())
        if chosen:
            self.set_source(Path(chosen))

    def _browse_archive(self) -> None:
        chosen, _filter = QFileDialog.getOpenFileName(
            self, "Choose a .zip/.cbz archive", self._browse_start(), "Archives (*.zip *.cbz);;All files (*)"
        )
        if chosen:
            self.set_source(Path(chosen))

    def _browse_start(self) -> str:
        """Where the file dialogs open: the last source's parent, else the user's home."""
        text = self.source_edit.text().strip()
        return str(Path(text).parent) if text else str(Path.home())

    def _source_from_urls(self, urls: list[QUrl]) -> Path | None:
        """The first URL that is a local folder or a .zip/.cbz file; None when the drop is ignorable."""
        for url in urls:
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_dir() or (path.is_file() and path.suffix.lower() in ARCHIVE_SUFFIXES):
                return path
        return None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._source_from_urls(event.mimeData().urls()) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        source = self._source_from_urls(event.mimeData().urls())
        if source is not None:
            self.set_source(source)

    # ------------------------------------------------------------------ planning

    def _start_plan(self) -> None:
        if self._busy:
            return
        text = self.source_edit.text().strip()
        if not text:
            self._set_status("Pick a source folder or archive first.", error=True)
            return
        series = self.series_edit.text().strip() or None
        self._drop_plan()
        self._busy = True
        self._sync_buttons()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._set_status(f"Planning {Path(text).name} …")
        signals = run_task(lambda progress: self._service.plan(Path(text), series=series))
        self._task_signals = signals
        signals.finished.connect(self._on_plan_done)
        signals.failed.connect(self._on_plan_failed)

    def _on_plan_done(self, plan: Any) -> None:
        self._finish_task()
        self._show_plan(plan)
        self._set_status(
            f"Planned {plan.series}: {len(plan.items)} chapter(s), "
            f"{sum(len(item.files) for item in plan.items)} page(s)"
        )

    def _on_plan_failed(self, error: str) -> None:
        self._finish_task()
        self._plan = None
        self.tree.clear()
        self._chapter_items = []
        self._target_combo_rebuild()
        self._refresh_preview()
        self.move_toggle.setEnabled(True)
        self._sync_buttons()
        self._set_status(error, error=True)

    def _show_plan(self, plan: ImportPlan) -> None:
        """Replace the working plan and rebuild every view of it."""
        old = self._plan
        self._plan = plan
        if old is not None and old.temp_dir is not None and old.temp_dir is not plan.temp_dir:
            old.cleanup()  # the old archive's extraction is superseded
        series_edit = self.series_edit
        series_edit.blockSignals(True)
        try:
            series_edit.setText(plan.series)
        finally:
            series_edit.blockSignals(False)
        self._rebuild_tree()
        self._target_combo_rebuild()
        self._refresh_preview()
        if plan.archive is not None:  # "move" would only consume throwaway extracted copies
            self.move_toggle.setChecked(False)
        self.move_toggle.setEnabled(plan.archive is None)

    def _drop_plan(self) -> None:
        """Forget the current plan (releasing an archive's temporary extraction)."""
        if self._plan is not None and self._plan.temp_dir is not None:
            self._plan.cleanup()
        self._plan = None

    # ------------------------------------------------------------------ preview

    def _rebuild_tree(self) -> None:
        """Chapters and their files, in plan order; convertible pages marked for conversion."""
        plan = self._plan
        self.tree.clear()
        self._chapter_items = []
        if plan is None:
            return
        for item_index, group in enumerate(plan.items):
            chapter_item = QTreeWidgetItem((group.chapter, f"{len(group.files)} page(s)"))
            chapter_item.setData(0, Qt.ItemDataRole.UserRole, (item_index, None))
            font = chapter_item.font(0)
            font.setBold(True)
            chapter_item.setFont(0, font)
            self.tree.addTopLevelItem(chapter_item)
            self._chapter_items.append(chapter_item)
            for file_index, file in enumerate(group.files):
                converts = file.suffix.lower() not in JPEG_SUFFIXES
                page_item = QTreeWidgetItem(
                    (file.name, "→ JPEG" if converts else file.suffix.lstrip(".").upper())
                )
                page_item.setData(0, Qt.ItemDataRole.UserRole, (item_index, file_index))
                if converts:
                    page_item.setBackground(1, CONVERT_BACKGROUND)
                    page_item.setToolTip(1, "will be converted to JPEG (quality 95) at import")
                chapter_item.addChild(page_item)
            chapter_item.setExpanded(True)
        self._restore_selection()

    def _restore_selection(self) -> None:
        """Re-select what the last edit kept (a page, a chapter, or the first node when nothing is marked)."""
        keep = self._keep
        self._keep = None
        if keep is not None and self._plan is not None:
            chapter_name, file = keep
            for item_index, group in enumerate(self._plan.items):
                if group.chapter != chapter_name:
                    continue
                chapter_item = self._chapter_items[item_index]
                if file is None:
                    self.tree.setCurrentItem(chapter_item)
                    return
                for file_index, candidate in enumerate(group.files):
                    if candidate == file:
                        page_item = chapter_item.child(file_index)
                        if page_item is not None:
                            self.tree.setCurrentItem(page_item)
                        return
        if self._chapter_items:
            first = self._chapter_items[0]
            page_item = first.child(0)
            self.tree.setCurrentItem(page_item if page_item is not None else first)

    def _target_combo_rebuild(self) -> None:
        """Refill the target combo with the chapters' names, keeping the chosen one if still there."""
        previous = self.target_combo.currentText()
        self.target_combo.blockSignals(True)
        try:
            self.target_combo.clear()
            if self._plan is not None:
                for group in self._plan.items:
                    self.target_combo.addItem(group.chapter)
            index = self.target_combo.findText(previous)
            self.target_combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self.target_combo.blockSignals(False)

    def _refresh_preview(self) -> None:
        """The conversion notice and the details pane from the current plan."""
        plan = self._plan
        text = conversion_text(plan) if plan is not None else ""
        self.conversion_label.setText(text)
        self.conversion_label.setVisible(bool(text))
        self.conversions_button.setVisible(bool(text))
        if not text:
            self.conversions_button.setChecked(False)
        self._refresh_details()

    def _refresh_details(self) -> None:
        """Warnings always; the per-file conversion list when "Which files?" is checked."""
        plan = self._plan
        if plan is None:
            self.details.setPlainText("")
            return
        lines = [f"Series: {plan.series}"]
        lines.extend(plan.warnings)
        if self.conversions_button.isChecked():
            if plan.warnings:
                lines.append("")
            lines.append("Will be converted to JPEG:")
            lines.extend(
                f"  {group.chapter}/{file.name} → {file.stem}.jpg"
                for group in plan.items
                for file in group.files
                if file.suffix.lower() not in JPEG_SUFFIXES
            )
        self.details.setPlainText("\n".join(lines))

    def _current(self) -> Selected | None:
        """(chapter index, page index or None) of the selected tree node."""
        item = self.tree.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None

    def _sync_selection(self) -> None:
        """Mirror the selection: the chapter edit shows the selection's chapter, buttons follow."""
        current = self._current()
        chapter_edit = self.chapter_edit
        chapter_edit.blockSignals(True)
        try:
            if current is not None and self._plan is not None:
                chapter_edit.setText(self._plan.items[current[0]].chapter)
            else:
                chapter_edit.clear()
        finally:
            chapter_edit.blockSignals(False)
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        current = self._current()
        busy = self._busy
        has_items = self._plan is not None and bool(self._plan.items)
        page_selected = current is not None and current[1] is not None
        self.import_button.setEnabled(has_items and not busy)
        self.plan_button.setEnabled(not busy and self.source_edit.text().strip() != "")
        self.folder_button.setEnabled(not busy)
        self.archive_button.setEnabled(not busy)
        self.target_combo.setEnabled(current is not None and not busy)
        self.move_button.setEnabled(current is not None and not busy)
        self.up_button.setEnabled(page_selected and not busy)
        self.down_button.setEnabled(page_selected and not busy)
        self.split_button.setEnabled(page_selected and not busy)
        self.chapter_edit.setEnabled(current is not None and not busy)

    # ------------------------------------------------------------------ editing

    def _series_edited(self, text: str) -> None:
        if self._plan is not None:
            self._plan = set_series(self._plan, text.strip())

    def _chapter_edited(self, text: str) -> None:
        current = self._current()
        name = text.strip()
        if self._plan is None or current is None or not name:
            return
        self._plan = rename_chapter(self._plan, current[0], name)
        self._chapter_items[current[0]].setText(0, name)

    def _move_selected(self) -> None:
        current = self._current()
        if self._plan is None or current is None:
            return
        item_index, file_index = current
        target_name = self.target_combo.currentText()
        target = next((i for i, group in enumerate(self._plan.items) if group.chapter == target_name), None)
        if target is None:
            return
        if file_index is None:  # a chapter is selected: this move is a merge
            if target == item_index:
                self._set_status("Pick a different chapter to merge into.", error=True)
                return
            self._plan = merge_chapters(self._plan, item_index, target)
            keep = (target_name, None)  # re-select the merged-into chapter
        else:
            if target == item_index:
                self._set_status(
                    f"That page is already in {self._plan.items[item_index].chapter}.", error=True
                )
                return
            file = self._plan.items[item_index].files[file_index]
            self._plan = move_file(self._plan, item_index, file_index, target)
            keep = (target_name, file)
        self._after_edit(keep)

    def _shift_selected(self, delta: int) -> None:
        current = self._current()
        if self._plan is None or current is None:
            return
        item_index, file_index = current
        if file_index is None:
            return
        new_index = file_index + delta
        files = self._plan.items[item_index].files
        if not 0 <= new_index < len(files):
            return  # already at that edge
        file = files[file_index]
        self._plan = move_file(self._plan, item_index, file_index, item_index, new_index)
        self._after_edit((self._plan.items[item_index].chapter, file))

    def _split_selected(self) -> None:
        current = self._current()
        if self._plan is None or current is None:
            return
        item_index, file_index = current
        if file_index is None:
            return
        file = self._plan.items[item_index].files[file_index]
        try:
            self._plan = split_chapter(self._plan, item_index, file_index)
        except ValueError as exc:
            self._set_status(str(exc), error=True)
            return
        self._after_edit((self._plan.items[item_index + 1].chapter, file))  # the split's first page

    def _after_edit(self, keep: tuple[str, Path | None]) -> None:
        """Rebuild from the edited plan, re-selecting `keep` (chapter, page-or-None) and refilling the target."""
        self._keep = keep
        self._rebuild_tree()
        self._target_combo_rebuild()
        self._sync_buttons()

    # ------------------------------------------------------------------ import

    def _start_import(self) -> None:
        plan = self._plan
        if plan is None or self._busy:
            return
        names = [group.chapter for group in plan.items]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            self._set_status(
                f"Two chapters would write to the same folder: {', '.join(duplicates)} — rename them first.",
                error=True,
            )
            return
        move = self.move_toggle.isChecked()
        self._busy = True
        self._sync_buttons()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._set_status(f"Importing {plan.series} …")
        signals = run_task(lambda progress: self._service.execute(plan, move=move, on_progress=progress))
        self._task_signals = signals
        signals.progress.connect(self._on_import_progress)
        signals.finished.connect(self._on_import_done)
        signals.failed.connect(self._on_import_failed)

    def _on_import_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.progress.setRange(0, total)
        self.progress.setValue(done)
        self._set_status(f"Importing … {done}/{total}")

    def _on_import_done(self, result: Any) -> None:
        self._finish_task()
        self._set_status(
            f"Imported: {len(result.chapters_written)} chapter(s), {result.files_copied} copied, "
            f"{result.files_converted} converted, {result.files_skipped_duplicate} duplicate(s) skipped"
        )

    def _on_import_failed(self, error: str) -> None:
        self._finish_task()
        self._set_status(error, error=True)

    def _finish_task(self) -> None:
        self._busy = False
        self._task_signals = None
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._sync_buttons()

    def _set_status(self, text: str, *, error: bool = False) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet("color: red;" if error else "")

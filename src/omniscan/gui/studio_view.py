"""StudioView: the Translator Studio page — one chapter's boxes, source text, English and lettering by hand.

Left: the raw strip with every text region outlined; the selected box can be dragged, resized by its handles
or nudged with the arrow keys, and `Draw box` (or Shift + drag) draws a region the detector missed. Middle
(with `Preview` on): the page as the release will look, rendered on the spot with the saved edits, scrolling
with the raw strip. Right: one row per region (page, kind, speaker, source, English, status, issues), editable
in place; several rows can be selected at once and the actions apply to all of them (remove, mark checked,
revert English, kind, lettering styles, translate). `Find missed text` runs the detector again on the page in
view and offers what no region covers, to add as hand-drawn regions. `Find & replace…` and `Consistency…`
(gui/series_dialogs.py) work across the chapter or the whole series; a line in the report opens here.
`Output cuts` shows where the exported images split and edits that on the strip (each change its own undo step).

Everything goes through `omniscan.edits.session.StudioSession`: changes stay in memory until Save, which
records them in edits.json (one undo step; Undo/Redo walk the chapter's shared history) and applies them to
ocr.json / final.json at once. Translate and Read again ask the models on a worker thread and put their
suggestions into the session, so nothing is written before Save. Re-letter runs typeset + export.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PySide6.QtCore import QItemSelectionModel, QModelIndex, QPersistentModelIndex, Qt, Signal
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QTableWidgetSelectionRange,
    QVBoxLayout,
    QWidget,
)

from omniscan.core.config import Config, series_config
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import BBox, CleanupPatch, QaIssue, RegionKind
from omniscan.edits import store
from omniscan.edits.session import StudioRow, StudioSession
from omniscan.gui.found_dialog import FoundDialog
from omniscan.gui.lettering_dialog import LetteringDialog
from omniscan.gui.run_worker import RunWorker
from omniscan.gui.series_dialogs import ConsistencyDialog, ReplaceDialog
from omniscan.gui.services import library, series_check
from omniscan.gui.services import studio as services
from omniscan.gui.services.library import Tile
from omniscan.gui.services.runs import RunController, RunOutcome, RunSpec
from omniscan.gui.services.series_check import Consistency
from omniscan.gui.services.studio import CleanMethod, CutsState, PagePreview, PageStroke, Suggested
from omniscan.gui.strip_view import StripView
from omniscan.gui.theme import set_role
from omniscan.gui.workers import WorkerSignals, run_task

if TYPE_CHECKING:
    from omniscan.detect.on_demand import Found

COLUMNS = ("Page", "Kind", "Speaker", "Source", "English", "Status", "Issues")
_PAGE_COL, _KIND_COL, _SPEAKER_COL, _SOURCE_COL, _ENGLISH_COL, _STATUS_COL, _ISSUES_COL = range(7)
_EDITABLE = (_SPEAKER_COL, _SOURCE_COL, _ENGLISH_COL)
RELETTER_STAGES = ("typeset", "export")
KINDS: tuple[RegionKind, ...] = ("bubble_text", "free_text", "sfx", "watermark")
ENABLED_PROFILES = "(enabled profiles)"
_KIND_PLACEHOLDER = "Kind…"
SERIES_THRESHOLD = "series setting"  # the find threshold spinner at its minimum: the detector's own threshold

ControllerFactory = Callable[[Config, RunSpec], Any]  # a RunController (tests pass a fake)
TranslateFn = Callable[[Config, ChapterPaths, list[str], str | None], list[Suggested]]
ReadFn = Callable[[Config, ChapterPaths, str], str]
PreviewFn = Callable[[Config, ChapterPaths, int], PagePreview]
FindFn = Callable[[Config, ChapterPaths, int, float | None], list["Found"]]
PickFn = Callable[
    [list["Found"], int, QWidget], list["Found"]
]  # which found areas to add (a dialog; tests fake it)
ReplaceFn = Callable[[Config, str, str, QWidget], int]  # find and replace in (series, chapter): lines changed
ConsistencyFn = Callable[[Config, str], Consistency]
QaFn = Callable[[Config, ChapterPaths], list[QaIssue]]  # re-read the finished pages: what still shows
CleanFn = Callable[[Config, ChapterPaths, PageStroke, CleanMethod], CleanupPatch]  # clean one brush stroke
CLEAN_METHODS: tuple[tuple[str, CleanMethod], ...] = (
    ("Fill with the colour around", "fill"),
    ("Inpaint", "inpaint"),
    ("LaMa", "lama"),
    ("Restore the raw page", "restore"),
)
BRUSH_RADIUS = 12  # strip px: the brush's starting radius


def _find_replace(cfg: Config, series: str, chapter: str, parent: QWidget) -> int:
    """Run the find-and-replace dialog over `chapter` (or the whole series); how many lines it changed."""
    dialog = ReplaceDialog(cfg, series, chapter, parent=parent)
    dialog.exec()
    return dialog.applied


def _pick_found(found: list[Found], page: int, parent: QWidget) -> list[Found]:
    """Ask which found areas to add (FoundDialog); none when it is cancelled."""
    dialog = FoundDialog(found, page=page, parent=parent)
    return dialog.picked() if dialog.exec() else []


class _SpeakerDelegate(QStyledItemDelegate):
    """The Speaker cell's editor: free text, completing the series' character names (voices.toml)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Start with no names."""
        super().__init__(parent)
        self.names: list[str] = []

    def createEditor(
        self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex
    ) -> QWidget:
        """A line edit that completes a character's name (case-insensitive, anywhere in the name)."""
        editor = QLineEdit(parent)
        completer = QCompleter(self.names, editor)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        editor.setCompleter(completer)
        return editor


def _qimage(pixels: np.ndarray) -> QImage:
    """A QImage of uint8 [h, w, 3] pixels (its own copy)."""
    rgb = np.ascontiguousarray(pixels[..., :3], dtype=np.uint8)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class StudioView(QWidget):
    """Edit one chapter's regions, lines and lettering; QA issues shown per row."""

    busy_changed = Signal(bool)  # a re-letter run started / ended

    def __init__(
        self,
        cfg: Config,
        *,
        controller_factory: ControllerFactory | None = None,
        translate_fn: TranslateFn | None = None,
        read_fn: ReadFn | None = None,
        preview_fn: PreviewFn | None = None,
        find_fn: FindFn | None = None,
        pick_fn: PickFn | None = None,
        replace_fn: ReplaceFn | None = None,
        consistency_fn: ConsistencyFn | None = None,
        qa_fn: QaFn | None = None,
        clean_fn: CleanFn | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build the page; the factories replace the re-letter run, the models, the renderer, the dialog that
        picks found text, the find-and-replace dialog and the consistency report (tests)."""
        super().__init__(parent)
        self._cfg = cfg
        self._controller_factory = controller_factory or (lambda cfg, spec: RunController(cfg, spec))
        self._translate_fn: TranslateFn = translate_fn or services.translate_regions
        self._read_fn: ReadFn = read_fn or services.read_region
        self._preview_fn: PreviewFn = preview_fn or services.render_preview
        self._find_fn: FindFn = find_fn or services.find_missed
        self._pick_fn: PickFn = pick_fn or _pick_found
        self._replace_fn: ReplaceFn = replace_fn or _find_replace
        self._consistency_fn: ConsistencyFn = consistency_fn or series_check.consistency
        self._qa_fn: QaFn = qa_fn or services.read_finished_pages
        self._clean_fn: CleanFn = clean_fn or services.clean_stroke
        self._strokes: list[tuple[str, int]] = []  # (patch id, page) of this chapter's strokes, newest last
        self.consistency_dialog: ConsistencyDialog | None = None  # the open report, if any
        self._session: StudioSession | None = None
        self._rows: list[StudioRow] = []
        self._issues: dict[str, list[str]] = {}
        self._typos: dict[str, list[str]] = {}  # region id -> the unknown words the check found in its line
        self._filling = False
        self._worker: RunWorker | None = None
        self._tasks: list[WorkerSignals] = []  # model calls and renders in flight
        self._page_rows: list[tuple[int, int, int]] = []  # (page, y0, y1) of the raw pages (ingest.json)
        self._rendered: set[int] = set()  # pages the preview holds (or is rendering)
        self._syncing = False  # True while this view itself aligns the two strips
        self._cuts: CutsState | None = None  # the chapter's output cuts while Output cuts is on

        # ---- navigation
        self.series_combo = QComboBox(self)
        self.chapter_combo = QComboBox(self)
        self.chapter_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.prev_chapter_button = QPushButton("◀", self)
        self.prev_chapter_button.setToolTip("Previous chapter (Ctrl+Shift+Left)")
        self.next_chapter_button = QPushButton("▶", self)
        self.next_chapter_button.setToolTip("Next chapter (Ctrl+Shift+Right)")
        self.page_spin = QSpinBox(self)
        self.page_spin.setToolTip("The page (output slice) shown; Page Up / Page Down step through them")
        self.prev_page_button = QPushButton("◀", self)
        self.next_page_button = QPushButton("▶", self)
        self.page_only = QCheckBox("Only this page", self)
        self.issues_only = QCheckBox("Only lines with issues", self)
        self.progress_label = QLabel(self)
        set_role(self.progress_label, "muted")
        self.replace_button = QPushButton("Find && replace…", self)
        self.replace_button.setToolTip(
            "Find and replace in this chapter's or the whole series' English lines or source texts (Ctrl+H)"
        )
        self.qa_button = QPushButton("Read finished pages", self)
        self.qa_button.setToolTip(
            "Re-read the exported pages with the OCR and flag lines whose source text or watermark still shows "
            "(omniscan qa; the models load for it)"
        )
        self.consistency_button = QPushButton("Consistency…", self)
        self.consistency_button.setToolTip(
            "The series' source lines translated in more than one way, and lines that miss a locked term"
        )

        # ---- actions
        self.undo_button = QPushButton("Undo", self)
        self.undo_button.setToolTip("Take back the last saved change (Ctrl+Z)")
        self.redo_button = QPushButton("Redo", self)
        self.redo_button.setToolTip("Make the last undone change again (Ctrl+Y)")
        self.check_button = QPushButton("Check", self)
        self.check_button.setToolTip("Run the automatic quality check")
        self.not_typo_button = QPushButton("Not a typo", self)
        self.not_typo_button.setToolTip("Accept the selected line's unknown words for the whole series")
        self.profile_combo = QComboBox(self)
        self.profile_combo.addItem(ENABLED_PROFILES)
        self.profile_combo.setToolTip("Which translation profile Translate asks")
        self.translate_button = QPushButton("Translate", self)
        self.translate_button.setToolTip("Ask the translation model for the selected lines (kept on Save)")
        self.read_button = QPushButton("Read again", self)
        self.read_button.setToolTip("Read the selected box again with the OCR engine (kept on Save)")
        self.revert_button = QPushButton("Revert English", self)
        self.revert_button.setToolTip("Back to the machine's line for the selected rows")
        self.checked_button = QPushButton("Mark checked", self)
        self.checked_button.setToolTip("Approve the selected lines as they are (proofreading)")
        self.unchecked_button = QPushButton("Unmark", self)
        self.unchecked_button.setToolTip("Take back the selected lines' approval")
        self.kind_combo = QComboBox(self)
        self.kind_combo.addItem(_KIND_PLACEHOLDER)
        self.kind_combo.addItems(KINDS)
        self.kind_combo.setToolTip(
            "Make the selected regions bubble text, free text, a sound effect or a watermark"
        )
        self.lettering_button = QPushButton("Lettering…", self)
        self.lettering_button.setToolTip(
            "Font, size, colours, outline, alignment and angle of the selected lines"
        )
        self.draw_button = QPushButton("Draw box", self)
        self.draw_button.setCheckable(True)
        self.draw_button.setToolTip(
            "Drag on the strip to draw a region the detector missed (Ctrl+B; or hold Shift)"
        )
        self.remove_button = QPushButton("Remove box", self)
        self.remove_button.setToolTip("Delete the selected regions (false detections; Delete)")
        self.find_button = QPushButton("Find missed text", self)
        self.find_button.setToolTip(
            "Run the detector again on the page in view and offer the text no box covers (the models load for "
            "the search)"
        )
        self.find_threshold = QDoubleSpinBox(self)
        self.find_threshold.setRange(0.0, 0.95)
        self.find_threshold.setSingleStep(0.05)
        self.find_threshold.setDecimals(2)
        self.find_threshold.setSpecialValueText(SERIES_THRESHOLD)
        self.find_threshold.setToolTip(
            "The detector's score threshold for Find missed text: lower finds fainter text (and more noise)"
        )

        # ---- hand cleanup
        self.brush_button = QPushButton("Brush", self)
        self.brush_button.setCheckable(True)
        self.brush_button.setToolTip(
            "Paint over leftover text or marks on the strip to clean them by hand; the Preview shows the result "
            "(Ctrl+E)"
        )
        self.clean_combo = QComboBox(self)
        for label, method in CLEAN_METHODS:
            self.clean_combo.addItem(label, method)
        self.clean_combo.setToolTip("How a stroke is cleaned (LaMa loads its model for each stroke)")
        self.brush_size = QSpinBox(self)
        self.brush_size.setRange(1, 200)
        self.brush_size.setValue(BRUSH_RADIUS)
        self.brush_size.setSuffix(" px")
        self.brush_size.setToolTip("The brush radius, in strip pixels")
        self.take_back_button = QPushButton("Take back stroke", self)
        self.take_back_button.setToolTip("Remove the last stroke cleaned here")

        # ---- output cuts
        self.cuts_button = QPushButton("Output cuts", self)
        self.cuts_button.setCheckable(True)
        self.cuts_button.setToolTip(
            "Show where the exported images split and change it: click the strip to add a cut, drag one to move "
            "it, right-click one to remove it"
        )
        self.snap_check = QCheckBox("Snap to calm rows", self)
        self.snap_check.setChecked(True)
        self.snap_check.setToolTip(
            f"A new or moved cut jumps into a uniform band within {services.SNAP_ROWS} rows (a clean place between "
            "panels)"
        )
        self.reset_cuts_button = QPushButton("One image per slice", self)
        self.reset_cuts_button.setToolTip("Drop the cuts set by hand: export one image per slice again")
        self.cuts_label = QLabel(self)
        self.preview_button = QPushButton("Preview", self)
        self.preview_button.setCheckable(True)
        self.preview_button.setToolTip("Show the page as the release will look, with the saved edits")
        self.save_button = QPushButton("Save", self)
        self.save_button.setToolTip("Record the changes (Ctrl+S); one Save is one undo step")
        self.reletter_button = QPushButton("Re-letter", self)
        set_role(self.reletter_button, "primary")
        self.reletter_button.setToolTip("Save, then redo lettering and export for this chapter")
        self.status_label = QLabel(self)

        # ---- strip, preview, table
        self.strip = StripView(self)
        self.strip.set_editable(True)
        self.preview = StripView(self)
        self.preview.setVisible(False)
        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for column in (_SOURCE_COL, _ENGLISH_COL, _ISSUES_COL):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self._speaker_delegate = _SpeakerDelegate(self.table)
        self.table.setItemDelegateForColumn(_SPEAKER_COL, self._speaker_delegate)

        nav = QHBoxLayout()
        for widget in (
            QLabel("Series", self),
            self.series_combo,
            QLabel("Chapter", self),
            self.prev_chapter_button,
            self.chapter_combo,
            self.next_chapter_button,
            QLabel("Page", self),
            self.prev_page_button,
            self.page_spin,
            self.next_page_button,
            self.page_only,
            self.issues_only,
        ):
            nav.addWidget(widget)
        nav.addStretch(1)
        nav.addWidget(self.progress_label)
        nav.addWidget(self.qa_button)
        nav.addWidget(self.replace_button)
        nav.addWidget(self.consistency_button)

        actions = QHBoxLayout()
        for widget in (
            self.undo_button,
            self.redo_button,
            self.check_button,
            self.not_typo_button,
            self.profile_combo,
            self.translate_button,
            self.read_button,
            self.revert_button,
            self.checked_button,
            self.unchecked_button,
            self.kind_combo,
            self.lettering_button,
            self.draw_button,
            self.remove_button,
            self.find_button,
            self.find_threshold,
        ):
            actions.addWidget(widget)
        actions.addStretch(1)
        for widget in (self.preview_button, self.save_button, self.reletter_button):
            actions.addWidget(widget)

        self.splitter = QSplitter(self)
        self.splitter.addWidget(self.strip)
        self.splitter.addWidget(self.preview)
        self.splitter.addWidget(self.table)
        self.splitter.setSizes([2, 2, 3])

        clean = QHBoxLayout()
        clean.addWidget(QLabel("Clean", self))
        for widget in (self.brush_button, self.clean_combo, self.brush_size, self.take_back_button):
            clean.addWidget(widget)
        clean.addSpacing(24)
        clean.addWidget(QLabel("Export", self))
        for widget in (self.cuts_button, self.snap_check, self.reset_cuts_button, self.cuts_label):
            clean.addWidget(widget)
        clean.addStretch(1)

        root = QVBoxLayout(self)
        root.addLayout(nav)
        root.addLayout(actions)
        root.addLayout(clean)
        root.addWidget(self.splitter, 1)
        root.addWidget(self.status_label)

        self.series_combo.currentTextChanged.connect(self._on_series)
        self.chapter_combo.currentTextChanged.connect(self._on_chapter)
        self.prev_chapter_button.clicked.connect(lambda: self.step_chapter(-1))
        self.next_chapter_button.clicked.connect(lambda: self.step_chapter(1))
        self.page_spin.valueChanged.connect(self._on_page)
        self.prev_page_button.clicked.connect(lambda: self.step_page(-1))
        self.next_page_button.clicked.connect(lambda: self.step_page(1))
        self.page_only.toggled.connect(lambda _checked: self._apply_filter())
        self.issues_only.toggled.connect(lambda _checked: self._apply_filter())
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.check_button.clicked.connect(self.run_check)
        self.not_typo_button.clicked.connect(self.allow_selected_words)
        self.translate_button.clicked.connect(self.translate_selected)
        self.read_button.clicked.connect(self.read_selected)
        self.revert_button.clicked.connect(self.revert_selected)
        self.checked_button.clicked.connect(lambda: self.mark_selected(True))
        self.unchecked_button.clicked.connect(lambda: self.mark_selected(False))
        self.kind_combo.activated.connect(self._on_kind_picked)
        self.lettering_button.clicked.connect(self.edit_lettering)
        self.draw_button.toggled.connect(self._on_draw_toggled)
        self.brush_button.toggled.connect(self.set_brush)
        self.brush_size.valueChanged.connect(
            lambda radius: self.strip.set_brush(radius) if self.brush_button.isChecked() else None
        )
        self.strip.stroke_painted.connect(self.clean_stroke)
        self.take_back_button.clicked.connect(self.take_back_stroke)
        self.cuts_button.toggled.connect(self.set_cut_mode)
        self.reset_cuts_button.clicked.connect(lambda: self.change_cuts(None))
        self.strip.cut_added.connect(self._on_cut_added)
        self.strip.cut_moved.connect(self._on_cut_moved)
        self.strip.cut_removed.connect(self._on_cut_removed)
        self.remove_button.clicked.connect(self.remove_selected)
        self.find_button.clicked.connect(self.find_missed)
        self.replace_button.clicked.connect(self.find_replace)
        self.consistency_button.clicked.connect(self.check_consistency)
        self.qa_button.clicked.connect(self.read_finished)
        self.preview_button.toggled.connect(self.set_preview_visible)
        self.save_button.clicked.connect(self.save)
        self.reletter_button.clicked.connect(self.reletter)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.strip.overlay_clicked.connect(self.select_region)
        self.strip.overlay_changed.connect(self._on_box_changed)
        self.strip.box_drawn.connect(self.add_box)
        self.strip.strip_y_changed.connect(lambda y: self._follow(self.strip, y=y))
        self.strip.zoom_changed.connect(lambda z: self._follow(self.strip, zoom=z))
        self.preview.strip_y_changed.connect(lambda y: self._follow(self.preview, y=y))
        self.preview.zoom_changed.connect(lambda z: self._follow(self.preview, zoom=z))
        self._shortcuts()
        self._load_profiles()
        self._reload_series()

    def _shortcuts(self) -> None:
        """The keyboard: undo/redo, save, pages, chapters, draw and remove."""
        context = Qt.ShortcutContext.WidgetWithChildrenShortcut
        for keys, slot in (
            (QKeySequence.StandardKey.Undo, self.undo),
            (QKeySequence.StandardKey.Redo, self.redo),
            (QKeySequence("Ctrl+Y"), self.redo),
            (QKeySequence.StandardKey.Save, self.save),
            (QKeySequence("PgDown"), lambda: self.step_page(1)),
            (QKeySequence("PgUp"), lambda: self.step_page(-1)),
            (QKeySequence("Ctrl+Shift+Right"), lambda: self.step_chapter(1)),
            (QKeySequence("Ctrl+Shift+Left"), lambda: self.step_chapter(-1)),
            (QKeySequence("Ctrl+B"), self.draw_button.toggle),
            (QKeySequence("Delete"), self.remove_selected),
            (QKeySequence("Ctrl+H"), self.find_replace),
            (QKeySequence("Ctrl+E"), self.brush_button.toggle),
        ):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(context)
            shortcut.activated.connect(slot)

    def _load_profiles(self) -> None:
        """Offer every known translation profile next to the enabled ones."""
        try:
            names = services.profile_names()
        except ValueError:
            names = []
        self.profile_combo.addItems(names)

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

    def step_chapter(self, step: int) -> bool:
        """Open the previous (-1) or next (+1) chapter of the series; False at either end."""
        index = self.chapter_combo.currentIndex() + step
        if not 0 <= index < self.chapter_combo.count():
            return False
        self.chapter_combo.setCurrentIndex(index)
        return True

    def step_page(self, step: int) -> bool:
        """Show the previous (-1) or next (+1) page that has regions; False at either end."""
        pages = self._pages()
        if not pages:
            return False
        current = self.page_spin.value()
        later = [page for page in pages if (page > current if step > 0 else page < current)]
        if not later:
            return False
        self.page_spin.setValue(later[0] if step > 0 else later[-1])
        return True

    def session(self) -> StudioSession | None:
        """The open chapter's edit session."""
        return self._session

    def selected_ids(self) -> list[str]:
        """The regions of the selected rows, in table order."""
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        return [self._rows[row].region_id for row in rows if 0 <= row < len(self._rows)]

    def select_region(self, region_id: str) -> None:
        """Select a region's row alone and show it in the strip."""
        for row, item in enumerate(self._rows):
            if item.region_id == region_id:
                self._select_rows([row])
                return

    def _select_rows(self, rows: Sequence[int]) -> None:
        """Select exactly `rows` (an explicit selection: unlike `selectRow`, it does not depend on which
        modifier keys the user holds); the first becomes the current row."""
        self._filling = True
        try:
            self.table.clearSelection()
            last = len(COLUMNS) - 1
            for row in rows:
                self.table.setRangeSelected(QTableWidgetSelectionRange(row, 0, row, last), True)
            if rows:
                self.table.setCurrentCell(rows[0], 0, QItemSelectionModel.SelectionFlag.NoUpdate)
        finally:
            self._filling = False
        self._on_selection()

    def run_check(self) -> int:
        """Run the QA pass, show each row's issues (with what the last re-read of the finished pages found);
        returns how many issues were found."""
        if self._session is None:
            return 0
        issues = self._session.issues()
        finished = services.finished_page_issues(self._session.paths)
        self._issues, self._typos = {}, {}
        for issue in issues:
            self._issues.setdefault(issue.region_id, []).append(issue.message)
            if issue.kind == "typo":
                self._typos.setdefault(issue.region_id, []).append(issue.word)
        for leftover in finished:
            self._issues.setdefault(leftover.region_id, []).append(f"finished page: {leftover.message}")
        count = len(issues) + len(finished)
        self._fill_table()
        self.status_label.setText(f"{count} issue(s) in {len(self._issues)} line(s)")
        return count

    def allow_selected_words(self) -> list[str]:
        """Mark the selected lines' unknown words "not a typo" for the series, then check again; returns them."""
        if self._session is None:
            return []
        words = [word for region_id in self.selected_ids() for word in self._typos.get(region_id, [])]
        for word in words:
            self._session.allow_word(word)
        self.run_check()
        if words:
            self.status_label.setText(f"{', '.join(words)}: not a typo in this series")
        return words

    def remove_selected(self) -> int:
        """Remove the selected regions from the chapter (applied on Save); how many."""
        ids = self.selected_ids()
        if self._session is None or not ids:
            return 0
        for region_id in ids:
            self._session.remove_region(region_id)
        self._refresh()
        return len(ids)

    def revert_selected(self) -> int:
        """Set the selected rows' English back to the machine's line (applied on Save); how many."""
        ids = self.selected_ids()
        if self._session is None or not ids:
            return 0
        for region_id in ids:
            self._session.set_translation(region_id, self._session.machine_line(region_id))
        self._refresh()
        return len(ids)

    def mark_selected(self, checked: bool) -> int:
        """Approve (or unapprove) the selected lines for proofreading (applied on Save); how many."""
        ids = self.selected_ids()
        if self._session is None or not ids:
            return 0
        for region_id in ids:
            self._session.set_checked(region_id, checked)
        self._refresh()
        return len(ids)

    def set_kind(self, kind: RegionKind) -> int:
        """Make the selected regions `kind` (applied on Save); how many."""
        ids = self.selected_ids()
        if self._session is None or not ids:
            return 0
        for region_id in ids:
            self._session.set_kind(region_id, kind)
        self._refresh()
        return len(ids)

    def apply_lettering(self, fields: dict[str, object] | None) -> int:
        """Set the given lettering styles on the selected regions (None: back to the typesetter); how many."""
        ids = self.selected_ids()
        if self._session is None or not ids:
            return 0
        for region_id in ids:
            if fields is None:
                self._session.revert_layout(region_id)
            else:
                self._session.set_layout(region_id, dict(fields))
        self._refresh()
        return len(ids)

    def edit_lettering(self) -> None:
        """Open the lettering dialog for the selected regions and apply what it says."""
        ids = self.selected_ids()
        if self._session is None or not ids:
            return
        dialog = LetteringDialog(
            services.font_names(), self._session.layout_of(ids[0]), count=len(ids), parent=self
        )
        if dialog.exec():
            self.apply_lettering(None if dialog.reverts() else dialog.fields() or None)

    def add_box(self, x0: int, y0: int, x1: int, y1: int) -> str | None:
        """Add a hand-drawn region (saved at once, its own undo step) and select it; its id."""
        if self._session is None:
            return None
        picked = self.kind_combo.currentText()
        kind: RegionKind = cast(RegionKind, picked) if picked in KINDS else "bubble_text"
        try:
            region_id = self._session.add_region(BBox(x0=x0, y0=y0, x1=x1, y1=y1), kind=kind)
        except (LookupError, OSError, ValueError) as error:
            self.status_label.setText(f"cannot add a box: {error}")
            return None
        self.draw_button.setChecked(False)
        self._refresh()
        self.select_region(region_id)
        self.status_label.setText(f"{region_id}: drawn; type its text, then Translate or Save")
        self._invalidate_preview()
        return region_id

    def save(self) -> int:
        """Save the edits (one undo step); returns how many changes were recorded."""
        if self._session is None:
            return 0
        count = self._session.save()
        self.status_label.setText(f"saved {count} change(s)" if count else "nothing to save")
        self._refresh()
        if count:
            self._invalidate_preview()
        return count

    def undo(self) -> bool:
        """Take back the last saved change of the chapter."""
        return self._history("undo")

    def redo(self) -> bool:
        """Make the last undone change again."""
        return self._history("redo")

    def translate_selected(self) -> bool:
        """Ask the translation model for the selected rows on a worker thread; the suggestions land in the
        English column, to keep with Save. False when nothing is selected or a model call is running."""
        ids = self.selected_ids()
        session = self._session
        if session is None or not ids or self._tasks:
            return False
        picked = self.profile_combo.currentText()
        profile = None if picked == ENABLED_PROFILES else picked
        cfg, paths = self._cfg, session.paths
        self._start_task(
            lambda _progress: self._translate_fn(cfg, paths, ids, profile),
            self._on_translated,
            f"translating {len(ids)} line(s)...",
        )
        return True

    def read_selected(self) -> bool:
        """Read the selected box again with the OCR engine on a worker thread; the reading becomes the source
        text, to keep with Save."""
        ids = self.selected_ids()
        session = self._session
        if session is None or not ids or self._tasks:
            return False
        cfg, paths, region_id = self._cfg, session.paths, ids[0]
        self._start_task(
            lambda _progress: (region_id, self._read_fn(cfg, paths, region_id)),
            self._on_read,
            f"reading {region_id} again...",
        )
        return True

    def find_missed(self) -> bool:
        """Run the detector again on the raw page in view (on a worker thread) and offer the text no region
        covers; the ticked areas become regions. False when no chapter is open or a model call is running."""
        session = self._session
        page = self._page_in_view()
        if session is None or page is None or self._tasks:
            return False
        value = self.find_threshold.value()
        threshold = None if value <= self.find_threshold.minimum() else value
        cfg, paths = self._cfg, session.paths
        self._start_task(
            lambda _progress: (page, self._find_fn(cfg, paths, page, threshold)),
            self._on_found,
            f"looking for missed text on page {page + 1}...",
        )
        return True

    def find_replace(self) -> int:
        """Save the pending edits, run the find-and-replace dialog and show the chapter again when it changed
        lines; how many lines it changed."""
        session = self._session
        if session is None or self._worker is not None:
            return 0
        if session.dirty:  # the replace writes edits.json itself: unsaved edits go first, as their own step
            self.save()
        series, chapter = session.paths.series, session.paths.chapter
        changed = self._replace_fn(self._cfg, series, chapter, self)
        if changed:
            selected = self.selected_ids()
            self._open(series, chapter)
            self._reselect(selected)
            self._invalidate_preview()
            self.status_label.setText(f"find and replace changed {changed} line(s)")
        return changed

    def set_brush(self, on: bool) -> None:
        """Turn the cleanup brush on (draw mode off, the Preview shown to see the result) or off."""
        if on:
            self.draw_button.setChecked(False)
            self.cuts_button.setChecked(False)
        self.brush_button.setChecked(on)
        self.strip.set_brush(self.brush_size.value() if on else None)
        if on and not self.preview.isVisible():
            self.set_preview_visible(True)

    def clean_stroke(self, points: list[tuple[float, float]], radius: int) -> bool:
        """Clean a stroke painted on the strip (strip points, the brush radius) with the picked method, on a worker
        thread; the part on another page than the stroke's middle is left out. False when it cannot be cleaned."""
        session = self._session
        if session is None or self._worker is not None:
            return False
        width, height = self.strip.strip_size()
        shape = services.stroke_mask(points, radius, width, height)
        ingest = services.load_ingest(session.paths)
        if shape is None or ingest is None:
            self.status_label.setText("the stroke is not on the strip")
            return False
        try:
            stroke = services.stroke_on_page(ingest, *shape)
        except ValueError as error:
            self.status_label.setText(str(error))
            return False
        method = cast(CleanMethod, self.clean_combo.currentData())
        cfg, paths = self._cfg, session.paths
        self._start_task(
            lambda _progress: (stroke.page, self._clean_fn(cfg, paths, stroke, method)),
            self._on_cleaned,
            f"cleaning a stroke on page {stroke.page + 1} ({method})...",
        )
        return True

    def set_cut_mode(self, on: bool) -> None:
        """Show the chapter's output cuts on the strip and edit them with the mouse (Draw box and the brush go
        off), or hide them."""
        if on:
            self.draw_button.setChecked(False)
            self.brush_button.setChecked(False)
        self.cuts_button.setChecked(on)
        self.strip.set_cut_mode(on)
        self._show_cuts()

    def change_cuts(self, cuts: list[int] | None) -> bool:
        """Store where the exported images split (strip rows; None: one image per slice) as its own undo step
        (unsaved edits are saved first); False when no chapter is open or the cuts are refused."""
        session = self._session
        if session is None or self._worker is not None:
            return False
        try:
            stored = session.set_cuts(cuts, max_height=services.cut_max_height(self._cfg, session.paths))
        except (LookupError, OSError, ValueError) as error:
            self.status_label.setText(f"cuts not changed: {error}")
            return False
        self._refresh()
        self._show_cuts()
        self.status_label.setText(
            "output cuts: one image per slice"
            if stored is None
            else f"output cuts saved: {len(stored) + 1} images"
        )
        return True

    def take_back_stroke(self) -> bool:
        """Remove the last stroke cleaned in this chapter; False when there is none."""
        if self._session is None or not self._strokes:
            return False
        patch_id, page = self._strokes.pop()
        try:
            services.remove_patch(self._session.paths, patch_id)
        except (LookupError, OSError, ValueError) as error:
            self.status_label.setText(f"cannot take back {patch_id}: {error}")
            return False
        self.status_label.setText(f"{patch_id}: taken back")
        self._render_page(page)
        self._update_buttons()
        return True

    def read_finished(self) -> bool:
        """Re-read the chapter's finished pages on a worker thread, then show what still shows in the Issues
        column; False when no chapter is open, a model call or a re-letter is running."""
        session = self._session
        if session is None or self._tasks or self._worker is not None:
            return False
        cfg, paths = self._cfg, session.paths
        self._start_task(
            lambda _progress: self._qa_fn(cfg, paths), self._on_finished_read, "reading the finished pages..."
        )
        return True

    def check_consistency(self) -> bool:
        """Build the series' consistency report on a worker thread, then show it; False when no chapter is open
        or a model call is running."""
        session = self._session
        if session is None or self._tasks:
            return False
        cfg, series = self._cfg, session.paths.series
        self._start_task(
            lambda _progress: self._consistency_fn(cfg, series),
            self._on_consistency,
            "checking the series...",
        )
        return True

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

    def is_busy(self) -> bool:
        """Whether a model call or a preview render is in flight."""
        return bool(self._tasks)

    def set_preview_visible(self, visible: bool) -> None:
        """Show (and render) or hide the preview of the current page."""
        self.preview.setVisible(visible)
        self.preview_button.setChecked(visible)
        if visible:
            self._render_preview()

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

    def _on_page(self, page: int) -> None:
        """Scroll the strip to the page's first region and refilter the table."""
        first = next((row for row in self._rows if row.page == page), None)
        if first is not None and not self._filling:
            box = next((box for box in self.strip.overlays() if box[0] == first.region_id), None)
            if box is not None:
                self.strip.set_strip_y(max(0.0, box[2] - 40.0))
        self._apply_filter()
        if self.preview.isVisible():
            self._render_preview()

    def _on_kind_picked(self, index: int) -> None:
        """The kind combo: apply the picked kind to the selection, then show the placeholder again."""
        picked = self.kind_combo.itemText(index)
        if picked in KINDS:
            self.set_kind(cast(RegionKind, picked))
        self.kind_combo.setCurrentIndex(0)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """An edited cell: hand the new text to the session."""
        if self._filling or self._session is None or not 0 <= item.row() < len(self._rows):
            return
        region_id = self._rows[item.row()].region_id
        if item.column() == _SPEAKER_COL:
            self._session.set_speaker(region_id, item.text())
        elif item.column() == _SOURCE_COL:
            self._session.set_source(region_id, item.text())
        elif item.column() == _ENGLISH_COL:
            self._session.set_translation(region_id, item.text())
        self._rows = self._session.rows()
        self._show_status_cell(item.row())
        self._update_buttons()

    def _on_selection(self) -> None:
        """Highlight the current row's box and scroll the strip to it."""
        row = self.table.currentRow()
        selected = self._rows[row].region_id if 0 <= row < len(self._rows) else None
        self.strip.select(selected)
        if selected is not None:
            box = next((box for box in self.strip.overlays() if box[0] == selected), None)
            if box is not None and not self._in_view(box[2], box[4]):
                self.strip.set_strip_y(max(0.0, box[2] - 40.0))
        self._update_buttons()

    def _on_box_changed(self, region_id: str, x0: int, y0: int, x1: int, y1: int) -> None:
        """A box moved or resized on the strip: pending in the session until Save."""
        if self._session is None:
            return
        self._session.set_bbox(region_id, BBox(x0=x0, y0=y0, x1=x1, y1=y1))
        self._rows = self._session.rows()
        for row, item in enumerate(self._rows):
            if item.region_id == region_id:
                self._show_status_cell(row)
        self._update_buttons()

    def _on_translated(self, suggestions: object) -> None:
        """The model answered: its lines go into the English column (kept on Save)."""
        if self._session is None or not isinstance(suggestions, list):
            return
        profiles: set[str] = set()
        for item in cast(list[Suggested], suggestions):
            try:
                self._session.set_translation(item.region_id, item.text)
            except KeyError:
                continue
            profiles.add(item.profile)
        self._refresh()
        self.status_label.setText(
            f"{len(suggestions)} suggestion(s) from {', '.join(sorted(profiles))} — Save to keep them"
            if suggestions
            else "no suggestion"
        )

    def _on_read(self, result: object) -> None:
        """The OCR read the box again: the reading becomes its source text (kept on Save)."""
        if self._session is None or not isinstance(result, tuple):
            return
        region_id, text = cast(tuple[str, str], result)
        try:
            self._session.set_source(region_id, text)
        except KeyError:
            return
        self._refresh()
        self.status_label.setText(f"{region_id} read again: {text!r} — Save to keep it")

    def _on_draw_toggled(self, drawing: bool) -> None:
        """Draw box on: the brush and the cut editing go off (one tool at a time)."""
        if drawing:
            self.brush_button.setChecked(False)
            self.cuts_button.setChecked(False)
        self.strip.set_draw_mode(drawing)

    def _on_cut_added(self, row: int) -> None:
        """A click in cut mode: a new cut there (snapped), starting from the slicer's cuts when none were set."""
        if self._cuts is not None:
            self.change_cuts(sorted({*self._cuts.cuts, self._snapped(row)}))

    def _on_cut_moved(self, old: int, new: int) -> None:
        """A cut dragged to another row (snapped)."""
        if self._cuts is not None:
            self.change_cuts(sorted({*(cut for cut in self._cuts.cuts if cut != old), self._snapped(new)}))

    def _on_cut_removed(self, row: int) -> None:
        """A right-click on a cut; removing the last one goes back to one image per slice."""
        if self._cuts is not None:
            self.change_cuts([cut for cut in self._cuts.cuts if cut != row] or None)

    def _snapped(self, row: int) -> int:
        """`row`, moved into a nearby uniform band when snapping is on."""
        if self._cuts is None or not self.snap_check.isChecked():
            return row
        return services.snap_to_band(row, self._cuts.bands)

    def _show_cuts(self) -> None:
        """Read the chapter's output cuts and draw them while Output cuts is on, with how many images they make and
        which regions they run through."""
        session = self._session
        if session is None or not self.cuts_button.isChecked():
            self._cuts = None
            self.strip.set_cut_lines(())
            self.cuts_label.clear()
            self._update_buttons()
            return
        try:
            state = services.load_cuts(session.paths)
        except (LookupError, OSError, ValueError) as error:
            self._cuts = None
            self.strip.set_cut_lines(())
            self.cuts_label.setText(str(error))
            self._update_buttons()
            return
        self._cuts = state
        self.strip.set_cut_lines(state.cuts, {cut for cut, _ in state.crossings})
        text = f"{len(state.cuts) + 1} images" + (" (one per slice)" if state.hand is None else "")
        if state.crossings:
            text += " · cuts through " + ", ".join(sorted({region_id for _, region_id in state.crossings}))
        self.cuts_label.setText(text)
        self._update_buttons()

    def _on_cleaned(self, result: object) -> None:
        """A stroke was cleaned: remember it for Take back and show its page again."""
        if self._session is None or not isinstance(result, tuple):
            return
        page, patch = cast(tuple[int, CleanupPatch], result)
        self._strokes.append((patch.id, page))
        self.status_label.setText(f"{patch.id}: cleaned ({patch.method}, {patch.mask_px} px)")
        self._render_page(page)
        self._update_buttons()

    def _on_finished_read(self, issues: object) -> None:
        """The finished pages were read again: show every row's issues, these included."""
        if self._session is None or not isinstance(issues, list):
            return
        self.run_check()
        lines = len({issue.region_id for issue in cast(list[QaIssue], issues)})
        self.status_label.setText(
            f"finished pages: {lines} line(s) still show source text or a watermark"
            if issues
            else "finished pages: no source text or watermark left"
        )

    def _on_consistency(self, report: object) -> None:
        """The report is ready: show it (a double-clicked row opens its line here)."""
        if self._session is None or not isinstance(report, Consistency):
            return
        series = self._session.paths.series
        self.status_label.setText(
            f"{len(report.divergences)} line(s) translated differently, {len(report.misses)} missed locked term(s)"
        )
        if self.consistency_dialog is not None:
            self.consistency_dialog.close()
        dialog = ConsistencyDialog(series, report, parent=self)
        dialog.open_line.connect(lambda chapter, region_id: self.open_line(series, chapter, region_id))
        self.consistency_dialog = dialog
        dialog.show()

    def open_line(self, series: str, chapter: str, region_id: str) -> bool:
        """Open `chapter` (another chapter's unsaved edits are saved first) and select `region_id`, with the row
        filters off; False when the chapter or the region cannot be shown."""
        session = self._session
        if session is None or (session.paths.series, session.paths.chapter) != (series, chapter):
            if session is not None and session.dirty:
                self.save()
            if not self.open_chapter(series, chapter):
                return False
        if not any(row.region_id == region_id for row in self._rows):
            return False
        self.page_only.setChecked(False)
        self.issues_only.setChecked(False)
        self.select_region(region_id)
        return True

    def _on_found(self, result: object) -> None:
        """The detector searched a page: ask which areas to add, then add them as regions (each saved at once,
        its own undo step) and select them."""
        if self._session is None or not isinstance(result, tuple):
            return
        page, found = cast(tuple[int, list["Found"]], result)
        if not found:
            self.status_label.setText(f"page {page + 1}: no missed text found")
            return
        added: list[str] = []
        for area in self._pick_fn(found, page, self):
            try:
                added.append(self._session.add_region(area.bbox, kind=area.kind, text=area.text))
            except (LookupError, OSError, ValueError) as error:
                self.status_label.setText(f"cannot add a box: {error}")
                break
        else:
            self.status_label.setText(
                f"page {page + 1}: added {len(added)} of {len(found)} found region(s) — Translate them next"
                if added
                else f"page {page + 1}: {len(found)} found, none added"
            )
        if added:
            self._refresh()
            self._reselect(added)
            self._invalidate_preview()

    def _on_preview(self, result: object) -> None:
        """A page rendered: show it in the preview strip."""
        if not isinstance(result, PagePreview):
            return
        self.preview.provide_image(services.preview_path(result.page), _qimage(result.pixels))

    def _on_task_failed(self, text: str) -> None:
        """A model call or render raised."""
        self.status_label.setText(f"failed: {text}")

    def _on_run_finished(self, outcome: RunOutcome) -> None:
        """Re-letter ended."""
        self.status_label.setText("re-lettered: output updated" if outcome.ok else "re-letter failed")
        self._invalidate_preview()

    def _on_run_failed(self, text: str) -> None:
        """Re-letter raised."""
        self.status_label.setText(f"re-letter failed: {text}")

    def _follow(self, source: StripView, *, y: float | None = None, zoom: float | None = None) -> None:
        """Keep the preview and the raw strip aligned (same strip y, same zoom), without echo signals."""
        if self._syncing or not self.preview.isVisible():
            return
        other = self.preview if source is self.strip else self.strip
        self._syncing = True
        try:
            if y is not None:
                other.set_strip_y(y, emit=False)
            if zoom is not None:
                other.set_zoom(zoom, emit=False)
        finally:
            self._syncing = False

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
        # the page's own config (not the process-wide one), with the series' series.toml applied
        direction = series_config(self._cfg, paths.raw_dir.parent).detect.reading_direction
        session = StudioSession(paths, direction=direction)
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
        self._strokes = []
        self._issues, self._typos = {}, {}
        voices_error = None
        try:
            self._speaker_delegate.names = services.speaker_names(paths)
        except ValueError as error:  # a broken voices.toml: the column still takes free text
            self._speaker_delegate.names = []
            voices_error = str(error)
        ingest = services.load_ingest(paths)
        self._page_rows = services.page_rows(ingest) if ingest is not None else []
        self._reset_preview(view.strip_width, view.strip_height)
        self._refresh()
        self._show_cuts()
        self.run_check()
        self._set_pages()
        if voices_error is not None:  # after the check's own line, so it is the one shown
            self.status_label.setText(f"{voices_error} (the Speaker column offers no names)")
        if self.preview.isVisible():
            self._render_preview()

    def _close(self) -> None:
        """No chapter open."""
        self._session = None
        self._strokes = []
        self._rows = []
        self._issues, self._typos = {}, {}
        self._page_rows = []
        self.strip.set_overlays(())
        self._cuts = None
        self.strip.set_cut_lines(())
        self.cuts_label.clear()
        self.preview.set_tiles((), 0, 0)
        self._rendered.clear()
        self._fill_table()
        self._set_pages()

    def _refresh(self) -> None:
        """Re-read the rows from the session and redraw."""
        self._rows = self._session.rows() if self._session is not None else []
        selected = self.selected_ids()
        self._fill_table()
        self._reselect(selected)
        self._set_pages()

    def _reselect(self, ids: Sequence[str]) -> None:
        """Select the rows of `ids` again after the table was rebuilt."""
        if not ids:
            return
        wanted = set(ids)
        rows = [row for row, item in enumerate(self._rows) if item.region_id in wanted]
        first = next((row for row, item in enumerate(self._rows) if item.region_id == ids[0]), None)
        if first is not None:
            rows.remove(first)
            rows.insert(0, first)
        self._select_rows(rows)

    def _history(self, step: str) -> bool:
        """Undo or redo, then show the chapter again."""
        if self._session is None or self._worker is not None:
            return False
        done = self._session.undo() if step == "undo" else self._session.redo()
        self._issues, self._typos = {}, {}
        self._refresh()
        self._show_cuts()
        self.run_check()
        self.status_label.setText(f"{step}: done" if done else f"nothing to {step}")
        if done:
            self._invalidate_preview()
        return done

    def _pages(self) -> list[int]:
        """The pages (slice indices) that hold regions."""
        return sorted({row.page for row in self._rows})

    def _set_pages(self) -> None:
        """Fit the page spinner to the chapter's pages and show the progress counts."""
        pages = self._pages()
        self.page_spin.blockSignals(True)
        try:
            if pages:
                self.page_spin.setRange(pages[0], pages[-1])
                if self.page_spin.value() not in pages:
                    self.page_spin.setValue(pages[0])
            else:
                self.page_spin.setRange(0, 0)
        finally:
            self.page_spin.blockSignals(False)
        if self._session is None:
            self.progress_label.setText("")
        else:
            counts = self._session.counts()
            self.progress_label.setText(
                f"{counts['todo']} todo · {counts['edited']} edited · {counts['checked']} checked"
            )

    def _fill_table(self) -> None:
        """Rebuild the table from the rows and issues."""
        self._filling = True
        try:
            self.table.setRowCount(len(self._rows))
            for row, item in enumerate(self._rows):
                cells = (
                    str(item.page),
                    item.kind,
                    item.speaker,
                    item.source,
                    item.english,
                    self._status_text(item),
                    "; ".join(self._issues.get(item.region_id, [])),
                )
                for column, text in enumerate(cells):
                    cell = QTableWidgetItem(text)
                    if column not in _EDITABLE:
                        cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    if column == _ENGLISH_COL and item.machine and item.english != item.machine:
                        cell.setToolTip(f"machine: {item.machine}")
                    self.table.setItem(row, column, cell)
        finally:
            self._filling = False
        self._apply_filter()
        self._draw_overlays()
        self._update_buttons()

    @staticmethod
    def _status_text(item: StudioRow) -> str:
        """A row's status cell: its review state, plus a mark for hand-set lettering."""
        return f"{item.status} · lettered" if item.lettered else item.status

    def _show_status_cell(self, row: int) -> None:
        """Refresh one row's status cell after an edit."""
        cell = self.table.item(row, _STATUS_COL)
        if cell is not None and 0 <= row < len(self._rows):
            self._filling = True
            try:
                cell.setText(self._status_text(self._rows[row]))
            finally:
                self._filling = False

    def _apply_filter(self) -> None:
        """Hide the rows the filters leave out (other pages, lines without issues)."""
        only_issues = self.issues_only.isChecked()
        only_page = self.page_only.isChecked()
        page = self.page_spin.value()
        for row, item in enumerate(self._rows):
            hidden = (only_issues and item.region_id not in self._issues) or (only_page and item.page != page)
            self.table.setRowHidden(row, hidden)

    def _draw_overlays(self) -> None:
        """Outline every remaining region on the strip (the selection stays)."""
        if self._session is None:
            self.strip.set_overlays(())
            return
        boxes = [
            (region.id, region.bbox.x0, region.bbox.y0, region.bbox.x1, region.bbox.y1)
            for region in self._session.regions()
        ]
        self.strip.set_overlays(boxes, self.strip.selected())

    def _page_in_view(self) -> int | None:
        """The raw page (SourceFile index) at the middle of the strip's view; None before ingest.json exists."""
        if not self._page_rows:
            return None
        middle = self.strip.strip_y() + self.strip.viewport().height() / self.strip.zoom() / 2
        for page, y0, y1 in self._page_rows:
            if y0 <= middle < y1:
                return page
        return self._page_rows[0][0] if middle < self._page_rows[0][1] else self._page_rows[-1][0]

    def _in_view(self, y0: int, y1: int) -> bool:
        """Whether strip rows y0..y1 are (at least partly) visible in the strip."""
        top = self.strip.strip_y()
        bottom = top + self.strip.viewport().height() / self.strip.zoom()
        return y1 > top and y0 < bottom

    # ---- preview

    def _reset_preview(self, strip_width: int, strip_height: int) -> None:
        """Lay the preview strip out like the raw one: one tile per raw page, rendered on demand."""
        tiles = tuple(
            Tile(y0, y1, services.preview_path(page), f"page {page + 1}: not rendered yet", "image")
            for page, y0, y1 in self._page_rows
        )
        self.preview.set_tiles(tiles, strip_width, strip_height)
        self._rendered.clear()

    def _invalidate_preview(self) -> None:
        """The chapter changed on disk: the rendered pages are stale; render the current one again if shown."""
        if self._session is None:
            return
        self._reset_preview(*self.preview.strip_size())
        if self.preview.isVisible():
            self._render_preview()

    def _preview_pages(self) -> list[int]:
        """The raw pages the current page's (slice's) regions lie on."""
        page = self.page_spin.value()
        rows = [row for row in self._rows if row.page == page]
        if self._session is None or not rows or not self._page_rows:
            return []
        wanted = {row.region_id for row in rows}
        boxes = [region.bbox for region in self._session.regions() if region.id in wanted]
        top, bottom = min(box.y0 for box in boxes), max(box.y1 for box in boxes)
        return [index for index, y0, y1 in self._page_rows if y1 > top and y0 < bottom]

    def _render_preview(self) -> None:
        """Render the current page's raw pages that are not rendered yet, each on a worker thread."""
        if self._session is None:
            return
        cfg, paths = self._cfg, self._session.paths
        for page in self._preview_pages():
            if page in self._rendered:
                continue
            self._rendered.add(page)
            self._start_task(
                lambda _progress, page=page: self._preview_fn(cfg, paths, page), self._on_preview, None
            )

    def _render_page(self, page: int) -> None:
        """Render raw page `page` into the preview again (a stroke changed it); nothing while it is hidden."""
        if self._session is None or not self.preview.isVisible():
            self._rendered.discard(page)  # rendered again when the preview opens there
            return
        cfg, paths = self._cfg, self._session.paths
        self._rendered.add(page)
        self._start_task(lambda _progress: self._preview_fn(cfg, paths, page), self._on_preview, None)

    # ---- worker tasks

    def _start_task(
        self, fn: Callable[[Any], object], done: Callable[[object], None], status: str | None
    ) -> None:
        """Run `fn` on a pool thread; `done` gets its result on the GUI thread."""
        signals = run_task(fn)
        self._tasks.append(signals)
        signals.finished.connect(done)
        signals.failed.connect(self._on_task_failed)
        signals.finished.connect(lambda _result, s=signals: self._end_task(s))
        signals.failed.connect(lambda _text, s=signals: self._end_task(s))
        if status is not None:
            self.status_label.setText(status)
        self._update_buttons()

    def _end_task(self, signals: WorkerSignals) -> None:
        """A task finished (or failed)."""
        if signals in self._tasks:
            self._tasks.remove(signals)
        self._update_buttons()

    def _update_buttons(self) -> None:
        """Enable the actions that make sense now."""
        session = self._session
        is_open = session is not None
        idle = self._worker is None
        free = idle and not self._tasks  # no model call or render in flight
        ids = self.selected_ids()
        some = bool(ids)
        steps = store.history_steps(session.paths) if session is not None else (0, 0)
        self.undo_button.setEnabled(is_open and idle and steps[0] > 0)
        self.redo_button.setEnabled(is_open and idle and steps[1] > 0)
        self.check_button.setEnabled(is_open)
        self.not_typo_button.setEnabled(is_open and any(region_id in self._typos for region_id in ids))
        self.translate_button.setEnabled(is_open and free and some)
        self.profile_combo.setEnabled(is_open and free)
        self.read_button.setEnabled(is_open and free and some)
        self.revert_button.setEnabled(is_open and idle and some)
        self.checked_button.setEnabled(is_open and idle and some)
        self.unchecked_button.setEnabled(is_open and idle and some)
        self.kind_combo.setEnabled(is_open and idle)
        self.lettering_button.setEnabled(is_open and idle and some)
        self.draw_button.setEnabled(is_open and idle)
        self.remove_button.setEnabled(is_open and idle and some)
        self.find_button.setEnabled(is_open and free and bool(self._page_rows))
        self.find_threshold.setEnabled(is_open)
        self.save_button.setEnabled(is_open and idle and session is not None and session.dirty)
        self.reletter_button.setEnabled(is_open and idle)
        self.preview_button.setEnabled(is_open)
        self.replace_button.setEnabled(is_open and idle)
        self.consistency_button.setEnabled(is_open and free)
        self.qa_button.setEnabled(is_open and free)
        self.brush_button.setEnabled(is_open and idle and bool(self._page_rows))
        self.take_back_button.setEnabled(is_open and idle and bool(self._strokes))
        self.cuts_button.setEnabled(is_open and idle)
        self.reset_cuts_button.setEnabled(
            is_open and idle and self._cuts is not None and self._cuts.hand is not None
        )
        self.prev_chapter_button.setEnabled(self.chapter_combo.currentIndex() > 0)
        self.next_chapter_button.setEnabled(
            self.chapter_combo.currentIndex() < self.chapter_combo.count() - 1
        )
        pages = self._pages()
        self.prev_page_button.setEnabled(bool(pages) and self.page_spin.value() > pages[0])
        self.next_page_button.setEnabled(bool(pages) and self.page_spin.value() < pages[-1])

    def _release_worker(self, worker: RunWorker) -> None:
        """Drop the finished worker."""
        if self._worker is worker:
            self._worker = None
            self._update_buttons()
            self.busy_changed.emit(False)

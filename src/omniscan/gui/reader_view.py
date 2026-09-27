"""ReaderView: the Reader page — the existing CompareView with chapter navigation and viewer controls.

Chapters come from `gui.services.library.list_chapter_names` (injectable for tests); the compare
view itself is reused unchanged. The toolbar adds prev/next, a chapter switcher, zoom buttons, the
side selector (both / raw / output) and a jump-to-slice combo; linked scrolling and fit-width stay
on the CompareView's own toolbar.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from omniscan.core.config import Config
from omniscan.gui.compare_view import CompareView
from omniscan.gui.services import library
from omniscan.gui.strip_view import StripView
from omniscan.gui.theme import set_role

ChaptersFn = Callable[[Config, str], list[str]]
READING_WIDTH_PX = 900  # the reading column never gets wider than this, however wide the screen
_PAGE_STEP = 0.9  # Space / Page Down scroll this share of the screen, so a line of context stays


class ReaderView(QWidget):
    """One chapter at a time, raw left and output right, with prev/next over the series' chapters.

    `Read` switches to reading mode: the translated pages only, in a centred column, full screen, paged with the
    keyboard (Space / Page Down, Page Up, arrows, N / P for the next / previous chapter, Esc to leave); the
    position in each chapter is remembered when `qsettings` is given.
    """

    reading_changed = Signal(bool)  # the window hides its chrome and goes full screen while reading

    def __init__(
        self,
        cfg: Config,
        *,
        chapters_fn: ChaptersFn | None = None,
        qsettings: QSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build the page; `chapters_fn` overrides the chapter-list source (tests inject it)."""
        super().__init__(parent)
        self._qsettings = qsettings
        self._reading = False
        self._cfg = cfg
        self._chapters_fn = chapters_fn or library.list_chapter_names
        self._series: str | None = None
        self._chapters: list[str] = []
        self._index = -1
        self._jump_ys: list[int] = []

        self.compare = CompareView(self)
        self.status_label = QLabel(self)

        self.prev_button = QPushButton("<", self)
        self.prev_button.setToolTip("Previous chapter")
        self.next_button = QPushButton(">", self)
        self.next_button.setToolTip("Next chapter")
        self.chapter_combo = QComboBox(self)
        self.chapter_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.zoom_out_button = QPushButton("-", self)
        self.zoom_out_button.setToolTip("Zoom out")
        self.zoom_in_button = QPushButton("+", self)
        self.zoom_in_button.setToolTip("Zoom in")
        self.sides_combo = QComboBox(self)
        self.sides_combo.addItems(("Both", "Raw only", "Output only"))
        self.sides_combo.setToolTip("Which side(s) to show")
        self.jump_combo = QComboBox(self)
        self.jump_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.read_button = QPushButton("Read", self)
        self.read_button.setToolTip("Reading mode: translated pages only, full screen (Esc to leave)")
        set_role(self.read_button, "primary")

        bar_layout = QHBoxLayout()
        for widget in (
            self.prev_button,
            self.next_button,
            QLabel("Chapter", self),
            self.chapter_combo,
            self.zoom_out_button,
            self.zoom_in_button,
            QLabel("Sides", self),
            self.sides_combo,
            self.jump_combo,
        ):
            bar_layout.addWidget(widget)
        bar_layout.addStretch(1)
        bar_layout.addWidget(self.read_button)
        self.toolbar = QWidget()
        self.toolbar.setLayout(bar_layout)

        root = QVBoxLayout(self)
        root.addWidget(self.toolbar)
        root.addWidget(self.compare, 1)
        root.addWidget(self.status_label)

        self.prev_button.clicked.connect(lambda: self._open_at(self._index - 1))
        self.next_button.clicked.connect(lambda: self._open_at(self._index + 1))
        self.chapter_combo.activated.connect(self._open_at)
        self.zoom_out_button.clicked.connect(lambda: self.compare.zoom_by(0.8))
        self.zoom_in_button.clicked.connect(lambda: self.compare.zoom_by(1.25))
        self.sides_combo.currentIndexChanged.connect(self._on_sides_changed)
        self.jump_combo.activated.connect(self._on_jump_activated)
        self.read_button.clicked.connect(lambda: self.set_reading(True))
        self._update_nav()

    # ------------------------------------------------------------------ state

    def open_chapter(self, series: str, chapter: str) -> bool:
        """Show one chapter (rebuilding the switcher when the series changes); False when it fails."""
        if series != self._series:
            self._series = series
            self._chapters = list(self._chapters_fn(self._cfg, series))
            self.chapter_combo.blockSignals(True)
            try:
                self.chapter_combo.clear()
                self.chapter_combo.addItems(self._chapters)
            finally:
                self.chapter_combo.blockSignals(False)
        try:
            index = self._chapters.index(chapter)
        except ValueError:
            return False
        self._open_at(index)
        return self.current() == (series, chapter)

    def current(self) -> tuple[str, str] | None:
        """The open (series, chapter), or None before the first successful open."""
        if self._series is None or not 0 <= self._index < len(self._chapters):
            return None
        return self._series, self._chapters[self._index]

    def reconfigure(self, cfg: Config) -> None:
        """Swap the config source (a settings change); the next open uses the new paths."""
        self._cfg = cfg

    # ------------------------------------------------------------------ reading mode

    def is_reading(self) -> bool:
        """Whether reading mode is on."""
        return self._reading

    def reading_strip(self) -> StripView:
        """The pane reading mode shows: the output, or the raw pages before a chapter is translated."""
        return self.compare.right if self.compare.right.tiles() else self.compare.left

    def set_reading(self, on: bool) -> None:
        """Enter or leave reading mode."""
        if on == self._reading or (on and self.current() is None):
            return
        self._reading = on
        compare = self.compare
        for widget in (
            self.toolbar,
            self.status_label,
            compare.sync_checkbox,
            compare.fit_button,
            compare.chapter_label,
            compare.left_caption,
            compare.right_caption,
        ):
            widget.setVisible(not on)
        if on:
            compare.set_visible_sides("output" if self.reading_strip() is compare.right else "raw")
        else:
            self._on_sides_changed(self.sides_combo.currentIndex())
        for strip in (compare.left, compare.right):
            strip.set_max_fit_width(READING_WIDTH_PX if on else None)
        self._restore_position()
        if on:
            self.setFocus(Qt.FocusReason.OtherFocusReason)
        self.reading_changed.emit(on)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Reading-mode keys; everything else goes to the default handler."""
        if not self._reading:
            super().keyPressEvent(event)
            return
        key = Qt.Key(event.key())
        strip = self.reading_strip()
        if key == Qt.Key.Key_Escape:
            self.set_reading(False)
        elif key in (Qt.Key.Key_Space, Qt.Key.Key_PageDown):
            if strip.at_end():
                self._open_at(self._index + 1)
            else:
                strip.scroll_screens(_PAGE_STEP)
        elif key == Qt.Key.Key_PageUp:
            strip.scroll_screens(-_PAGE_STEP)
        elif key == Qt.Key.Key_Down:
            strip.scroll_screens(0.15)
        elif key == Qt.Key.Key_Up:
            strip.scroll_screens(-0.15)
        elif key in (Qt.Key.Key_N, Qt.Key.Key_Right):
            self._open_at(self._index + 1)
        elif key in (Qt.Key.Key_P, Qt.Key.Key_Left):
            self._open_at(self._index - 1)
        elif key == Qt.Key.Key_Home:
            strip.set_strip_y(0.0)
        elif key == Qt.Key.Key_End:
            strip.set_strip_y(strip.max_strip_y())
        else:
            super().keyPressEvent(event)
            return
        self._save_position()

    def _position_key(self) -> str | None:
        """QSettings key of the open chapter's reading position."""
        current = self.current()
        return None if current is None else f"reader/position/{current[0]}/{current[1]}"

    def _save_position(self) -> None:
        """Remember how far the open chapter was read (share of the strip, so zoom does not matter)."""
        key = self._position_key()
        strip = self.reading_strip()
        if self._qsettings is None or key is None or strip.max_strip_y() <= 0:
            return
        self._qsettings.setValue(key, strip.strip_y() / strip.max_strip_y())

    def _restore_position(self) -> None:
        """Scroll reading mode back to where the open chapter was left."""
        key = self._position_key()
        if not self._reading or self._qsettings is None or key is None:
            return
        share = cast(float, self._qsettings.value(key, 0.0, type=float))
        strip = self.reading_strip()
        strip.set_strip_y(min(max(share, 0.0), 1.0) * strip.max_strip_y())

    # ------------------------------------------------------------------ internals

    def _open_at(self, index: int) -> None:
        """Open the chapter at switcher position `index` (clamped to the valid range)."""
        if self._series is None or not 0 <= index < len(self._chapters):
            return
        chapter = self._chapters[index]
        try:
            view = library.load_chapter_view(self._cfg, self._series, chapter)
        except (OSError, ValueError) as error:
            self.status_label.setText(f"cannot open {chapter}: {error}")
            return
        self._save_position()
        self.compare.set_chapter(view)
        self._index = index
        self.chapter_combo.blockSignals(True)
        try:
            self.chapter_combo.setCurrentIndex(index)
        finally:
            self.chapter_combo.blockSignals(False)
        self._rebuild_jump()
        self._update_nav()
        if self._reading:
            compare = self.compare
            compare.set_visible_sides("output" if self.reading_strip() is compare.right else "raw")
            self.reading_strip().set_max_fit_width(READING_WIDTH_PX)
            self._restore_position()
        self.status_label.setText(
            f"{view.series} — {view.chapter}: {len(view.raw)} raw page(s), "
            f"{len(view.output)} output tile(s), strip {view.strip_width}x{view.strip_height}"
        )

    def _rebuild_jump(self) -> None:
        """Fill the jump combo from the output tiles (raw pages when there is no output)."""
        self._jump_ys = []
        self.jump_combo.blockSignals(True)
        try:
            self.jump_combo.clear()
            self.jump_combo.addItem("Jump to slice...")
            tiles = self.compare.right.tiles() or self.compare.left.tiles()
            for tile in tiles:
                self.jump_combo.addItem(f"{tile.label}  (y={tile.y0})")
                self._jump_ys.append(tile.y0)
        finally:
            self.jump_combo.blockSignals(False)

    def _on_jump_activated(self, index: int) -> None:
        """Scroll both sides to the chosen tile's strip y (works in every side mode)."""
        if 0 < index <= len(self._jump_ys):
            y = float(self._jump_ys[index - 1])
            self.compare.left.set_strip_y(y, emit=False)
            self.compare.right.set_strip_y(y, emit=False)

    def _on_sides_changed(self, index: int) -> None:
        """Map the combo row to CompareView's side mode."""
        self.compare.set_visible_sides(("both", "raw", "output")[max(0, index)])

    def _update_nav(self) -> None:
        """Enable prev/next only when the target index exists."""
        self.prev_button.setEnabled(self._index > 0)
        self.next_button.setEnabled(0 <= self._index < len(self._chapters) - 1)

"""CompareView: raw chapter on the left, translated output on the right, linked by strip space.

Because both sides share the strip y coordinates (see `omniscan.gui.services.library`), linked
scrolling simply means "same strip y, same zoom on both sides"; the view that scrolled or zoomed
last is the master, the other one follows silently (no signal feedback loops).
"""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from omniscan.gui.services.library import ChapterView
from omniscan.gui.strip_view import StripView

type SyncMode = Literal["linked", "independent"]


class CompareView(QWidget):
    """The reader core: two `StripView`s over one chapter, with linked or independent scrolling."""

    sync_mode_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build an empty compare view (no chapter loaded yet)."""
        super().__init__(parent)
        self._mode: SyncMode = "linked"
        self._master: StripView | None = None  # the view scrolled/zoomed last
        self._syncing = False  # True while CompareView itself drives both views

        self.sync_checkbox = QCheckBox("Linked scrolling")
        self.sync_checkbox.setChecked(True)
        self.fit_button = QPushButton("Fit width")
        self.chapter_label = QLabel("")
        self.left_caption = QLabel("Raw")
        self.right_caption = QLabel("Output")
        self.left = StripView()
        self.right = StripView()

        top = QHBoxLayout()
        top.addWidget(self.sync_checkbox)
        top.addWidget(self.fit_button)
        top.addStretch(1)
        top.addWidget(self.chapter_label)

        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.left_caption)
        left_layout.addWidget(self.left)

        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.right_caption)
        right_layout.addWidget(self.right)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(left_pane)
        self.splitter.addWidget(right_pane)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([1, 1])  # even halves by default

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(self.splitter)

        self.sync_checkbox.toggled.connect(self._on_checkbox_toggled)
        self.fit_button.clicked.connect(self._on_fit_clicked)
        self.left.strip_y_changed.connect(lambda y, s=self.left: self._on_strip_y(s, y))
        self.right.strip_y_changed.connect(lambda y, s=self.right: self._on_strip_y(s, y))
        self.left.zoom_changed.connect(lambda z, s=self.left: self._on_zoom(s, z))
        self.right.zoom_changed.connect(lambda z, s=self.right: self._on_zoom(s, z))
        self._master = self.left

    # ------------------------------------------------------------------ state

    def set_chapter(self, view: ChapterView) -> None:
        """Show one chapter: raw tiles left, output tiles right; both fitted and scrolled to 0."""
        self.chapter_label.setText(f"{view.series} — {view.chapter}")
        self.right_caption.setText("Output" if view.has_output else "Output (not translated yet)")
        self._syncing = True
        try:
            self.left.set_tiles(view.raw, view.strip_width, view.strip_height)
            self.right.set_tiles(view.output, view.strip_width, view.strip_height)
            self._master = self.left
            self.right.set_strip_y(0.0, emit=False)
            if self._mode == "linked":
                self.right.set_zoom(self.left.zoom(), emit=False)
        finally:
            self._syncing = False

    def sync_mode(self) -> SyncMode:
        """The current scrolling mode: "linked" (same strip y/zoom) or "independent"."""
        return self._mode

    def set_sync_mode(self, mode: SyncMode) -> None:
        """Switch between linked and independent scrolling; the checkbox follows along."""
        if mode == self._mode:
            return
        self._mode = mode  # set first: the checkbox handler re-enters here and must no-op
        if mode == "linked":
            master = self._master or self.left
            self._syncing = True
            try:
                self._other(master).set_strip_y(master.strip_y(), emit=False)
                self._other(master).set_zoom(master.zoom(), emit=False)
            finally:
                self._syncing = False
        self.sync_checkbox.setChecked(mode == "linked")
        self.sync_mode_changed.emit(mode)

    def strip_y(self) -> float:
        """Strip y of the master view (the one scrolled/zoomed most recently)."""
        return (self._master or self.left).strip_y()

    # ------------------------------------------------------------------ internals

    def _other(self, source: StripView) -> StripView:
        """The view that is not `source`."""
        return self.right if source is self.left else self.left

    def _on_checkbox_toggled(self, checked: bool) -> None:
        self.set_sync_mode("linked" if checked else "independent")

    def _on_fit_clicked(self) -> None:
        """Fit the width of both sides; in linked mode the emitted zooms keep them equal."""
        self.left.fit_width()
        self.right.fit_width()

    def _on_strip_y(self, source: StripView, y: float) -> None:
        """Follow the master: same strip y on both sides, without echo signals."""
        self._master = source
        if self._mode == "linked" and not self._syncing:
            self._other(source).set_strip_y(y, emit=False)

    def _on_zoom(self, source: StripView, zoom: float) -> None:
        """Follow the master: same zoom on both sides, without echo signals."""
        self._master = source
        if self._mode == "linked" and not self._syncing:
            self._other(source).set_zoom(zoom, emit=False)

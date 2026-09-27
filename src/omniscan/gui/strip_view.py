"""StripView: a vertical strip of tiles (raw pages or output slices) drawn in strip coordinates.

The view scrolls in strip y (`strip_y_changed`) and scales with a single zoom factor
(`zoom_changed`, widget pixels per strip pixel). Two `StripView`s sharing one strip space can be
linked exactly: same strip y, same zoom — this is what `omniscan.gui.compare_view` builds on.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from math import ceil
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QPen,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QAbstractScrollArea, QWidget

from omniscan.gui.services.library import Tile

Overlay = tuple[str, int, int, int, int]  # (id, x0, y0, x1, y1) in strip space

MIN_ZOOM = 0.05
MAX_ZOOM = 4.0
ZOOM_WHEEL_FACTOR = 1.15


def _load_image(path: Path) -> QImage | None:
    """Read an image file, or None when it is missing or not a valid image."""
    image = QImage(str(path))
    return None if image.isNull() else image


class StripView(QAbstractScrollArea):
    """Scrollable, zoomable renderer of a chapter strip: one tile per raw page or output slice."""

    strip_y_changed = Signal(float)  # strip y of the viewport's top edge
    zoom_changed = Signal(float)
    overlay_clicked = Signal(str)  # id of the overlay box under a left click

    def __init__(self, parent: QWidget | None = None, *, max_cached_images: int = 32) -> None:
        """Create an empty view; `set_tiles` fills it."""
        super().__init__(parent)
        self._tiles: tuple[Tile, ...] = ()
        self._strip_width = 0
        self._strip_height = 0
        self._zoom = 1.0
        self._fit_active = False  # refit the width on every viewport resize until a manual zoom
        self._sync_guard = False  # True while this view itself moves the bars (no strip_y_changed)
        self._image_cache: OrderedDict[Path, QImage] = OrderedDict()  # LRU: path -> image
        self._max_cached_images = max(1, max_cached_images)
        self._max_fit_width: int | None = (
            None  # fit_width never makes the strip wider than this (reading mode)
        )
        self._overlays: tuple[Overlay, ...] = ()
        self._selected: str | None = None
        self.verticalScrollBar().valueChanged.connect(self._on_value_changed)
        self.viewport().installEventFilter(self)

    # ------------------------------------------------------------------ tiles

    def set_tiles(self, tiles: Sequence[Tile], strip_width: int, strip_height: int) -> None:
        """Show a chapter strip; resets the scroll position to 0 and fits the width."""
        self._tiles = tuple(tiles)
        self._strip_width = int(strip_width)
        self._strip_height = int(strip_height)
        self._image_cache.clear()
        self._sync_guard = True
        try:
            self._update_bars(y=0.0)
        finally:
            self._sync_guard = False
        self.fit_width()
        self.viewport().update()

    def tiles(self) -> tuple[Tile, ...]:
        """The tiles currently displayed (empty tuple before the first `set_tiles`)."""
        return self._tiles

    def tile_at(self, strip_y: float) -> Tile | None:
        """The tile covering strip row `strip_y` (None in a gap or outside the strip)."""
        for tile in self._tiles:
            if tile.y0 <= strip_y < tile.y1:
                return tile
        return None

    # ------------------------------------------------------------------ overlays

    def set_overlays(self, overlays: Sequence[Overlay], selected: str | None = None) -> None:
        """Outline boxes over the strip (e.g. text regions); `selected` is drawn filled."""
        self._overlays = tuple(overlays)
        self._selected = selected
        self.viewport().update()

    def overlays(self) -> tuple[Overlay, ...]:
        """The overlay boxes currently drawn."""
        return self._overlays

    def overlay_at(self, x: float, y: float) -> str | None:
        """Id of the smallest overlay box containing strip point (x, y), or None."""
        hits = [box for box in self._overlays if box[1] <= x < box[3] and box[2] <= y < box[4]]
        if not hits:
            return None
        return min(hits, key=lambda box: (box[3] - box[1]) * (box[4] - box[2]))[0]

    def _strip_x_offset(self) -> float:
        """Widget x of strip x 0 (the strip is centred when narrower than the viewport)."""
        strip_w_px = self._strip_width * self._zoom
        width = self.viewport().width()
        if strip_w_px <= width:
            return (width - strip_w_px) / 2.0
        return float(-self.horizontalScrollBar().value())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """A left click on an overlay box emits its id."""
        if event.button() == Qt.MouseButton.LeftButton and self._overlays:
            pos = event.position()
            x = (pos.x() - self._strip_x_offset()) / self._zoom
            y = (pos.y() + self.verticalScrollBar().value()) / self._zoom
            hit = self.overlay_at(x, y)
            if hit is not None:
                self.overlay_clicked.emit(hit)
        super().mousePressEvent(event)

    # ------------------------------------------------------------------ zoom

    def zoom(self) -> float:
        """Widget pixels per strip pixel."""
        return self._zoom

    def set_zoom(self, zoom: float, *, emit: bool = True) -> None:
        """Clamp `zoom` to [0.05, 4.0], keep the strip y of the viewport top and redraw."""
        new_zoom = min(max(zoom, MIN_ZOOM), MAX_ZOOM)
        self._fit_active = False  # any manual zoom ends fit-width mode (even a no-op zoom)
        if new_zoom == self._zoom:
            return
        y = self.strip_y()  # strip y of the viewport top under the *old* zoom
        self._zoom = new_zoom
        self._sync_guard = True
        try:
            self._update_bars(y=y)
        finally:
            self._sync_guard = False
        self.viewport().update()
        if emit:
            self.zoom_changed.emit(new_zoom)

    def fit_width(self, *, emit: bool = True) -> None:
        """Zoom so the strip fills the viewport width (capped by `set_max_fit_width`); refits on resize."""
        width = self.viewport().width()
        if self._max_fit_width is not None:
            width = min(width, self._max_fit_width)
        if self._strip_width > 0 and width > 0:
            self.set_zoom(width / self._strip_width, emit=emit)
        self._fit_active = True

    def set_max_fit_width(self, pixels: int | None) -> None:
        """Cap the width `fit_width` fills (a readable column on a wide screen); None fills the viewport."""
        self._max_fit_width = pixels
        self.fit_width()

    def scroll_screens(self, screens: float) -> None:
        """Scroll by a fraction of the viewport height (negative scrolls up)."""
        self.set_strip_y(self.strip_y() + screens * self.viewport().height() / self._zoom)

    def at_end(self) -> bool:
        """Whether the viewport shows the bottom of the strip."""
        return self.strip_y() >= self.max_strip_y() - 0.5

    # ------------------------------------------------------------------ scrolling

    def strip_y(self) -> float:
        """Strip y of the viewport's top edge."""
        return self.verticalScrollBar().value() / self._zoom

    def set_strip_y(self, y: float, *, emit: bool = True) -> None:
        """Scroll so strip row `y` is at the viewport top (clamped); `emit=False` moves silently."""
        target = min(max(y, 0.0), self.max_strip_y())
        value = round(target * self._zoom)
        bar = self.verticalScrollBar()
        if value == bar.value():
            return
        self._sync_guard = not emit
        try:
            bar.setValue(value)
        finally:
            self._sync_guard = False
        self.viewport().update()

    def max_strip_y(self) -> float:
        """The largest strip y the viewport top can reach."""
        return max(0.0, self._strip_height - self.viewport().height() / self._zoom)

    # ------------------------------------------------------------------ Qt plumbing

    def _update_bars(self, *, y: float | None = None) -> None:
        """Recompute scrollbar ranges for the current zoom/viewport (preserving or resetting y)."""
        zoom = self._zoom
        vp = self.viewport().size()
        strip_y = self.strip_y() if y is None else y
        vbar = self.verticalScrollBar()
        vbar.setRange(0, max(0, round(self._strip_height * zoom) - vp.height()))
        vbar.setPageStep(max(1, vp.height()))
        vbar.setValue(round(strip_y * zoom))
        hbar = self.horizontalScrollBar()
        hbar.setRange(0, max(0, ceil(self._strip_width * zoom) - vp.width()))
        hbar.setPageStep(max(1, vp.width()))
        hbar.setValue(hbar.value())

    def _handle_viewport_resize(self) -> None:
        """Keep the strip y on viewport changes; refit the width while fit-width is active."""
        if self._fit_active:
            self.fit_width()
        else:
            self._sync_guard = True
            try:
                self._update_bars()
            finally:
                self._sync_guard = False

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Refit the width (or keep the strip y) when the widget is resized."""
        super().resizeEvent(event)
        self._handle_viewport_resize()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Refit/keep alignment when the viewport resizes (e.g. a scrollbar appearing)."""
        if obj is self.viewport() and event.type() == QEvent.Type.Resize:
            self._handle_viewport_resize()
        return super().eventFilter(obj, event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Ctrl + wheel zooms around the viewport top; a plain wheel scrolls the strip."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = ZOOM_WHEEL_FACTOR if event.angleDelta().y() > 0 else 1.0 / ZOOM_WHEEL_FACTOR
            self.set_zoom(self._zoom * factor)
            event.accept()
        else:
            # QAbstractScrollArea's default ignores the wheel over the viewport; the bar scrolls.
            QCoreApplication.sendEvent(self.verticalScrollBar(), event)
            event.accept()

    def _on_value_changed(self, value: int) -> None:
        """Forward scroll-bar moves as strip y (unless the view itself moved the bar)."""
        if not self._sync_guard:
            self.strip_y_changed.emit(value / self._zoom)
        self.viewport().update()

    # ------------------------------------------------------------------ painting

    def _image_for(self, tile: Tile) -> QImage | None:
        """The tile's image from the LRU cache (loaded on demand); None when unreadable."""
        if tile.path is None:
            return None
        cached = self._image_cache.get(tile.path)
        if cached is not None:
            self._image_cache.move_to_end(tile.path)
            return cached
        image = _load_image(tile.path)
        if image is None:
            return None  # failures are not cached; paintEvent draws them like a missing tile
        self._image_cache[tile.path] = image
        while len(self._image_cache) > self._max_cached_images:
            self._image_cache.popitem(last=False)
        return image

    def _paint_gap(self, painter: QPainter, rect: QRectF, label: str) -> None:
        """Draw a filtered/missing/failed tile: a hatched rectangle with the label centred."""
        palette = self.palette()
        painter.fillRect(rect, palette.color(QPalette.ColorRole.Base))
        painter.fillRect(rect, QBrush(palette.color(QPalette.ColorRole.Mid), Qt.BrushStyle.BDiagPattern))
        painter.setPen(palette.color(QPalette.ColorRole.WindowText))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the tiles intersecting the viewport, centred when narrower than the viewport."""
        painter = QPainter(self.viewport())
        vp = self.viewport().rect()
        painter.fillRect(vp, self.palette().color(QPalette.ColorRole.Base))
        if not self._tiles or self._strip_width <= 0:
            return
        zoom = self._zoom
        strip_w_px = self._strip_width * zoom
        x_off = self._strip_x_offset()
        value = self.verticalScrollBar().value()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        for tile in self._tiles:
            y_top = tile.y0 * zoom - value
            y_bottom = tile.y1 * zoom - value
            if y_bottom <= 0 or y_top >= vp.height():
                continue
            rect = QRectF(x_off, y_top, strip_w_px, y_bottom - y_top)
            image = self._image_for(tile) if tile.kind == "image" else None
            if image is not None:
                painter.drawImage(rect, image)
            else:
                self._paint_gap(painter, rect, tile.label)
        self._paint_overlays(painter, x_off, value)
        painter.end()

    def _paint_overlays(self, painter: QPainter, x_off: float, value: int) -> None:
        """Outline every overlay box in the highlight colour; fill the selected one lightly."""
        if not self._overlays:
            return
        zoom = self._zoom
        color = self.palette().color(QPalette.ColorRole.Highlight)
        fill = QColor(color)
        fill.setAlpha(60)
        pen = QPen(color, 2.0)
        painter.setPen(pen)
        for region_id, x0, y0, x1, y1 in self._overlays:
            rect = QRectF(x_off + x0 * zoom, y0 * zoom - value, (x1 - x0) * zoom, (y1 - y0) * zoom)
            if rect.bottom() < 0 or rect.top() > self.viewport().height():
                continue
            if region_id == self._selected:
                painter.fillRect(rect, fill)
            painter.drawRect(rect)

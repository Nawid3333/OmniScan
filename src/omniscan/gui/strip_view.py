"""StripView: a vertical strip of tiles (raw pages or output slices) drawn in strip coordinates.

The view scrolls in strip y (`strip_y_changed`) and scales with a single zoom factor
(`zoom_changed`, widget pixels per strip pixel). Two `StripView`s sharing one strip space can be
linked exactly: same strip y, same zoom — this is what `omniscan.gui.compare_view` builds on.

Overlay boxes (text regions) can be outlined over the strip; an *editable* view (the Studio) also lets
the selected box be dragged, resized by its eight handles or nudged with the arrow keys
(`overlay_changed`), and a new box be drawn in draw mode or with Shift held (`box_drawn`).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeyEvent,
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
Handle = Literal["move", "n", "s", "e", "w", "nw", "ne", "sw", "se"]

MIN_ZOOM = 0.05
MAX_ZOOM = 4.0
ZOOM_WHEEL_FACTOR = 1.15
HANDLE_PX = 5  # half the side of a resize handle, in widget pixels
MIN_BOX_PX = 4  # a box can never be dragged or drawn smaller than this (strip pixels)
NUDGE_PX = 1
NUDGE_SHIFT_PX = 10
_HANDLE_CURSORS: dict[Handle, Qt.CursorShape] = {
    "move": Qt.CursorShape.SizeAllCursor,
    "n": Qt.CursorShape.SizeVerCursor,
    "s": Qt.CursorShape.SizeVerCursor,
    "e": Qt.CursorShape.SizeHorCursor,
    "w": Qt.CursorShape.SizeHorCursor,
    "nw": Qt.CursorShape.SizeFDiagCursor,
    "se": Qt.CursorShape.SizeFDiagCursor,
    "ne": Qt.CursorShape.SizeBDiagCursor,
    "sw": Qt.CursorShape.SizeBDiagCursor,
}


@dataclass(slots=True)
class _Drag:
    """A box being moved, resized or drawn: where it started and the box it started from."""

    handle: Handle | Literal["draw"]
    region_id: str | None
    start: tuple[float, float]  # strip point under the press
    box: tuple[int, int, int, int]  # the box before the drag (the press point twice for a draw)
    current: tuple[int, int, int, int]


def _load_image(path: Path) -> QImage | None:
    """Read an image file, or None when it is missing or not a valid image."""
    image = QImage(str(path))
    return None if image.isNull() else image


class StripView(QAbstractScrollArea):
    """Scrollable, zoomable renderer of a chapter strip: one tile per raw page or output slice."""

    strip_y_changed = Signal(float)  # strip y of the viewport's top edge
    zoom_changed = Signal(float)
    overlay_clicked = Signal(str)  # id of the overlay box under a left click
    overlay_changed = Signal(str, int, int, int, int)  # a box moved, resized or nudged by hand (strip space)
    box_drawn = Signal(int, int, int, int)  # a new box drawn by hand (strip space)
    stroke_painted = Signal(list, int)  # a brush stroke: its strip points [(x, y), ...] and the brush radius

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
        self._provided: dict[Path, QImage] = {}  # in-memory tiles (a rendered preview), by their tile path
        self._editable = False
        self._draw_mode = False
        self._drag: _Drag | None = None
        self._brush: int | None = None  # brush radius in strip px while painting is on
        self._stroke: list[tuple[float, float]] | None = None  # the stroke being painted
        self.verticalScrollBar().valueChanged.connect(self._on_value_changed)
        self.viewport().installEventFilter(self)
        self.viewport().setMouseTracking(True)  # the cursor shows what a press would do
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # the arrow keys nudge the selected box

    # ------------------------------------------------------------------ tiles

    def set_tiles(self, tiles: Sequence[Tile], strip_width: int, strip_height: int) -> None:
        """Show a chapter strip; resets the scroll position to 0 and fits the width."""
        self._tiles = tuple(tiles)
        self._strip_width = int(strip_width)
        self._strip_height = int(strip_height)
        self._image_cache.clear()
        self._provided.clear()
        self._sync_guard = True
        try:
            self._update_bars(y=0.0)
        finally:
            self._sync_guard = False
        self.fit_width()
        self.viewport().update()

    def provide_image(self, path: Path, image: QImage) -> None:
        """Show `image` for the tile whose path is `path` without reading a file (a rendered preview);
        cleared by the next `set_tiles`."""
        self._provided[path] = image
        self.viewport().update()

    def tiles(self) -> tuple[Tile, ...]:
        """The tiles currently displayed (empty tuple before the first `set_tiles`)."""
        return self._tiles

    def strip_size(self) -> tuple[int, int]:
        """The strip's width and height in strip pixels."""
        return self._strip_width, self._strip_height

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

    def select(self, region_id: str | None) -> None:
        """Mark one overlay box selected (drawn filled; the one hand edits apply to)."""
        self._selected = region_id
        self.viewport().update()

    def selected(self) -> str | None:
        """The selected overlay box's id."""
        return self._selected

    def set_editable(self, editable: bool) -> None:
        """Let the selected box be moved, resized and nudged, and new boxes be drawn."""
        self._editable = editable
        if not editable:
            self._drag = None
            self._draw_mode = False
            self._brush, self._stroke = None, None
            self.viewport().unsetCursor()
        self.viewport().update()

    def is_editable(self) -> bool:
        """Whether boxes can be edited by hand."""
        return self._editable

    def set_draw_mode(self, drawing: bool) -> None:
        """In draw mode a left drag over the strip draws a new box (`box_drawn`) instead of selecting."""
        self._draw_mode = drawing and self._editable
        if self._draw_mode:
            self._brush, self._stroke = None, None
        self.viewport().setCursor(
            Qt.CursorShape.CrossCursor if self._draw_mode else Qt.CursorShape.ArrowCursor
        )

    def is_drawing(self) -> bool:
        """Whether draw mode is on."""
        return self._draw_mode

    def set_brush(self, radius: int | None) -> None:
        """Paint with a round brush of `radius` strip px: a left drag paints a stroke (`stroke_painted` on release)
        instead of selecting or drawing; None turns painting off (only in an editable view)."""
        self._brush = radius if radius is not None and radius > 0 and self._editable else None
        self._stroke = None
        if self._brush is not None:
            self._draw_mode = False
        self.viewport().setCursor(
            Qt.CursorShape.CrossCursor
            if self._brush is not None or self._draw_mode
            else Qt.CursorShape.ArrowCursor
        )
        self.viewport().update()

    def brush(self) -> int | None:
        """The brush radius while painting is on, else None."""
        return self._brush

    def nudge(self, dx: int, dy: int) -> None:
        """Move the selected box by (dx, dy) strip pixels (`overlay_changed`)."""
        box = self._selected_box()
        if box is None or not self._editable:
            return
        region_id, x0, y0, x1, y1 = box
        moved = self._clamp_box((x0 + dx, y0 + dy, x1 + dx, y1 + dy), keep_size=True)
        self._replace_overlay(region_id, moved)
        self.overlay_changed.emit(region_id, *moved)

    def _selected_box(self) -> Overlay | None:
        """The selected overlay, or None."""
        return next((box for box in self._overlays if box[0] == self._selected), None)

    def _replace_overlay(self, region_id: str, box: tuple[int, int, int, int]) -> None:
        """Redraw one overlay at a new box."""
        self._overlays = tuple((region_id, *box) if item[0] == region_id else item for item in self._overlays)
        self.viewport().update()

    def _clamp_box(
        self, box: tuple[int, int, int, int], *, keep_size: bool = False, min_size: bool = True
    ) -> tuple[int, int, int, int]:
        """Keep a box inside the strip and (unless `min_size` is off) at least MIN_BOX_PX wide and tall; a
        moved box keeps its size."""
        x0, y0, x1, y1 = box
        width, height = self._strip_width, self._strip_height
        if keep_size:
            w, h = min(x1 - x0, width), min(y1 - y0, height)
            x0 = min(max(x0, 0), width - w)
            y0 = min(max(y0, 0), height - h)
            return x0, y0, x0 + w, y0 + h
        x0, x1 = sorted((min(max(x0, 0), width), min(max(x1, 0), width)))
        y0, y1 = sorted((min(max(y0, 0), height), min(max(y1, 0), height)))
        if not min_size:
            return x0, y0, x1, y1
        if x1 - x0 < MIN_BOX_PX:
            x1 = min(x0 + MIN_BOX_PX, width)
            x0 = x1 - MIN_BOX_PX
        if y1 - y0 < MIN_BOX_PX:
            y1 = min(y0 + MIN_BOX_PX, height)
            y0 = y1 - MIN_BOX_PX
        return max(x0, 0), max(y0, 0), x1, y1

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

    def strip_point(self, pos: QPointF) -> tuple[float, float]:
        """The strip point under a viewport position."""
        x = (pos.x() - self._strip_x_offset()) / self._zoom
        y = (pos.y() + self.verticalScrollBar().value()) / self._zoom
        return x, y

    def _widget_rect(self, box: Overlay) -> QRectF:
        """An overlay box in viewport pixels."""
        _region_id, x0, y0, x1, y1 = box
        zoom = self._zoom
        value = self.verticalScrollBar().value()
        return QRectF(
            self._strip_x_offset() + x0 * zoom, y0 * zoom - value, (x1 - x0) * zoom, (y1 - y0) * zoom
        )

    def _handle_points(self, rect: QRectF) -> dict[Handle, QPointF]:
        """The eight resize handles of a box, in viewport pixels."""
        cx, cy = rect.center().x(), rect.center().y()
        return {
            "nw": rect.topLeft(),
            "n": QPointF(cx, rect.top()),
            "ne": rect.topRight(),
            "e": QPointF(rect.right(), cy),
            "se": rect.bottomRight(),
            "s": QPointF(cx, rect.bottom()),
            "sw": rect.bottomLeft(),
            "w": QPointF(rect.left(), cy),
        }

    def handle_at(self, pos: QPointF) -> Handle | None:
        """What a press at viewport position `pos` would do to the selected box: a resize handle, "move"
        inside it, or None (only in an editable view)."""
        box = self._selected_box()
        if box is None or not self._editable:
            return None
        rect = self._widget_rect(box)
        for name, point in self._handle_points(rect).items():
            if abs(point.x() - pos.x()) <= HANDLE_PX and abs(point.y() - pos.y()) <= HANDLE_PX:
                return name
        return "move" if rect.contains(pos) else None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """A left press selects the box under it (`overlay_clicked`); in an editable view it also starts a
        move or resize of the selected box, or draws a new box (draw mode, or Shift held)."""
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        pos = event.position()
        x, y = self.strip_point(pos)
        if self._brush is not None and self._strip_width > 0:
            self._stroke = [(x, y)]
            self.viewport().update()
            event.accept()
            return
        drawing = self._editable and (
            self._draw_mode or bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        )
        if drawing and self._strip_width > 0:
            point = (round(x), round(y))
            self._drag = _Drag("draw", None, (x, y), (*point, *point), (*point, *point))
            event.accept()
            return
        handle = self.handle_at(pos)
        if handle is None and self._overlays:
            hit = self.overlay_at(x, y)
            if hit is not None:
                self._selected = hit
                self.overlay_clicked.emit(hit)
                handle = self.handle_at(pos)
        box = self._selected_box()
        if handle is not None and box is not None:
            self._drag = _Drag(handle, box[0], (x, y), box[1:], box[1:])
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Drag the box being moved, resized or drawn, or paint on; otherwise show what a press would do."""
        if self._stroke is not None:
            self._stroke.append(self.strip_point(event.position()))
            self.viewport().update()
            event.accept()
            return
        drag = self._drag
        if drag is None:
            if self._editable and not self._draw_mode:
                handle = self.handle_at(event.position())
                if handle is None:
                    self.viewport().unsetCursor()
                else:
                    self.viewport().setCursor(_HANDLE_CURSORS[handle])
            super().mouseMoveEvent(event)
            return
        x, y = self.strip_point(event.position())
        dx, dy = round(x - drag.start[0]), round(y - drag.start[1])
        x0, y0, x1, y1 = drag.box
        if drag.handle == "draw":
            drag.current = self._clamp_box((x0, y0, round(x), round(y)), min_size=False)
            self.viewport().update()
            event.accept()
            return
        if drag.handle == "move":
            drag.current = self._clamp_box((x0 + dx, y0 + dy, x1 + dx, y1 + dy), keep_size=True)
        else:
            if "w" in drag.handle:
                x0 += dx
            if "e" in drag.handle:
                x1 += dx
            if "n" in drag.handle:
                y0 += dy
            if "s" in drag.handle:
                y1 += dy
            drag.current = self._clamp_box((x0, y0, x1, y1))
        if drag.region_id is not None:
            self._replace_overlay(drag.region_id, drag.current)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """End a drag: emit the moved/resized box, the drawn one (when it is big enough) or the painted stroke."""
        stroke = self._stroke
        if stroke is not None and event.button() == Qt.MouseButton.LeftButton:
            self._stroke = None
            self.viewport().update()
            if self._brush is not None:
                self.stroke_painted.emit(stroke, self._brush)
            event.accept()
            return
        drag = self._drag
        if drag is None or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        self._drag = None
        self.viewport().update()
        if drag.handle == "draw":
            x0, y0, x1, y1 = drag.current
            if x1 - x0 >= MIN_BOX_PX and y1 - y0 >= MIN_BOX_PX and drag.current != drag.box:
                self.box_drawn.emit(x0, y0, x1, y1)
        elif drag.region_id is not None and drag.current != drag.box:
            self.overlay_changed.emit(drag.region_id, *drag.current)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """The arrow keys nudge the selected box by 1 px (Shift: 10 px) in an editable view."""
        steps = {
            Qt.Key.Key_Left: (-1, 0),
            Qt.Key.Key_Right: (1, 0),
            Qt.Key.Key_Up: (0, -1),
            Qt.Key.Key_Down: (0, 1),
        }
        key = Qt.Key(event.key())
        if self._editable and self._selected is not None and key in steps:
            step = NUDGE_SHIFT_PX if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else NUDGE_PX
            dx, dy = steps[key]
            self.nudge(dx * step, dy * step)
            event.accept()
            return
        super().keyPressEvent(event)

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
        provided = self._provided.get(tile.path)
        if provided is not None:
            return provided
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
        self._paint_stroke(painter, x_off, value)
        painter.end()

    def _paint_stroke(self, painter: QPainter, x_off: float, value: int) -> None:
        """The stroke being painted, as a translucent band as wide as the brush."""
        if not self._stroke or self._brush is None:
            return
        zoom = self._zoom
        color = QColor(self.palette().color(QPalette.ColorRole.Highlight))
        color.setAlpha(110)
        pen = QPen(color, 2 * self._brush * zoom, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        points = [QPointF(x_off + x * zoom, y * zoom - value) for x, y in self._stroke]
        if len(points) == 1:
            painter.drawPoint(points[0])
        else:
            painter.drawPolyline(points)

    def _paint_overlays(self, painter: QPainter, x_off: float, value: int) -> None:
        """Outline every overlay box in the highlight colour; fill the selected one lightly and, in an
        editable view, draw its handles; a box being drawn shows as a dashed rectangle."""
        drag = self._drag
        if not self._overlays and (drag is None or drag.handle != "draw"):
            return
        zoom = self._zoom
        color = self.palette().color(QPalette.ColorRole.Highlight)
        fill = QColor(color)
        fill.setAlpha(60)
        pen = QPen(color, 2.0)
        painter.setPen(pen)
        selected_rect: QRectF | None = None
        for region_id, x0, y0, x1, y1 in self._overlays:
            rect = QRectF(x_off + x0 * zoom, y0 * zoom - value, (x1 - x0) * zoom, (y1 - y0) * zoom)
            if rect.bottom() < 0 or rect.top() > self.viewport().height():
                continue
            if region_id == self._selected:
                painter.fillRect(rect, fill)
                selected_rect = rect
            painter.drawRect(rect)
        if selected_rect is not None and self._editable:
            painter.setBrush(QBrush(self.palette().color(QPalette.ColorRole.Base)))
            for point in self._handle_points(selected_rect).values():
                painter.drawRect(
                    QRectF(point.x() - HANDLE_PX, point.y() - HANDLE_PX, 2 * HANDLE_PX, 2 * HANDLE_PX)
                )
            painter.setBrush(Qt.BrushStyle.NoBrush)
        if drag is not None and drag.handle == "draw":
            x0, y0, x1, y1 = drag.current
            painter.setPen(QPen(color, 1.5, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(x_off + x0 * zoom, y0 * zoom - value, (x1 - x0) * zoom, (y1 - y0) * zoom))

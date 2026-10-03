"""StripView tests (offscreen): painting in strip space, zoom, scrolling, signals, LRU, wheel."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PIL import Image
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from omniscan.gui import strip_view as strip_module
from omniscan.gui.services.library import Tile
from omniscan.gui.strip_view import StripView

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
STRIP_WIDTH = 60


def _png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> Path:
    """Write a solid-colour PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _standard_tiles(tmp_path: Path) -> tuple[Tile, ...]:
    """Red 0–100, green 100–200, blue 200–300 (strip 60 px wide) plus a gap tile 300–400."""
    return (
        Tile(0, 100, _png(tmp_path / "red.png", (60, 100), RED), "red.png", "image"),
        Tile(100, 200, _png(tmp_path / "green.png", (60, 100), GREEN), "green.png", "image"),
        Tile(200, 300, _png(tmp_path / "blue.png", (60, 100), BLUE), "blue.png", "image"),
        Tile(300, 400, None, "filtered slice 3", "filtered"),
    )


def _pixel(image, x: int, y: int) -> tuple[int, int, int]:
    """A grabbed image's pixel as an RGB tuple."""
    color = image.pixelColor(x, y)
    return (color.red(), color.green(), color.blue())


def _converge(qapp: QApplication, view: StripView, width: int, height: int) -> None:
    """Resize the widget until the *viewport* is exactly width x height (frame/scrollbars included)."""
    for _ in range(8):
        qapp.processEvents()
        dw = width - view.viewport().width()
        dh = height - view.viewport().height()
        if dw == 0 and dh == 0:
            return
        view.resize(max(1, view.width() + dw), max(1, view.height() + dh))
    qapp.processEvents()
    assert (view.viewport().width(), view.viewport().height()) == (width, height)


def _prepare(
    qapp: QApplication,
    tmp_path: Path,
    *,
    tiles: Sequence[Tile] | None = None,
    zoom: float = 1.0,
    vp: tuple[int, int] = (60, 100),
    max_cached: int = 32,
) -> StripView:
    """A shown StripView with given tiles, a pinned zoom and an exact viewport size."""
    view = StripView(max_cached_images=max_cached)
    view.set_tiles(_standard_tiles(tmp_path) if tiles is None else tuple(tiles), STRIP_WIDTH, 400)
    view.resize(300, 250)
    view.show()
    qapp.processEvents()
    view.set_zoom(zoom)  # pin the zoom; this also ends fit-width mode
    _converge(qapp, view, *vp)
    return view


def _wheel(ctrl: bool, dy: int = 120) -> QWheelEvent:
    """A wheel event with the given angle delta (Ctrl for zoom)."""
    modifiers = Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier
    return QWheelEvent(
        QPointF(50.0, 50.0),
        QPointF(50.0, 50.0),
        QPoint(0, 0),
        QPoint(0, dy),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


# ---------------------------------------------------------------------- painting / scrolling (test 5)


def test_viewport_shows_the_tile_at_strip_y(qapp: QApplication, tmp_path: Path) -> None:
    view = _prepare(qapp, tmp_path, zoom=1.0, vp=(60, 100))

    image = view.viewport().grab().toImage()
    assert _pixel(image, 30, 50) == RED  # middle of the viewport: strip row 50

    view.set_strip_y(100)
    assert view.strip_y() == 100.0
    image = view.viewport().grab().toImage()
    assert _pixel(image, 30, 50) == GREEN  # strip row 150

    view.set_strip_y(250)  # viewport shows strip rows 250-350
    assert view.strip_y() == 250.0
    image = view.viewport().grab().toImage()
    assert _pixel(image, 30, 25) == BLUE  # strip row 275
    assert _pixel(image, 5, 75) != BLUE  # strip row 325: the hatched gap, not blue


def test_set_strip_y_clamps(qapp: QApplication, tmp_path: Path) -> None:
    view = _prepare(qapp, tmp_path, zoom=1.0, vp=(60, 100))
    view.set_strip_y(-5)
    assert view.strip_y() == 0.0
    view.set_strip_y(10_000)
    assert view.strip_y() == view.max_strip_y()
    assert view.max_strip_y() == 300.0


# ---------------------------------------------------------------------- zoom (tests 6)


def test_set_zoom_doubles_drawn_size_and_keeps_strip_y(qapp: QApplication, tmp_path: Path) -> None:
    view = _prepare(qapp, tmp_path, zoom=1.0, vp=(130, 260))
    image = view.viewport().grab().toImage()
    assert _pixel(image, 65, 98) == RED  # red/green boundary at widget y=100
    assert _pixel(image, 65, 102) == GREEN

    view.set_zoom(2.0)
    assert view.zoom() == 2.0
    assert view.strip_y() == 0.0
    image = view.viewport().grab().toImage()
    assert _pixel(image, 65, 196) == RED  # same boundary now at widget y=200
    assert _pixel(image, 65, 204) == GREEN

    view.set_strip_y(30)
    view.set_zoom(3.0)
    assert view.strip_y() == 30.0  # strip y of the viewport top is kept


def test_set_zoom_clamps(qapp: QApplication, tmp_path: Path) -> None:
    view = _prepare(qapp, tmp_path, zoom=1.0, vp=(130, 100))
    view.set_zoom(0.001)
    assert view.zoom() == 0.05
    view.set_zoom(100.0)
    assert view.zoom() == 4.0


def test_fit_width_matches_viewport_and_refits_on_resize(qapp: QApplication, tmp_path: Path) -> None:
    view = _prepare(qapp, tmp_path, zoom=1.0, vp=(120, 100))
    view.fit_width()
    assert view.zoom() == 2.0

    view.resize(view.width() + 60, view.height())  # fit-width is still active
    qapp.processEvents()
    assert view.zoom() == pytest.approx(view.viewport().width() / STRIP_WIDTH)

    view.set_zoom(1.0)  # a manual zoom ends fit-width mode
    view.resize(view.width() + 60, view.height())
    qapp.processEvents()
    assert view.zoom() == 1.0


# ---------------------------------------------------------------------- signals (test 7)


def test_bar_move_emits_strip_y_once(qapp: QApplication, tmp_path: Path) -> None:
    view = _prepare(qapp, tmp_path, zoom=1.0, vp=(130, 100))
    received: list[float] = []
    view.strip_y_changed.connect(received.append)

    view.verticalScrollBar().setValue(50)
    assert received == [50.0]

    view.set_strip_y(80, emit=False)
    assert received == [50.0]
    assert view.strip_y() == 80.0

    view.set_zoom(2.0, emit=False)
    received.clear()
    view.verticalScrollBar().setValue(100)
    assert received == [100.0 / 2.0]  # strip y = value / zoom


def test_set_zoom_emits_zoom_changed_once(qapp: QApplication, tmp_path: Path) -> None:
    view = _prepare(qapp, tmp_path, zoom=1.0, vp=(130, 100))
    zooms: list[float] = []
    view.zoom_changed.connect(zooms.append)

    view.set_zoom(2.0)
    assert zooms == [2.0]

    view.set_zoom(3.0, emit=False)
    assert zooms == [2.0]
    assert view.zoom() == 3.0


# ---------------------------------------------------------------------- tile_at (test 8)


def test_tile_at(qapp: QApplication, tmp_path: Path) -> None:
    view = _prepare(qapp, tmp_path, vp=(60, 100))
    tiles = view.tiles()
    assert view.tile_at(0) is tiles[0]  # y0 inclusive
    assert view.tile_at(99) is tiles[0]
    assert view.tile_at(100) is tiles[1]  # y1 exclusive
    assert view.tile_at(200) is tiles[2]
    assert view.tile_at(350) is tiles[3]  # inside the gap tile: the gap tile itself
    assert view.tile_at(400) is None  # below the strip
    assert view.tile_at(-1) is None


# ---------------------------------------------------------------------- failures / LRU / empty (test 9)


def test_invalid_image_draws_hatched_without_raising(qapp: QApplication, tmp_path: Path) -> None:
    missing = tmp_path / "missing_file.png"
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"this is not an image")
    tiles = (
        Tile(0, 100, missing, "missing_file.png", "image"),
        Tile(100, 200, corrupt, "corrupt.png", "image"),
    )
    view = _prepare(qapp, tmp_path, tiles=tiles, zoom=1.0, vp=(60, 200))

    image = view.viewport().grab().toImage()
    assert _pixel(image, 30, 50) != RED  # not the tile image: hatched like a missing tile
    assert _pixel(image, 30, 150) != RED


def test_image_cache_never_exceeds_max(qapp: QApplication, tmp_path: Path, monkeypatch) -> None:
    calls: list[Path] = []
    real_load = strip_module._load_image

    def counting_load(path: Path):
        calls.append(path)
        return real_load(path)

    monkeypatch.setattr(strip_module, "_load_image", counting_load)

    tiles = tuple(
        Tile(y, y + 100, _png(tmp_path / f"t{y}.png", (60, 100), RED), f"t{y}.png", "image")
        for y in range(0, 400, 100)
    )
    view = _prepare(qapp, tmp_path, tiles=tiles, zoom=0.5, vp=(130, 300), max_cached=2)

    view.viewport().grab()
    view.viewport().grab()
    assert len(view._image_cache) == 2  # the LRU keeps only the last two painted tiles
    assert set(view._image_cache) == {tiles[2].path, tiles[3].path}
    assert all(calls)  # the loader was really used, not bypassed


def test_set_tiles_empty_is_blank(qapp: QApplication) -> None:
    view = StripView()
    view.set_tiles([], 0, 0)
    view.resize(200, 150)
    view.show()
    qapp.processEvents()

    assert view.tiles() == ()
    assert view.max_strip_y() == 0.0
    assert view.tile_at(0) is None
    image = view.viewport().grab().toImage()  # paints without errors
    assert not image.isNull()
    assert _pixel(image, 10, 10) == _pixel(image, 190, 130)  # uniform background


# ---------------------------------------------------------------------- wheel (test 10)


def test_ctrl_wheel_zooms_by_factor_and_ends_fit_mode(qapp: QApplication, tmp_path: Path) -> None:
    view = StripView()
    view.set_tiles(_standard_tiles(tmp_path), STRIP_WIDTH, 400)
    view.resize(190, 150)
    view.show()
    qapp.processEvents()
    fit_zoom = view.zoom()

    QApplication.sendEvent(view.viewport(), _wheel(ctrl=True))
    assert view.zoom() == pytest.approx(fit_zoom * 1.15)

    view.resize(view.width() + 40, view.height())  # manual zoom ended fit-width mode
    qapp.processEvents()
    assert view.zoom() == pytest.approx(fit_zoom * 1.15)


def test_plain_wheel_scrolls(qapp: QApplication, tmp_path: Path) -> None:
    view = _prepare(qapp, tmp_path, zoom=1.0, vp=(130, 100))
    received: list[float] = []
    view.strip_y_changed.connect(received.append)

    before = view.verticalScrollBar().value()
    QApplication.sendEvent(view.viewport(), _wheel(ctrl=False, dy=-120))
    assert view.verticalScrollBar().value() > before
    assert view.strip_y() > 0.0
    assert received == [view.strip_y()]


# ---------------------------------------------------------------------- box editing (the Studio)


def _mouse(kind: QEvent.Type, x: float, y: float, *, shift: bool = False) -> QMouseEvent:
    """A left-button mouse event at viewport position (x, y)."""
    modifiers = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    return QMouseEvent(
        kind, QPointF(x, y), QPointF(x, y), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, modifiers
    )


def _drag(
    view: StripView, start: tuple[float, float], end: tuple[float, float], *, shift: bool = False
) -> None:
    """Press at `start`, move to `end`, release."""
    view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, *start, shift=shift))
    view.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, *end, shift=shift))
    view.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, *end, shift=shift))


def _editable(qapp: QApplication, tmp_path: Path) -> tuple[StripView, list[tuple]]:
    """An editable view at zoom 1 with one selected box (10,10)-(30,30), recording overlay_changed."""
    view = _prepare(qapp, tmp_path, zoom=1.0, vp=(60, 100))
    view.set_overlays([("a", 10, 10, 30, 30), ("b", 40, 60, 55, 80)], "a")
    view.set_editable(True)
    changes: list[tuple] = []
    view.overlay_changed.connect(lambda *args: changes.append(args))
    return view, changes


def test_dragging_inside_the_selected_box_moves_it(qapp: QApplication, tmp_path: Path) -> None:
    view, changes = _editable(qapp, tmp_path)
    assert view.handle_at(QPointF(20, 20)) == "move"
    _drag(view, (20, 20), (25, 28))
    assert changes == [("a", 15, 18, 35, 38)]
    assert view.overlays()[0] == ("a", 15, 18, 35, 38)


def test_dragging_a_handle_resizes_the_box(qapp: QApplication, tmp_path: Path) -> None:
    view, changes = _editable(qapp, tmp_path)
    assert view.handle_at(QPointF(30, 30)) == "se" and view.handle_at(QPointF(10, 20)) == "w"
    _drag(view, (30, 30), (34, 36))
    assert changes == [("a", 10, 10, 34, 36)]
    _drag(view, (10, 23), (4, 23))  # the west edge; a drag that would invert is clamped to the minimum size
    assert changes[-1] == ("a", 4, 10, 34, 36)


def test_clicking_another_box_selects_it_and_a_drag_then_moves_that_one(
    qapp: QApplication, tmp_path: Path
) -> None:
    view, changes = _editable(qapp, tmp_path)
    clicked: list[str] = []
    view.overlay_clicked.connect(clicked.append)
    _drag(view, (47, 70), (49, 72))  # the middle of b: a move, not its west handle
    assert clicked == ["b"] and view.selected() == "b"
    assert changes == [("b", 42, 62, 57, 82)]


def test_a_shift_drag_or_draw_mode_draws_a_new_box(qapp: QApplication, tmp_path: Path) -> None:
    view, changes = _editable(qapp, tmp_path)
    drawn: list[tuple] = []
    view.box_drawn.connect(lambda *args: drawn.append(args))
    _drag(view, (40, 20), (55, 45), shift=True)
    assert drawn == [(40, 20, 55, 45)] and changes == []
    view.set_draw_mode(True)
    assert view.is_drawing()
    _drag(view, (50, 10), (42, 2))  # drawn upwards: the corners are sorted
    assert drawn[-1] == (42, 2, 50, 10)
    _drag(view, (5, 5), (6, 6))  # too small: nothing
    assert len(drawn) == 2
    view.set_editable(False)
    assert not view.is_drawing() and view.handle_at(QPointF(20, 20)) is None


def test_arrow_keys_nudge_the_selected_box(qapp: QApplication, tmp_path: Path) -> None:
    from PySide6.QtTest import QTest

    view, changes = _editable(qapp, tmp_path)
    view.setFocus()
    QTest.keyClick(view, Qt.Key.Key_Right)
    QTest.keyClick(view, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
    assert changes == [("a", 11, 10, 31, 30), ("a", 11, 20, 31, 40)]
    view.nudge(100, 0)  # clamped to the strip, keeping the size
    assert changes[-1] == ("a", 40, 20, 60, 40)


def test_a_provided_image_is_drawn_without_a_file(qapp: QApplication, tmp_path: Path) -> None:
    tiles = (Tile(0, 100, Path("preview") / "0", "page 1: not rendered yet", "image"),)
    view = _prepare(qapp, tmp_path, tiles=tiles, zoom=1.0, vp=(60, 100))
    assert _pixel(view.viewport().grab().toImage(), 30, 50) != GREEN  # a hatched gap
    image = QImage(60, 100, QImage.Format.Format_RGB888)
    image.fill(QColor(*GREEN))
    view.provide_image(Path("preview") / "0", image)
    assert _pixel(view.viewport().grab().toImage(), 30, 50) == GREEN
    view.set_tiles(tiles, STRIP_WIDTH, 100)  # new tiles drop the provided images
    assert _pixel(view.viewport().grab().toImage(), 30, 50) != GREEN


def test_the_brush_paints_a_stroke_instead_of_selecting(qapp: QApplication, tmp_path: Path) -> None:
    view, changes = _editable(qapp, tmp_path)
    strokes: list[tuple[list[tuple[float, float]], int]] = []
    view.stroke_painted.connect(lambda points, radius: strokes.append((points, radius)))
    view.set_brush(6)
    assert view.brush() == 6 and not view.is_drawing()
    _drag(view, (20, 20), (25, 28))  # inside the selected box: painted over, not moved
    assert changes == [] and view.overlays()[0] == ("a", 10, 10, 30, 30)
    assert strokes == [([view.strip_point(QPointF(20, 20)), view.strip_point(QPointF(25, 28))], 6)]
    view.set_draw_mode(True)  # one tool at a time
    assert view.brush() is None and view.is_drawing()
    view.set_brush(4)
    view.set_editable(False)
    assert view.brush() is None


def test_cut_mode_adds_moves_and_removes_output_cuts(qapp: QApplication, tmp_path: Path) -> None:
    view, changes = _editable(qapp, tmp_path)
    added: list[int] = []
    moved: list[tuple[int, int]] = []
    removed: list[int] = []
    view.cut_added.connect(added.append)
    view.cut_moved.connect(lambda old, new: moved.append((old, new)))
    view.cut_removed.connect(removed.append)
    view.set_brush(5)
    view.set_cut_mode(True)  # one tool at a time
    assert view.is_cutting() and view.brush() is None
    view.set_cut_lines([50, 20], crossing=[20])
    assert view.cut_lines() == (20, 50)

    _drag(view, (20, 35), (20, 35))  # a click away from every cut, inside box "a": a cut, not a selection
    assert added == [35] and changes == []
    assert view.cut_at(QPointF(10, 52)) == 50 and view.cut_at(QPointF(10, 35)) is None
    _drag(view, (10, 51), (10, 70))  # grab the cut at 50, drop it at 70
    assert moved == [(50, 70)]
    view.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(10, 21),
            QPointF(10, 21),
            Qt.MouseButton.RightButton,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert removed == [20]
    view.set_draw_mode(True)
    assert not view.is_cutting()
    view.set_cut_mode(True)
    view.set_editable(False)
    assert not view.is_cutting()

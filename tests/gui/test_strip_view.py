"""StripView tests (offscreen): painting in strip space, zoom, scrolling, signals, LRU, wheel."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from omniscan.gui import strip_view as strip_module  # noqa: E402
from omniscan.gui.services.library import Tile  # noqa: E402
from omniscan.gui.strip_view import StripView  # noqa: E402

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
    modifiers = (
        Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier
    )
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
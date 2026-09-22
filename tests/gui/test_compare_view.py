"""CompareView tests (offscreen): chapter loading, linked/independent scrolling, fit, signals."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PIL import Image
from PySide6.QtWidgets import QApplication, QWidget

from omniscan.gui.compare_view import CompareView
from omniscan.gui.services.library import ChapterView, Tile

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
STRIP_WIDTH = 60


def _png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> Path:
    """Write a solid-colour PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _chapter_view(tmp_path: Path) -> ChapterView:
    """Raw red/green/blue 0–300, output 4 tiles with a filtered gap 200–300; strip 60x400."""
    raw = (
        Tile(0, 100, _png(tmp_path / "r0.png", (60, 100), RED), "r0.png", "image"),
        Tile(100, 200, _png(tmp_path / "r1.png", (60, 100), GREEN), "r1.png", "image"),
        Tile(200, 300, _png(tmp_path / "r2.png", (60, 100), BLUE), "r2.png", "image"),
    )
    output = (
        Tile(0, 100, _png(tmp_path / "o0.png", (60, 100), YELLOW), "o0.png", "image"),
        Tile(100, 200, _png(tmp_path / "o1.png", (60, 100), CYAN), "o1.png", "image"),
        Tile(200, 300, None, "filtered slice 2", "filtered"),
        Tile(300, 400, _png(tmp_path / "o3.png", (60, 100), MAGENTA), "o3.png", "image"),
    )
    return ChapterView("S", "Chapter 1", STRIP_WIDTH, 400, raw, output, True)


def _shown(qapp: QApplication, compare: CompareView) -> CompareView:
    """Resize, show and settle a compare view sized so fit-width is not zoom-clamped."""
    compare.resize(240, 400)
    compare.show()
    qapp.processEvents()
    return compare


def _pixel(image, x: int, y: int) -> tuple[int, int, int]:
    """A grabbed image's pixel as an RGB tuple."""
    color = image.pixelColor(x, y)
    return (color.red(), color.green(), color.blue())


def _pane(compare: CompareView, index: int) -> QWidget:
    """One splitter pane (asserted present)."""
    widget = compare.splitter.widget(index)
    assert widget is not None
    return widget


def _centre_pixel(compare: CompareView, side: str, y: int) -> tuple[int, int, int]:
    """The pixel at the centre of one side's viewport at viewport y."""
    view = compare.left if side == "left" else compare.right
    image = view.viewport().grab().toImage()
    return _pixel(image, view.viewport().width() // 2, y)


# ---------------------------------------------------------------------- set_chapter (test 11)


def test_set_chapter_shows_both_sides_at_zero_and_same_zoom(qapp: QApplication, tmp_path: Path) -> None:
    compare = _shown(qapp, CompareView())
    view = _chapter_view(tmp_path)
    compare.set_chapter(view)
    qapp.processEvents()

    assert compare.left.tiles() == view.raw
    assert compare.right.tiles() == view.output
    assert compare.left.strip_y() == 0.0
    assert compare.right.strip_y() == 0.0
    assert compare.left.zoom() == compare.right.zoom()
    assert compare.chapter_label.text() == "S — Chapter 1"
    assert compare.right_caption.text() == "Output"

    assert _centre_pixel(compare, "left", 20) == RED  # strip row ~20: raw page 1
    assert _centre_pixel(compare, "right", 20) == YELLOW  # the same strip row, output slice 0


def test_set_chapter_without_output(qapp: QApplication, tmp_path: Path) -> None:
    compare = _shown(qapp, CompareView())
    with_output = _chapter_view(tmp_path)
    compare.set_chapter(with_output)
    empty = ChapterView("S", "Chapter 1", STRIP_WIDTH, 300, with_output.raw, (), False)
    compare.set_chapter(empty)

    assert compare.right.tiles() == ()
    assert compare.right_caption.text() == "Output (not translated yet)"
    assert compare.left.tiles() == with_output.raw  # the raw side is unaffected


# ---------------------------------------------------------------------- linked (test 12)


def test_linked_follows_strip_y_and_zoom(qapp: QApplication, tmp_path: Path) -> None:
    compare = _shown(qapp, CompareView())
    compare.set_chapter(_chapter_view(tmp_path))

    compare.left.set_strip_y(80)
    assert compare.right.strip_y() == 80.0
    compare.right.set_strip_y(120)  # scrolling the right view moves the left one
    assert compare.left.strip_y() == 120.0
    assert compare.strip_y() == 120.0  # of the master (the right view)

    compare.left.set_zoom(2.0)
    assert compare.right.zoom() == 2.0

    # both viewports show the same strip rows: the same widget y is the same strip row
    compare.left.set_strip_y(40)  # strip row 40+25/2 = 52.5: raw page 1 / output slice 0
    assert _centre_pixel(compare, "left", 25) == RED
    assert _centre_pixel(compare, "right", 25) == YELLOW
    compare.right.set_strip_y(150)  # strip row 150+25/2 = 162.5: raw page 2 / output slice 1
    assert _centre_pixel(compare, "left", 25) == GREEN
    assert _centre_pixel(compare, "right", 25) == CYAN


# ---------------------------------------------------------------------- independent (test 13)


def test_independent_and_realign_on_switch_back(qapp: QApplication, tmp_path: Path) -> None:
    compare = _shown(qapp, CompareView())
    compare.set_chapter(_chapter_view(tmp_path))
    modes: list[str] = []
    compare.sync_mode_changed.connect(modes.append)

    compare.sync_checkbox.setChecked(False)
    assert compare.sync_mode() == "independent"
    assert not compare.sync_checkbox.isChecked()
    assert modes == ["independent"]

    compare.left.set_strip_y(80)
    assert compare.right.strip_y() == 0.0  # the other side does not follow
    compare.right.set_strip_y(120)
    assert compare.left.strip_y() == 80.0

    compare.set_sync_mode("linked")  # master = right (moved last): left is re-aligned to it
    assert modes == ["independent", "linked"]
    assert compare.left.strip_y() == 120.0
    assert compare.right.strip_y() == 120.0
    assert compare.sync_checkbox.isChecked()
    assert compare.sync_mode() == "linked"

    compare.set_sync_mode("independent")  # checkbox and set_sync_mode stay consistent
    assert not compare.sync_checkbox.isChecked()
    compare.sync_checkbox.setChecked(True)
    assert compare.sync_mode() == "linked"
    assert modes == ["independent", "linked", "independent", "linked"]


# ---------------------------------------------------------------------- fit + no loop (test 14)


def test_fit_button_fits_both_without_signal_loop(qapp: QApplication, tmp_path: Path) -> None:
    compare = _shown(qapp, CompareView())
    compare.set_chapter(_chapter_view(tmp_path))
    left_ys: list[float] = []
    right_ys: list[float] = []
    compare.left.strip_y_changed.connect(left_ys.append)
    compare.right.strip_y_changed.connect(right_ys.append)

    compare.fit_button.click()
    # linked mode keeps one shared scale: the fit chain ends with the right side's fit on both
    assert compare.right.zoom() == pytest.approx(compare.right.viewport().width() / STRIP_WIDTH)
    assert compare.left.zoom() == compare.right.zoom()

    compare.left.set_strip_y(120)
    assert left_ys == [120.0]  # the master emits exactly once
    assert right_ys == []  # the follower moves silently


# ---------------------------------------------------------------------- zoom_by / sides (U3c)


def test_zoom_by_multiplies_the_master_zoom(qapp: QApplication, tmp_path: Path) -> None:
    compare = _shown(qapp, CompareView())
    compare.set_chapter(_chapter_view(tmp_path))
    before = compare.left.zoom()

    compare.zoom_by(1.25)
    assert compare.left.zoom() == pytest.approx(before * 1.25)
    assert compare.right.zoom() == compare.left.zoom()  # linked: the other side follows

    compare.set_sync_mode("independent")
    compare.right.set_zoom(1.0)  # the right view is now the master
    compare.zoom_by(0.8)
    assert compare.right.zoom() == pytest.approx(0.8)
    assert compare.left.zoom() != compare.right.zoom()  # independent: only the master zoomed


def test_set_visible_sides_collapses_one_pane(qapp: QApplication, tmp_path: Path) -> None:
    compare = _shown(qapp, CompareView())
    compare.set_chapter(_chapter_view(tmp_path))

    compare.set_visible_sides("raw")
    assert _pane(compare, 0).isVisible()
    assert not _pane(compare, 1).isVisible()

    compare.set_visible_sides("output")
    assert not _pane(compare, 0).isVisible()
    assert _pane(compare, 1).isVisible()

    compare.set_visible_sides("both")
    assert _pane(compare, 0).isVisible()
    assert _pane(compare, 1).isVisible()

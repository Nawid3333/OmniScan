"""The desktop Studio's output cuts (offscreen): shown on the strip, added, moved, removed, refused and undone."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PIL import Image
from PySide6.QtWidgets import QApplication

from omniscan.core.config import Config
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import Band, BBox, Region, RegionsArtifact
from omniscan.edits import store
from omniscan.edits.session import StudioSession
from omniscan.gui.services.studio import load_cuts, snap_to_band
from omniscan.gui.studio_view import StudioView
from tests.fixtures.gui_library import CHAPTERS, SERIES, _artifacts, build_library


def _chapter(cfg: Config) -> ChapterPaths:
    """Episode 03: three 100-row pages, one slice each (the slicer's cuts at 100 and 200), one region at rows
    10-40 of the first page."""
    paths = SeriesPaths.from_config(cfg, SERIES).chapter(CHAPTERS[2])
    _artifacts(paths, {"ingest", "slice"})
    for index in range(3):
        Image.new("RGB", (40, 100), (90, 120, 200)).save(paths.raw_dir / f"{index + 1:04d}.jpg", "JPEG")
    RegionsArtifact(
        regions=[
            Region(
                id="r0001",
                slice_index=0,
                kind="bubble_text",
                bbox=BBox(x0=5, y0=10, x1=35, y1=40),
                text="안녕",
            )
        ]
    ).save(paths.artifact("ocr.json"))
    return paths


def test_a_cut_snaps_to_the_middle_of_the_nearest_calm_band() -> None:
    bands = [(100, 110), (180, 200)]
    assert snap_to_band(90, bands) == 105  # 10 rows above the first band
    assert snap_to_band(150, bands) == 190  # 30 rows above the second (41 below the first)
    assert snap_to_band(150, bands, max_distance=20) == 150  # nothing that close
    assert snap_to_band(105, []) == 105


def test_the_slicers_cuts_hold_until_cuts_are_set_by_hand(tmp_path: Path) -> None:
    paths = _chapter(build_library(tmp_path / "lib"))
    state = load_cuts(paths)
    assert (state.hand, state.cuts, state.strip_height, state.crossings) == (None, [100, 200], 300, [])

    session = StudioSession(paths, direction="ltr")
    assert session.set_cuts([30, 150]) == [30, 150]
    state = load_cuts(paths)
    assert state.cuts == [30, 150] and state.crossings == [(30, "r0001")]  # through the region's text
    assert store.history_steps(paths) == (1, 0)  # one undo step
    with pytest.raises(ValueError, match=r"more than 120"):
        session.set_cuts([150], max_height=120)
    assert load_cuts(paths).cuts == [30, 150]  # a refused change keeps what was there


def test_output_cuts_are_drawn_and_edited_on_the_strip(qapp: QApplication, tmp_path: Path) -> None:
    cfg = build_library(tmp_path / "lib")
    paths = _chapter(cfg)
    view = StudioView(cfg)
    view.resize(1200, 600)
    view.show()
    assert view.open_chapter(SERIES, CHAPTERS[2])
    assert view.cuts_button.isEnabled() and view.strip.cut_lines() == ()

    view.draw_button.setChecked(True)
    view.cuts_button.setChecked(True)  # one tool at a time
    assert not view.draw_button.isChecked() and view.strip.is_cutting()
    assert view.strip.cut_lines() == (100, 200) and view.cuts_label.text() == "3 images (one per slice)"
    assert not view.reset_cuts_button.isEnabled()  # nothing set by hand yet

    view.snap_check.setChecked(False)
    view.strip.cut_added.emit(150)
    assert load_cuts(paths).hand == [100, 150, 200] and view.strip.cut_lines() == (100, 150, 200)
    assert view.status_label.text() == "output cuts saved: 4 images" and view.cuts_label.text() == "4 images"
    view.strip.cut_moved.emit(100, 25)  # into the region's text
    assert load_cuts(paths).hand == [25, 150, 200]
    assert view.cuts_label.text() == "4 images · cuts through r0001"
    view.strip.cut_removed.emit(25)
    assert view.strip.cut_lines() == (150, 200) and view.reset_cuts_button.isEnabled()

    assert not view.change_cuts([500]) and view.status_label.text().startswith(
        "cuts not changed: cut 500 lies"
    )
    assert view.undo() and view.strip.cut_lines() == (25, 150, 200)  # one step per change
    view.reset_cuts_button.click()
    assert load_cuts(paths).hand is None and view.cuts_label.text() == "3 images (one per slice)"

    view.brush_button.setChecked(True)
    assert not view.cuts_button.isChecked() and not view.strip.is_cutting() and view.cuts_label.text() == ""


def test_a_new_cut_snaps_into_a_calm_band_unless_snapping_is_off(qapp: QApplication, tmp_path: Path) -> None:
    cfg = build_library(tmp_path / "lib")
    paths = _chapter(cfg)
    slices = store.load_slices(paths)
    slices.bands = [Band(y0=40, y1=60, color=(0, 0, 0))]
    slices.save(paths.artifact("slices.json"))
    view = StudioView(cfg)
    assert view.open_chapter(SERIES, CHAPTERS[2])
    view.set_cut_mode(True)
    view.strip.cut_added.emit(70)
    assert load_cuts(paths).hand == [50, 100, 200]  # 70 is 11 rows below the band 40-60: its middle
    view.snap_check.setChecked(False)
    view.strip.cut_added.emit(70)
    assert load_cuts(paths).hand == [50, 70, 100, 200]

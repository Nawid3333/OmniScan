"""The desktop Studio's cleanup brush (offscreen): the stroke as a mask, on its page, cleaned and taken back."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PIL import Image
from PySide6.QtWidgets import QApplication

from omniscan.cleanup.store import load_cleanup
from omniscan.core.config import Config
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import BBox, IngestArtifact, Region, RegionsArtifact, SourceFile
from omniscan.gui.services.studio import stroke_mask, stroke_on_page
from omniscan.gui.studio_view import StudioView
from tests.fixtures.gui_library import CHAPTERS, SERIES, _artifacts, build_library


def test_a_stroke_becomes_a_round_brush_mask_cut_to_the_strip() -> None:
    shaped = stroke_mask([(10.0, 10.0), (20.0, 10.0)], 3, 40, 300)
    assert shaped is not None
    box, mask = shaped
    assert box == BBox(x0=7, y0=7, x1=24, y1=14) and mask.shape == (7, 17)
    assert mask[10 - 7, 15 - 7] and mask[10 - 7, 10 - 7]  # along the line and at its ends
    assert not mask[0, 0] and not mask[0, -1]  # round ends, not a rectangle
    edge = stroke_mask([(1.0, 1.0)], 5, 40, 300)
    assert edge is not None and edge[0] == BBox(x0=0, y0=0, x1=7, y1=7)  # a dab cut to the strip
    assert stroke_mask([(100.0, 10.0)], 3, 40, 300) is None  # beside the strip
    assert stroke_mask([], 3, 40, 300) is None


def test_a_strip_stroke_lands_on_the_page_under_its_middle_in_page_pixels() -> None:
    ingest = IngestArtifact(
        series="S",
        chapter="C",
        strip_width=40,
        strip_height=150,
        files=[
            SourceFile(index=0, name="1.jpg", sha256="0" * 64, width=80, height=200, y0=0, y1=100, scale=0.5),
            SourceFile(index=1, name="2.jpg", sha256="1" * 64, width=40, height=50, y0=100, y1=150),
        ],
    )
    mask = np.ones((10, 6), dtype=bool)
    on_first = stroke_on_page(ingest, BBox(x0=4, y0=20, x1=10, y1=30), mask)
    assert (on_first.page, on_first.box) == (0, BBox(x0=8, y0=40, x1=20, y1=60))  # twice as big on the page
    assert on_first.mask.shape == (20, 12) and on_first.mask.all()
    across = stroke_on_page(ingest, BBox(x0=4, y0=96, x1=10, y1=106), mask)  # middle at row 101: page 1
    assert (across.page, across.box) == (1, BBox(x0=4, y0=0, x1=10, y1=6))  # the part on page 0 is left out
    with pytest.raises(ValueError, match="not on a page"):
        stroke_on_page(ingest, BBox(x0=4, y0=150, x1=10, y1=160), mask)


def _cleanable(cfg: Config) -> ChapterPaths:
    """Episode 03 with its pages where a real library keeps them, page geometry and one region."""
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


def _wait(qapp: QApplication, done: Callable[[], bool], seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not done() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert done()


def test_a_painted_stroke_is_cleaned_shown_and_taken_back(qapp: QApplication, tmp_path: Path) -> None:
    cfg = build_library(tmp_path / "lib")
    paths = _cleanable(cfg)
    view = StudioView(cfg)
    view.resize(1200, 600)
    view.show()
    assert view.open_chapter(SERIES, CHAPTERS[2])
    assert view.brush_button.isEnabled() and not view.take_back_button.isEnabled()
    view.draw_button.setChecked(True)
    view.brush_button.setChecked(True)  # one tool at a time; the preview opens to show the result
    assert not view.draw_button.isChecked() and view.strip.brush() == 12 and view.preview.isVisible()
    view.clean_combo.setCurrentIndex(0)  # fill with the colour around
    assert view.clean_stroke([(10.0, 150.0), (30.0, 150.0)], 4)  # on page 1 (rows 100-200)
    _wait(qapp, lambda: not view.is_busy())
    cleanup = load_cleanup(paths)
    assert cleanup is not None and [(p.id, p.method) for p in cleanup.patches] == [("c0001", "fill")]
    assert cleanup.patches[0].box.y0 >= 100 and view.status_label.text().startswith("c0001: cleaned (fill, ")
    assert 1 in view._rendered and view.take_back_button.isEnabled()

    assert view.take_back_stroke() and view.status_label.text() == "c0001: taken back"
    cleanup = load_cleanup(paths)
    assert cleanup is not None and cleanup.patches == []
    assert not view.take_back_stroke() and not view.take_back_button.isEnabled()
    assert (
        not view.clean_stroke([(100.0, 150.0)], 4)
        and view.status_label.text() == "the stroke is not on the strip"
    )

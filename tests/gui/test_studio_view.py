"""StudioView tests (offscreen): the table, editing, overlays, QA filter, save and re-letter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from omniscan.core.config import Config
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import BBox, FinalArtifact, FinalLine, Region, RegionsArtifact
from omniscan.edits.store import load_edits
from omniscan.gui.services.runs import RunOutcome, RunSpec
from omniscan.gui.studio_view import RELETTER_STAGES, StudioView
from tests.fixtures.gui_library import CHAPTERS, SERIES, build_library

CHAPTER = CHAPTERS[0]


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """The fixture library with OCR and judge output for the first chapter."""
    cfg = build_library(tmp_path / "lib")
    paths = SeriesPaths.from_config(cfg, SERIES).chapter(CHAPTER)
    regions = [
        Region(
            id="r0001", slice_index=0, kind="bubble_text", bbox=BBox(x0=5, y0=10, x1=35, y1=40), text="안녕"
        ),
        Region(
            id="r0002", slice_index=2, kind="bubble_text", bbox=BBox(x0=5, y0=220, x1=35, y1=260), text="뭐"
        ),
    ]
    RegionsArtifact(regions=regions).save(paths.artifact("ocr.json"))
    FinalArtifact(
        judge_model="fake", lines=[FinalLine(region_id="r0001", text="Hello", decision="pick")]
    ).save(paths.artifact("final.json"))
    return cfg


class _FakeController:
    """Stands in for RunController: records the spec and finishes at once."""

    specs: ClassVar[list[RunSpec]] = []

    def __init__(self, cfg: Config, spec: RunSpec) -> None:
        self.specs.append(spec)

    def run(self, **_: Any) -> RunOutcome:
        return RunOutcome(ok=True, aborted=None, failed=(), chapters_run=1)


def _view(qapp: QApplication, cfg: Config, **kwargs: Any) -> StudioView:
    view = StudioView(cfg, **kwargs)
    view.resize(900, 600)
    view.show()
    assert view.open_chapter(SERIES, CHAPTER)
    qapp.processEvents()
    return view


def test_opens_a_chapter_with_rows_overlays_and_issues(qapp: QApplication, cfg: Config) -> None:
    view = _view(qapp, cfg)
    assert view.table.rowCount() == 2
    assert [view.table.item(0, c).text() for c in range(4)] == ["0", "bubble_text", "안녕", "Hello"]  # type: ignore[union-attr]
    assert [box[0] for box in view.strip.overlays()] == ["r0001", "r0002"]
    assert view.table.item(1, 4).text() == "no English line"  # type: ignore[union-attr]
    view.issues_only.setChecked(True)
    assert view.table.isRowHidden(0) and not view.table.isRowHidden(1)


def test_editing_a_cell_and_saving_writes_the_edits(qapp: QApplication, cfg: Config) -> None:
    view = _view(qapp, cfg)
    assert not view.save_button.isEnabled()
    view.table.item(1, 3).setText("What?")  # type: ignore[union-attr]
    assert view.save_button.isEnabled()
    assert view.save() == 1
    session = view.session()
    assert session is not None
    written = {edit.region_id: edit.text for edit in load_edits(session.paths).translations}
    assert written == {"r0002": "What?"}
    assert view.run_check() == 0


def test_clicking_a_box_selects_its_row_and_remove_drops_it(qapp: QApplication, cfg: Config) -> None:
    view = _view(qapp, cfg)
    view.select_region("r0002")
    assert view.table.currentRow() == 1
    assert view.strip.overlay_at(10, 230) == "r0002"
    view.remove_selected()
    assert view.table.rowCount() == 1
    assert [box[0] for box in view.strip.overlays()] == ["r0001"]


def test_reletter_runs_typeset_and_export_for_the_chapter(qapp: QApplication, cfg: Config) -> None:
    _FakeController.specs = []
    view = _view(qapp, cfg, controller_factory=_FakeController)
    view.reletter()
    worker = view._worker
    assert worker is not None
    worker.wait(5000)
    for _ in range(20):
        qapp.processEvents()
    assert _FakeController.specs == [
        RunSpec(series=SERIES, mode="subset", chapters=(CHAPTER,), stages=RELETTER_STAGES)
    ]
    assert not view.is_running()
    assert view.status_label.text() == "re-lettered: output updated"


def test_chapter_without_regions_says_what_to_run(qapp: QApplication, cfg: Config) -> None:
    view = StudioView(cfg)
    view.open_chapter(SERIES, CHAPTERS[2])
    assert view.session() is None and view.table.rowCount() == 0
    assert "run detection and OCR first" in view.status_label.text()

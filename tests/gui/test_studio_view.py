"""StudioView tests (offscreen): the table, editing, overlays, QA filter, save and re-letter."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest

pytest.importorskip("PySide6")

import numpy as np
from PySide6.QtWidgets import QApplication, QTableWidgetSelectionRange

from omniscan.core.config import Config
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import BBox, FinalArtifact, FinalLine, Region, RegionsArtifact
from omniscan.edits.store import load_edits
from omniscan.gui.services.runs import RunOutcome, RunSpec
from omniscan.gui.services.studio import PagePreview, Suggested
from omniscan.gui.studio_view import COLUMNS, RELETTER_STAGES, StudioView
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
    assert [view.table.item(0, c).text() for c in range(5)] == ["0", "bubble_text", "", "안녕", "Hello"]  # type: ignore[union-attr]
    assert [box[0] for box in view.strip.overlays()] == ["r0001", "r0002"]
    assert view.table.item(1, 6).text() == "no English line"  # type: ignore[union-attr]
    view.issues_only.setChecked(True)
    assert view.table.isRowHidden(0) and not view.table.isRowHidden(1)


def test_editing_a_cell_and_saving_writes_the_edits(qapp: QApplication, cfg: Config) -> None:
    view = _view(qapp, cfg)
    assert not view.save_button.isEnabled()
    view.table.item(1, 4).setText("What?")  # type: ignore[union-attr]
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


def test_not_a_typo_accepts_the_selected_lines_words_for_the_series(qapp: QApplication, cfg: Config) -> None:
    view = _view(qapp, cfg)
    view.table.item(1, 4).setText("Grab teh sword")  # type: ignore[union-attr]
    assert view.run_check() == 1
    assert "'teh' is not in the dictionary" in view.table.item(1, 6).text()  # type: ignore[union-attr]
    view.select_region("r0001")
    assert not view.not_typo_button.isEnabled()  # "Hello" has no unknown word
    view.select_region("r0002")
    assert view.not_typo_button.isEnabled()
    assert view.allow_selected_words() == ["teh"]
    assert view.run_check() == 0
    words = SeriesPaths.from_config(cfg, SERIES).library_dir / "typo_words.txt"
    assert words.read_text(encoding="utf-8") == "teh\n"


def _wait(qapp: QApplication, done: Callable[[], bool], *, seconds: float = 5.0) -> None:
    """Pump the event loop until `done()` (a worker task's result landed) or the time is up."""
    deadline = time.monotonic() + seconds
    while not done() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert done()


def _select(view: StudioView, *ids: str) -> None:
    """Select the rows of `ids` together."""
    view.table.clearSelection()
    for row, item in enumerate(view._rows):
        if item.region_id in ids:
            view.table.setRangeSelected(QTableWidgetSelectionRange(row, 0, row, len(COLUMNS) - 1), True)


def test_multi_selection_applies_to_every_selected_row(qapp: QApplication, cfg: Config) -> None:
    view = _view(qapp, cfg)
    _select(view, "r0001", "r0002")
    assert view.selected_ids() == ["r0001", "r0002"]
    assert view.mark_selected(True) == 2
    assert [view.table.item(r, 5).text() for r in range(2)] == ["checked", "checked"]  # type: ignore[union-attr]
    assert view.set_kind("sfx") == 2
    assert [view.table.item(r, 1).text() for r in range(2)] == ["sfx", "sfx"]  # type: ignore[union-attr]
    assert view.progress_label.text() == "0 todo · 0 edited · 2 checked"
    assert view.save() == 4
    session = view.session()
    assert session is not None
    edits = load_edits(session.paths)
    assert sorted(edit.region_id for edit in edits.checked) == ["r0001", "r0002"]
    assert sorted((edit.region_id, edit.kind) for edit in edits.regions) == [
        ("r0001", "sfx"),
        ("r0002", "sfx"),
    ]
    _select(view, "r0001", "r0002")
    assert view.remove_selected() == 2 and view.table.rowCount() == 0


def test_lettering_styles_apply_to_the_selection_and_show_in_the_status(
    qapp: QApplication, cfg: Config
) -> None:
    view = _view(qapp, cfg)
    _select(view, "r0001", "r0002")
    assert view.apply_lettering({"size_px": 30, "color": [255, 0, 0]}) == 2
    assert view.table.item(0, 5).text() == "todo · lettered"  # type: ignore[union-attr]
    assert view.apply_lettering({"size_px": 24}) == 2  # a second dialog keeps the colour
    assert view.save() == 2
    session = view.session()
    assert session is not None
    assert session.layout_of("r0002") == {"size_px": 24, "color": (255, 0, 0), "hidden": False}
    assert {edit.region_id: edit.size_px for edit in load_edits(session.paths).layout} == {
        "r0001": 24,
        "r0002": 24,
    }
    _select(view, "r0001")
    assert view.apply_lettering(None) == 1 and view.save() == 1
    assert [edit.region_id for edit in load_edits(session.paths).layout] == ["r0002"]


def test_undo_and_redo_walk_the_saved_steps(qapp: QApplication, cfg: Config) -> None:
    view = _view(qapp, cfg)
    assert not view.undo_button.isEnabled()
    view.table.item(1, 4).setText("What?")  # type: ignore[union-attr]
    view.save()
    assert view.undo_button.isEnabled() and not view.redo_button.isEnabled()
    assert view.undo() and view.table.item(1, 4).text() == ""  # type: ignore[union-attr]
    assert view.redo_button.isEnabled()
    assert view.redo() and view.table.item(1, 4).text() == "What?"  # type: ignore[union-attr]
    assert not view.redo() and view.status_label.text() == "nothing to redo"


def test_page_navigation_steps_over_pages_with_regions(qapp: QApplication, cfg: Config) -> None:
    view = _view(qapp, cfg)
    assert (view.page_spin.minimum(), view.page_spin.maximum(), view.page_spin.value()) == (0, 2, 0)
    assert view.step_page(1) and view.page_spin.value() == 2  # page 1 has no region
    assert not view.step_page(1)
    view.page_only.setChecked(True)
    assert view.table.isRowHidden(0) and not view.table.isRowHidden(1)
    assert view.step_page(-1) and view.page_spin.value() == 0
    assert not view.table.isRowHidden(0) and view.table.isRowHidden(1)
    assert not view.prev_chapter_button.isEnabled() and view.next_chapter_button.isEnabled()


def test_a_box_moved_on_the_strip_is_saved_as_a_region_edit(qapp: QApplication, cfg: Config) -> None:
    view = _view(qapp, cfg)
    assert view.strip.is_editable()
    view.strip.overlay_changed.emit("r0001", 6, 12, 36, 42)
    session = view.session()
    assert session is not None
    assert session.regions()[0].bbox == BBox(x0=6, y0=12, x1=36, y1=42)
    assert view.table.item(0, 5).text() == "edited"  # type: ignore[union-attr]
    assert view.save() == 1
    assert load_edits(session.paths).regions[0].bbox == BBox(x0=6, y0=12, x1=36, y1=42)


def test_a_drawn_box_becomes_a_region_of_the_picked_kind(qapp: QApplication, cfg: Config) -> None:
    view = _view(qapp, cfg)
    view.kind_combo.setCurrentText("free_text")
    view.draw_button.setChecked(True)
    assert view.strip.is_drawing()
    assert view.add_box(2, 110, 30, 140) == "m0001"
    assert not view.draw_button.isChecked()
    assert view.selected_ids() == ["m0001"]
    assert [box[0] for box in view.strip.overlays()] == ["r0001", "r0002", "m0001"]
    session = view.session()
    assert session is not None
    edit = load_edits(session.paths).regions[0]
    assert (edit.region_id, edit.added, edit.kind) == ("m0001", True, "free_text")


def test_translate_puts_the_models_lines_in_the_table_to_keep_on_save(
    qapp: QApplication, cfg: Config
) -> None:
    asked: list[tuple[list[str], str | None]] = []

    def fake_translate(cfg: Config, paths: object, ids: list[str], profile: str | None) -> list[Suggested]:
        asked.append((ids, profile))
        return [Suggested(region_id, "What?", "fake-profile") for region_id in ids]

    view = _view(qapp, cfg, translate_fn=fake_translate)
    assert not view.translate_selected()  # nothing selected
    _select(view, "r0002")
    assert view.translate_selected() and view.is_busy() and not view.translate_button.isEnabled()
    _wait(qapp, lambda: not view.is_busy())
    assert asked == [(["r0002"], None)]
    assert view.table.item(1, 4).text() == "What?"  # type: ignore[union-attr]
    assert "fake-profile" in view.status_label.text() and view.save_button.isEnabled()
    assert view.save() == 1


def test_read_again_puts_the_reading_in_the_source_column(qapp: QApplication, cfg: Config) -> None:
    view = _view(qapp, cfg, read_fn=lambda cfg, paths, region_id: "안녕하세요")
    _select(view, "r0001")
    assert view.read_selected()
    _wait(qapp, lambda: not view.is_busy())
    assert view.table.item(0, 3).text() == "안녕하세요"  # type: ignore[union-attr]
    assert view.save() == 1


def test_a_failed_model_call_is_reported(qapp: QApplication, cfg: Config) -> None:
    def boom(cfg: Config, paths: object, region_id: str) -> str:
        raise RuntimeError("no daemon")

    view = _view(qapp, cfg, read_fn=boom)
    _select(view, "r0001")
    assert view.read_selected()
    _wait(qapp, lambda: not view.is_busy())
    assert view.status_label.text() == "failed: RuntimeError: no daemon"
    assert view.read_button.isEnabled()


def test_the_speaker_column_takes_a_name_and_offers_the_series_characters(
    qapp: QApplication, cfg: Config
) -> None:
    series = SeriesPaths.from_config(cfg, SERIES)
    (series.library_dir / "voices.toml").write_text(
        '[[character]]\nname = "Jinwoo"\naliases = ["진우"]\n\n[[character]]\nname = "Hae-in"\n',
        encoding="utf-8",
    )
    view = _view(qapp, cfg)
    assert COLUMNS[2] == "Speaker" and view._speaker_delegate.names == ["Jinwoo", "Hae-in"]
    view.table.item(0, 2).setText(" Jinwoo ")  # type: ignore[union-attr]
    assert view.table.item(0, 5).text() == "edited" and view.save_button.isEnabled()  # type: ignore[union-attr]
    assert view.save() == 1
    session = view.session()
    assert session is not None
    assert [(edit.region_id, edit.speaker) for edit in load_edits(session.paths).regions] == [
        ("r0001", "Jinwoo")
    ]
    assert view.table.item(0, 2).text() == "Jinwoo"  # type: ignore[union-attr]
    assert session.regions()[0].speaker == "Jinwoo"


def test_a_broken_voices_file_is_reported_and_the_speaker_stays_free_text(
    qapp: QApplication, cfg: Config
) -> None:
    (SeriesPaths.from_config(cfg, SERIES).library_dir / "voices.toml").write_text("[[character]]\n", "utf-8")
    view = _view(qapp, cfg)
    assert view._speaker_delegate.names == []
    assert "voices.toml" in view.status_label.text() and "has no name" in view.status_label.text()


def test_find_missed_text_adds_the_ticked_areas_as_regions(qapp: QApplication, cfg: Config) -> None:
    from omniscan.detect.on_demand import Found

    searched: list[tuple[int, float | None]] = []
    found = [
        Found("free_text", BBox(x0=2, y0=110, x1=30, y1=140), None, 0.41, "쾅", 0.9),
        Found("bubble_text", BBox(x0=4, y0=150, x1=36, y1=190), None, 0.38, "어?", 0.8),
    ]

    def fake_find(cfg: Config, paths: object, page: int, threshold: float | None) -> list[Found]:
        searched.append((page, threshold))
        return found

    offered: list[int] = []

    def pick_second(areas: list[Found], page: int, parent: object) -> list[Found]:
        offered.append(len(areas))
        return areas[1:]

    view = _view(qapp, cfg, find_fn=fake_find, pick_fn=pick_second)
    assert view.find_button.isEnabled() and view.find_threshold.text() == "series setting"
    view.strip.set_zoom(4.0)
    half = view.strip.viewport().height() / view.strip.zoom() / 2
    view.strip.set_strip_y(150.0 - half)  # the middle of the view on strip row 150: raw page 1 (rows 100-200)
    view.find_threshold.setValue(0.3)
    assert view.find_missed()
    _wait(qapp, lambda: not view.is_busy())
    assert searched == [(1, 0.3)] and offered == [2]
    assert [box[0] for box in view.strip.overlays()] == ["r0001", "r0002", "m0001"]
    assert view.selected_ids() == ["m0001"]
    session = view.session()
    assert session is not None
    added = load_edits(session.paths).regions[0]
    assert (added.region_id, added.added, added.kind, added.text) == ("m0001", True, "bubble_text", "어?")
    assert view.status_label.text() == "page 2: added 1 of 2 found region(s) — Translate them next"

    view = _view(qapp, cfg, find_fn=lambda *_args: [], pick_fn=pick_second)
    assert view.find_missed()
    _wait(qapp, lambda: not view.is_busy())
    assert view.status_label.text().endswith("no missed text found") and offered == [2]


def test_preview_renders_the_current_pages_on_demand(qapp: QApplication, cfg: Config) -> None:
    rendered: list[int] = []

    def fake_render(cfg: Config, paths: object, page: int) -> PagePreview:
        rendered.append(page)
        return PagePreview(
            page=page, y0=page * 100, y1=(page + 1) * 100, pixels=np.full((100, 40, 3), 200, np.uint8)
        )

    view = _view(qapp, cfg, preview_fn=fake_render)
    assert not view.preview.isVisible()
    view.set_preview_visible(True)
    _wait(qapp, lambda: not view.is_busy())
    assert rendered == [0] and view.preview.isVisible()  # the regions of page 0 lie on raw page 0 only
    assert [tile.label for tile in view.preview.tiles()][:2] == [
        "page 1: not rendered yet",
        "page 2: not rendered yet",
    ]
    assert view.preview._provided and view.preview.strip_y() == view.strip.strip_y()
    view.page_spin.setValue(2)
    _wait(qapp, lambda: not view.is_busy())
    assert rendered == [0, 2]
    view.table.item(1, 4).setText("What?")  # type: ignore[union-attr]
    view.save()  # a save makes the rendered pages stale: the current one renders again
    _wait(qapp, lambda: not view.is_busy())
    assert rendered == [0, 2, 2]

"""ImportView tests (offscreen): plan preview, grouping edits, conversion notice, commit worker."""
# pyright: reportOptionalMemberAccess=false, reportAttributeAccessIssue=false

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication, QTreeWidgetItem

from omniscan.acquire.drm import NOTICE
from omniscan.gui.import_view import ImportView
from omniscan.importer.execute import ImportResult
from omniscan.importer.plan import ImportPlan, ImportPlanError, ImportPlanItem

WAIT_S = 5.0

# ---------------------------------------------------------------- fixtures


def item(chapter: str, names: list[str]) -> ImportPlanItem:
    """One chapter holding fake source files (only names and suffixes matter to the view)."""
    return ImportPlanItem(chapter=chapter, files=[Path(f"C:/src/{chapter}/{name}") for name in names])


def plan(
    chapters: list[tuple[str, list[str]]],
    series: str = "My Series",
    *,
    warnings: list[str] | None = None,
    archive: Path | None = None,
) -> ImportPlan:
    """A hand-made plan for the fake service."""
    return ImportPlan(
        series=series,
        items=[item(chapter, names) for chapter, names in chapters],
        warnings=warnings or [],
        archive=archive,
    )


MIXED = [("Chapter 1", ["p1.jpg", "p2.png"]), ("Chapter 10", ["p1.jpg", "p2.webp", "p3.jpg"])]


class FakeService:
    """One canned plan; records plan/execute calls; execute can block on an Event like the real worker."""

    def __init__(self, canned: ImportPlan) -> None:
        self._canned = canned
        self._hardware_line = "machine: fake GPU · 16 GB · rocm — conversion runs on the CPU (libjpeg-turbo)"
        self.planned: list[Path] = []
        self.plan_series: list[str | None] = []
        self.plan_error: Exception | None = None
        self.executed: list[ImportPlan] = []
        self.execute_moves: list[bool] = []
        self.execute_threads: list[int] = []
        self.execute_progress: list[tuple[int, int]] = []
        self.execute_release = threading.Event()
        self.execute_result = ImportResult(
            chapters_written=["Chapter 1", "Chapter 10"],
            files_copied=4,
            files_skipped_duplicate=0,
            files_converted=1,
            converted=["p2.png"],
        )
        self.execute_error: Exception | None = None

    def hardware_line(self) -> str:
        return self._hardware_line

    def plan(self, source: Path, *, series: str | None = None) -> ImportPlan:
        self.planned.append(Path(source))
        self.plan_series.append(series)
        if self.plan_error is not None:
            raise self.plan_error
        return self._canned

    def execute(self, plan: ImportPlan, *, move: bool = False, on_progress: Any = None) -> ImportResult:
        self.executed.append(plan)
        self.execute_moves.append(move)
        self.execute_threads.append(threading.get_ident())
        for done, total in self.execute_progress:
            on_progress(done, total)
        self.execute_release.wait(WAIT_S)
        if self.execute_error is not None:
            raise self.execute_error
        return self.execute_result


def view_of(qapp: QApplication, service: FakeService) -> ImportView:
    view = ImportView(service)
    view.resize(1200, 800)
    view.show()
    qapp.processEvents()
    return view


def pump(qapp: QApplication, predicate: Any, timeout_s: float = WAIT_S) -> None:
    """Process events until `predicate()` holds (worker signals arrive as queued events)."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("timed out waiting for the view to settle")
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()


def planned_view(qapp: QApplication, service: FakeService, source: Path = Path("C:/raws")) -> ImportView:
    """A view whose canned plan has loaded (set_source driven through the worker)."""
    view = view_of(qapp, service)
    view.set_source(source)
    pump(qapp, lambda: view.tree.topLevelItemCount() > 0)
    return view


def chapter_texts(view: ImportView) -> list[str]:
    return [view.tree.topLevelItem(i).text(0) for i in range(view.tree.topLevelItemCount())]


def page_texts(view: ImportView, chapter: int) -> list[str]:
    node = view.tree.topLevelItem(chapter)
    return [node.child(i).text(0) for i in range(node.childCount())]


def node(view: ImportView, chapter: int, page: int | None = None) -> QTreeWidgetItem:
    """The tree node for a chapter (or one of its pages); present whenever the plan is loaded."""
    chapter_item = view.tree.topLevelItem(chapter)
    assert chapter_item is not None
    if page is None:
        return chapter_item
    page_item = chapter_item.child(page)
    assert page_item is not None
    return page_item


# ---------------------------------------------------------------- notice and context


def test_drm_notice_is_verbatim(qapp: QApplication) -> None:
    view = view_of(qapp, FakeService(plan(MIXED)))

    assert view.notice_label.text() == NOTICE


def test_context_line_comes_from_the_service(qapp: QApplication) -> None:
    service = FakeService(plan(MIXED))

    view = view_of(qapp, service)

    assert view.context_label.text() == service._hardware_line


# ---------------------------------------------------------------- plan preview


def test_plan_loads_the_chapter_file_breakdown(qapp: QApplication) -> None:
    service = FakeService(plan(MIXED))

    view = planned_view(qapp, service)

    assert chapter_texts(view) == ["Chapter 1", "Chapter 10"]
    assert view.tree.topLevelItem(0).text(1) == "2 page(s)"
    assert view.tree.topLevelItem(1).text(1) == "3 page(s)"
    assert page_texts(view, 0) == ["p1.jpg", "p2.png"]
    assert page_texts(view, 1) == ["p1.jpg", "p2.webp", "p3.jpg"]
    assert view.tree.topLevelItem(0).child(0).text(1) == "JPG"
    assert view.tree.topLevelItem(0).child(1).text(1) == "→ JPEG"
    assert view.tree.topLevelItem(1).child(1).text(1) == "→ JPEG"  # .webp
    # convertible pages carry the explanation where the user is looking
    assert "converted to JPEG (quality 95)" in view.tree.topLevelItem(0).child(1).toolTip(1)
    assert view.status_label.text() == "Planned My Series: 2 chapter(s), 5 page(s)"
    # the first page is selected and the chapter edit mirrors its chapter
    assert view.tree.currentItem().text(0) == "p1.jpg"
    assert view.chapter_edit.text() == "Chapter 1"
    assert service.plan_series == [None]  # the series field was empty at plan time


def test_conversion_notice_only_when_non_jpeg_present(qapp: QApplication) -> None:
    view = planned_view(qapp, FakeService(plan(MIXED)))
    assert not view.conversion_label.isHidden()
    assert view.conversion_label.text() == (
        "2 of 5 file(s) will be converted to JPEG for consistent, fast processing."
    )
    assert not view.conversions_button.isHidden()

    all_jpeg = planned_view(qapp, FakeService(plan([("Chapter 1", ["p1.jpg", "p2.jpeg"])])))
    assert all_jpeg.conversion_label.isHidden()
    assert all_jpeg.conversions_button.isHidden()
    assert all_jpeg.conversion_label.text() == ""


def test_which_files_lists_the_conversions_and_warnings(qapp: QApplication) -> None:
    service = FakeService(
        plan(MIXED, warnings=["skipped non-image file: cover.txt"]),
    )
    view = planned_view(qapp, service)

    text = view.details.toPlainText()
    assert text.startswith("Series: My Series\nskipped non-image file: cover.txt")
    assert "Will be converted" not in text

    view.conversions_button.setChecked(True)
    text = view.details.toPlainText()
    assert "Will be converted to JPEG:" in text
    assert "  Chapter 1/p2.png → p2.jpg" in text
    assert "  Chapter 10/p2.webp → p2.jpg" in text
    assert "p1.jpg" not in text.split("Will be converted to JPEG:")[1]

    view.conversions_button.setChecked(False)
    assert "Will be converted" not in view.details.toPlainText()


def test_archive_plan_disables_the_move_toggle(qapp: QApplication) -> None:
    archive_view = planned_view(qapp, FakeService(plan(MIXED, archive=Path("C:/raws.zip"))))
    assert not archive_view.move_toggle.isEnabled()
    assert not archive_view.move_toggle.isChecked()

    folder_view = planned_view(qapp, FakeService(plan(MIXED)))
    assert folder_view.move_toggle.isEnabled()


# ---------------------------------------------------------------- grouping edits


def test_move_page_to_another_chapter_updates_the_commit_plan(qapp: QApplication) -> None:
    service = FakeService(plan(MIXED))
    view = planned_view(qapp, service)
    view.tree.setCurrentItem(node(view, 0, 1))  # p2.png
    view.target_combo.setCurrentIndex(1)  # Chapter 10

    view.move_button.click()

    assert [(g.chapter, [p.name for p in g.files]) for g in view._plan.items] == [
        ("Chapter 1", ["p1.jpg"]),
        ("Chapter 10", ["p1.jpg", "p2.webp", "p3.jpg", "p2.png"]),
    ]
    assert page_texts(view, 1) == ["p1.jpg", "p2.webp", "p3.jpg", "p2.png"]  # the tree shows the edit
    assert view.tree.currentItem().text(0) == "p2.png"  # selection followed the move

    view.import_button.click()
    pump(qapp, lambda: len(service.executed) == 1)
    assert [(g.chapter, [p.name for p in g.files]) for g in service.executed[0].items] == [
        ("Chapter 1", ["p1.jpg"]),
        ("Chapter 10", ["p1.jpg", "p2.webp", "p3.jpg", "p2.png"]),
    ]
    service.execute_release.set()
    pump(qapp, lambda: view.status_label.text().startswith("Imported:"))


def test_reorder_pages_within_a_chapter(qapp: QApplication) -> None:
    service = FakeService(plan([("Chapter 1", ["a.jpg", "b.jpg", "c.jpg"])]))
    view = planned_view(qapp, service)
    view.tree.setCurrentItem(node(view, 0, 2))  # c.jpg

    view.down_button.click()  # already last: no-op, plan untouched
    assert [p.name for p in view._plan.items[0].files] == ["a.jpg", "b.jpg", "c.jpg"]

    view.tree.setCurrentItem(node(view, 0, 1))  # b.jpg
    view.up_button.click()
    assert [p.name for p in view._plan.items[0].files] == ["b.jpg", "a.jpg", "c.jpg"]
    assert page_texts(view, 0) == ["b.jpg", "a.jpg", "c.jpg"]
    assert view.tree.currentItem().text(0) == "b.jpg"  # selection followed the page

    view.down_button.click()
    assert [p.name for p in view._plan.items[0].files] == ["a.jpg", "b.jpg", "c.jpg"]
    assert service.executed == []  # nothing was committed by the edits alone


def test_rename_chapter_updates_the_commit_plan(qapp: QApplication) -> None:
    service = FakeService(plan(MIXED))
    view = planned_view(qapp, service)
    view.tree.setCurrentItem(node(view, 0, 0))  # a page of Chapter 1

    view.chapter_edit.setText("Chapter 1a")

    assert chapter_texts(view) == ["Chapter 1a", "Chapter 10"]
    assert view._plan.items[0].chapter == "Chapter 1a"

    view.import_button.click()
    pump(qapp, lambda: len(service.executed) == 1)
    assert service.executed[0].items[0].chapter == "Chapter 1a"
    service.execute_release.set()
    pump(qapp, lambda: view.status_label.text().startswith("Imported:"))


def test_merge_appends_the_selected_chapter_into_the_target(qapp: QApplication) -> None:
    service = FakeService(plan([("Chapter 1", ["a.jpg"]), ("Chapter 2", ["b.jpg", "c.jpg"])]))
    view = planned_view(qapp, service)
    view.tree.setCurrentItem(node(view, 1))  # the Chapter 2 chapter row
    view.target_combo.setCurrentIndex(0)  # Chapter 1

    view.move_button.click()

    assert [(g.chapter, [p.name for p in g.files]) for g in view._plan.items] == [
        ("Chapter 1", ["a.jpg", "b.jpg", "c.jpg"]),
    ]
    assert view.tree.currentItem().text(0) == "Chapter 1"  # the merged-into chapter stays selected


def test_split_starts_a_new_chapter_at_the_selected_page(qapp: QApplication) -> None:
    service = FakeService(plan([("Chapter 1", ["a.jpg", "b.jpg", "c.jpg"]), ("Chapter 5", ["d.jpg"])]))
    view = planned_view(qapp, service)
    view.tree.setCurrentItem(node(view, 0, 1))  # b.jpg

    view.split_button.click()

    assert [(g.chapter, [p.name for p in g.files]) for g in view._plan.items] == [
        ("Chapter 1", ["a.jpg"]),
        ("Chapter 6", ["b.jpg", "c.jpg"]),  # named after the plan's top chapter number
        ("Chapter 5", ["d.jpg"]),
    ]
    assert view.tree.currentItem().text(0) == "b.jpg"  # the split's first page stays selected


def test_split_at_the_first_page_is_refused(qapp: QApplication) -> None:
    service = FakeService(plan([("Chapter 1", ["a.jpg", "b.jpg"])]))
    view = planned_view(qapp, service)
    view.tree.setCurrentItem(node(view, 0, 0))

    view.split_button.click()

    assert "cannot split" in view.status_label.text()
    assert "red" in view.status_label.styleSheet()
    assert [p.name for p in view._plan.items[0].files] == ["a.jpg", "b.jpg"]  # plan untouched


def test_series_edit_updates_the_commit_plan(qapp: QApplication) -> None:
    service = FakeService(plan(MIXED))
    view = planned_view(qapp, service)

    view.series_edit.setText("Renamed Series")
    assert view._plan.series == "Renamed Series"

    view.import_button.click()
    pump(qapp, lambda: len(service.executed) == 1)
    assert service.executed[0].series == "Renamed Series"
    service.execute_release.set()
    pump(qapp, lambda: view.status_label.text().startswith("Imported:"))


def test_duplicate_chapter_names_block_the_import(qapp: QApplication) -> None:
    service = FakeService(plan(MIXED))
    view = planned_view(qapp, service)
    view.tree.setCurrentItem(node(view, 0, 0))
    view.chapter_edit.setText("Chapter 10")  # collides with the second chapter

    view.import_button.click()
    qapp.processEvents()

    assert service.executed == []
    assert "Two chapters would write to the same folder: Chapter 10" in view.status_label.text()
    assert "red" in view.status_label.styleSheet()


# ---------------------------------------------------------------- commit


def test_import_runs_on_a_worker_with_progress_and_summary(qapp: QApplication) -> None:
    service = FakeService(plan(MIXED))
    service.execute_progress = [(1, 5), (2, 5)]
    view = planned_view(qapp, service)
    main_thread = threading.get_ident()

    view.import_button.click()
    pump(qapp, lambda: len(service.executed) == 1)

    assert service.execute_threads[0] != main_thread  # the service ran on a worker thread
    assert service.execute_moves == [False]
    pump(qapp, lambda: view.progress.maximum() == 5)
    assert view.progress.value() == 2 and not view.progress.isHidden()
    assert view.status_label.text() == "Importing … 2/5"
    assert not view.import_button.isEnabled()  # no second task while one runs

    service.execute_release.set()
    pump(qapp, lambda: view.status_label.text().startswith("Imported:"))
    assert view.status_label.text() == "Imported: 2 chapter(s), 4 copied, 1 converted, 0 duplicate(s) skipped"
    assert view.progress.isHidden()
    assert view.import_button.isEnabled()


def test_move_toggle_reaches_the_worker(qapp: QApplication) -> None:
    service = FakeService(plan(MIXED))
    view = planned_view(qapp, service)

    view.move_toggle.setChecked(True)
    view.import_button.click()
    pump(qapp, lambda: len(service.executed) == 1)

    assert service.execute_moves == [True]
    service.execute_release.set()
    pump(qapp, lambda: view.status_label.text().startswith("Imported:"))


def test_import_failure_surfaces_red_instead_of_crashing(qapp: QApplication) -> None:
    service = FakeService(plan(MIXED))
    service.execute_error = ImportPlanError("destination exists with different content: p1.jpg")
    view = planned_view(qapp, service)

    view.import_button.click()
    service.execute_release.set()
    pump(qapp, lambda: "ImportPlanError" in view.status_label.text())

    assert view.status_label.text() == "ImportPlanError: destination exists with different content: p1.jpg"
    assert "red" in view.status_label.styleSheet()
    assert view.import_button.isEnabled()  # the view survives and stays usable


def test_plan_failure_surfaces_red_and_clears_the_preview(qapp: QApplication) -> None:
    service = FakeService(plan(MIXED))
    service.plan_error = ImportPlanError("can't read archive C:/raws.zip: bad magic")
    view = view_of(qapp, service)

    view.set_source(Path("C:/raws.zip"))
    pump(qapp, lambda: "ImportPlanError" in view.status_label.text())

    assert view.status_label.text() == "ImportPlanError: can't read archive C:/raws.zip: bad magic"
    assert "red" in view.status_label.styleSheet()
    assert view.tree.topLevelItemCount() == 0
    assert not view.import_button.isEnabled()
    assert view.conversion_label.isHidden()
    assert view.plan_button.isEnabled()  # the user can retry


# ---------------------------------------------------------------- drop seam


def test_drop_seam_accepts_folders_and_archives_only(qapp: QApplication, tmp_path: Path) -> None:
    view = ImportView(FakeService(plan(MIXED)))
    folder = tmp_path / "raws"
    folder.mkdir()
    archive = tmp_path / "raws.cbz"
    archive.write_bytes(b"PK")
    text = tmp_path / "notes.txt"
    text.write_text("x")

    assert view._source_from_urls([QUrl.fromLocalFile(str(folder))]) == folder
    assert view._source_from_urls([QUrl.fromLocalFile(str(archive))]) == archive
    assert view._source_from_urls([QUrl.fromLocalFile(str(text))]) is None
    assert view._source_from_urls([QUrl("https://example.com/a.zip")]) is None  # not a local file
    assert view._source_from_urls([]) is None


# ---------------------------------------------------------------- real service end to end


def test_real_service_end_to_end_through_the_view(qapp: QApplication, tmp_path: Path) -> None:
    """Real ImporterService, real files: plan → edit nothing → commit; the PNG lands converted."""
    from PIL import Image

    from omniscan.core.config import Config, PathsConfig
    from omniscan.gui.services.importer import ImporterService

    class Gpu:
        name = "Fake Radeon"
        vram_gb = 16.0
        backend = "rocm"

    class Hw:
        gpus = (Gpu(),)

    src = tmp_path / "Real Series"
    (src / "Chapter 1").mkdir(parents=True)
    (src / "Chapter 2").mkdir(parents=True)
    for name in ("1.jpg", "2.jpg"):
        Image.new("RGB", (4, 4), (10, 20, 30)).save(src / "Chapter 1" / name)
    Image.new("RGB", (4, 4), (200, 10, 10)).save(src / "Chapter 2" / "1.png")
    cfg = Config(paths=PathsConfig(library_root=tmp_path / "lib", models_dir=tmp_path / "models"))
    service = ImporterService(cfg, hardware=lambda: Hw())  # pyright: ignore[reportArgumentType]
    view = ImportView(service)
    view.resize(1200, 800)
    view.show()
    qapp.processEvents()

    view.set_source(src)
    pump(qapp, lambda: view.tree.topLevelItemCount() == 2)
    assert not view.conversion_label.isHidden()
    assert "1 of 3" in view.conversion_label.text()

    view.import_button.click()
    pump(qapp, lambda: view.status_label.text().startswith("Imported:"))

    assert view.status_label.text() == "Imported: 2 chapter(s), 2 copied, 1 converted, 0 duplicate(s) skipped"
    assert (tmp_path / "lib" / "Real Series" / "Chapter 1" / "1.jpg").is_file()
    assert (tmp_path / "lib" / "Real Series" / "Chapter 2" / "1.jpg").is_file()  # the PNG, converted
    assert not (tmp_path / "lib" / "Real Series" / "Chapter 2" / "1.png").exists()

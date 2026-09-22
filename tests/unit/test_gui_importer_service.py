"""Tests for the Qt-free ImporterService and the plan-editing helpers (no Qt import in this file)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from omniscan.core.config import Config, PathsConfig
from omniscan.gui.services.importer import (
    ImporterService,
    conversion_text,
    merge_chapters,
    move_file,
    rename_chapter,
    set_series,
    split_chapter,
)
from omniscan.importer.execute import execute_import
from omniscan.importer.plan import ImportPlan, ImportPlanItem, plan_import

# ---------------------------------------------------------------- fixtures


def make_cfg(tmp_path: Path, **paths: Path) -> Config:
    """A Config whose library root (and models dir, for the hardware probe) live in tmp_path."""
    return Config(
        paths=PathsConfig(
            library_root=paths.get("library", tmp_path / "library"), models_dir=tmp_path / "models"
        )
    )


def _image(path: Path, color=(5, 10, 15), size=(4, 4)) -> Path:
    """One small synthetic image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _folder_source(tmp_path: Path) -> Path:
    """A folder-of-folders source: two chapters, one of them holding a convertible PNG."""
    src = tmp_path / "My Series"
    _image(src / "Chapter 1" / "p1.jpg")
    _image(src / "Chapter 1" / "p2.png")
    _image(src / "Chapter 2" / "p1.jpg")
    return src


def _archive_source(tmp_path: Path) -> Path:
    """A .cbz with two chapters, one page each, both JPEG."""
    src = tmp_path / "Archive Series.cbz"
    with zipfile.ZipFile(src, "w") as archive:
        archive.writestr("Chapter 1/1.jpg", b"jpeg bytes")
        archive.writestr("Chapter 2/1.jpg", b"jpeg bytes")
    return src


def _plan(items: list[tuple[str, list[str]]], series: str = "S") -> ImportPlan:
    """A hand-made plan whose files are plain names (editing helpers only shuffle them)."""
    return ImportPlan(
        series=series,
        items=[ImportPlanItem(chapter=name, files=[Path(f) for f in files]) for name, files in items],
        warnings=[],
    )


# ---------------------------------------------------------------- the service


def test_plan_wraps_plan_import(tmp_path: Path) -> None:
    service = ImporterService(make_cfg(tmp_path))

    plan = service.plan(_folder_source(tmp_path))

    assert plan.series == "My Series"
    assert [item.chapter for item in plan.items] == ["Chapter 1", "Chapter 2"]


def test_plan_accepts_an_archive(tmp_path: Path) -> None:
    service = ImporterService(make_cfg(tmp_path))

    plan = service.plan(_archive_source(tmp_path))

    assert plan.series == "Archive Series"
    assert plan.archive == _archive_source(tmp_path)


def test_execute_writes_into_the_configured_library(tmp_path: Path) -> None:
    service = ImporterService(make_cfg(tmp_path, library=tmp_path / "lib"))

    result = service.execute(service.plan(_folder_source(tmp_path)))

    assert (tmp_path / "lib" / "My Series" / "Chapter 1" / "p1.jpg").is_file()
    assert result.chapters_written == ["Chapter 1", "Chapter 2"]


def test_execute_reports_progress(tmp_path: Path) -> None:
    service = ImporterService(make_cfg(tmp_path))
    seen: list[tuple[int, int | None]] = []

    service.execute(
        service.plan(_folder_source(tmp_path)), on_progress=lambda done, total: seen.append((done, total))
    )

    assert seen[-1] == (3, 3)
    assert seen[0] == (1, 3)


def test_hardware_line_names_the_gpu_and_the_truth_about_the_codec(tmp_path: Path) -> None:
    class Gpu:
        name = "AMD Radeon RX 9070 XT"
        vram_gb = 16.0
        backend = "rocm"

    class Hw:
        gpus = (Gpu(),)

    service = ImporterService(make_cfg(tmp_path), hardware=lambda: Hw())  # pyright: ignore[reportArgumentType]

    line = service.hardware_line()

    assert "AMD Radeon RX 9070 XT" in line and "rocm" in line
    assert "CPU (libjpeg-turbo)" in line  # honest about what runs today


def test_hardware_line_without_gpus(tmp_path: Path) -> None:
    class Hw:
        gpus: tuple[Any, ...] = ()

    service = ImporterService(make_cfg(tmp_path), hardware=lambda: Hw())  # pyright: ignore[reportArgumentType]

    assert "no GPU" in service.hardware_line()


# ---------------------------------------------------------------- conversion preview


def test_conversion_text_counts_conversions(tmp_path: Path) -> None:
    plan = plan_import(_folder_source(tmp_path))

    assert (
        conversion_text(plan) == "1 of 3 file(s) will be converted to JPEG for consistent, fast processing."
    )


def test_conversion_text_empty_when_everything_is_jpeg(tmp_path: Path) -> None:
    src = tmp_path / "Only Jpeg"
    _image(src / "Chapter 1" / "p1.jpg")

    assert conversion_text(plan_import(src)) == ""


# ---------------------------------------------------------------- editing helpers


def test_set_series() -> None:
    plan = _plan([("Chapter 1", ["1.jpg"])])

    assert set_series(plan, "Renamed").series == "Renamed"
    assert plan.series == "S"  # the input plan is untouched


def test_rename_chapter() -> None:
    plan = _plan([("Chapter 1", ["1.jpg"]), ("Chapter 2", ["2.jpg"])])

    renamed = rename_chapter(plan, 1, "Chapter 2b")

    assert [item.chapter for item in renamed.items] == ["Chapter 1", "Chapter 2b"]


def test_move_file_to_another_chapter() -> None:
    plan = _plan([("Chapter 1", ["a.jpg", "b.jpg"]), ("Chapter 2", ["c.jpg"])])

    moved = move_file(plan, 0, 1, 1)  # b.jpg -> end of Chapter 2

    assert [(item.chapter, [p.name for p in item.files]) for item in moved.items] == [
        ("Chapter 1", ["a.jpg"]),
        ("Chapter 2", ["c.jpg", "b.jpg"]),
    ]


def test_move_file_to_a_position() -> None:
    plan = _plan([("Chapter 1", ["a.jpg", "b.jpg"]), ("Chapter 2", ["c.jpg"])])

    moved = move_file(plan, 1, 0, 0, 1)  # c.jpg -> Chapter 1, after a.jpg

    assert [p.name for p in moved.items[0].files] == ["a.jpg", "c.jpg", "b.jpg"]
    assert len(moved.items) == 1  # Chapter 2 emptied and vanished


def test_move_file_reorders_within_a_chapter() -> None:
    plan = _plan([("Chapter 1", ["a.jpg", "b.jpg", "c.jpg"])])

    up = move_file(plan, 0, 1, 0, 0)  # b up
    down = move_file(up, 0, 0, 0, 1)  # b back down

    assert [p.name for p in up.items[0].files] == ["b.jpg", "a.jpg", "c.jpg"]
    assert [p.name for p in down.items[0].files] == ["a.jpg", "b.jpg", "c.jpg"]


def test_merge_chapters_appends_in_order() -> None:
    plan = _plan([("Chapter 1", ["a.jpg"]), ("Chapter 2", ["b.jpg", "c.jpg"])])

    merged = merge_chapters(plan, 1, 0)

    assert [(item.chapter, [p.name for p in item.files]) for item in merged.items] == [
        ("Chapter 1", ["a.jpg", "b.jpg", "c.jpg"]),
    ]


def test_merge_into_itself_raises() -> None:
    with pytest.raises(ValueError):
        merge_chapters(_plan([("Chapter 1", ["a.jpg"])]), 0, 0)


def test_split_chapter_takes_the_tail() -> None:
    plan = _plan([("Chapter 1", ["a.jpg", "b.jpg", "c.jpg"]), ("Chapter 5", ["d.jpg"])])

    split = split_chapter(plan, 0, 1)

    assert [(item.chapter, [p.name for p in item.files]) for item in split.items] == [
        ("Chapter 1", ["a.jpg"]),
        ("Chapter 6", ["b.jpg", "c.jpg"]),  # named after the plan's top chapter number
        ("Chapter 5", ["d.jpg"]),
    ]


def test_split_needs_a_real_boundary() -> None:
    plan = _plan([("Chapter 1", ["a.jpg"])])

    with pytest.raises(ValueError):
        split_chapter(plan, 0, 0)
    with pytest.raises(ValueError):
        split_chapter(plan, 0, 1)


def test_edits_preserve_an_archive_extraction(tmp_path: Path) -> None:
    plan = plan_import(_archive_source(tmp_path))
    assert plan.temp_dir is not None

    edited = rename_chapter(move_file(plan, 0, 0, 1), 0, "Renamed")

    assert edited.temp_dir is plan.temp_dir  # dataclasses.replace carries the extraction handle
    assert edited.items[0].files[1].is_file()  # the moved file is still readable for the commit


def test_service_execute_commits_an_edited_plan(tmp_path: Path) -> None:
    service = ImporterService(make_cfg(tmp_path, library=tmp_path / "lib"))
    plan = move_file(service.plan(_folder_source(tmp_path)), 0, 1, 1)  # p2.png joins Chapter 2

    result = service.execute(rename_chapter(plan, 1, "Chapter 9"), on_progress=None)

    assert result.chapters_written == ["Chapter 1", "Chapter 9"]
    assert (tmp_path / "lib" / "My Series" / "Chapter 9" / "p2.jpg").is_file()  # converted on the way


def test_edited_plan_commits_through_plain_execute_import(tmp_path: Path) -> None:
    service = ImporterService(make_cfg(tmp_path))
    plan = split_chapter(service.plan(_folder_source(tmp_path)), 0, 1)

    result = execute_import(plan, tmp_path / "lib2")

    assert result.chapters_written == ["Chapter 1", "Chapter 3", "Chapter 2"]  # top number + 1, after item 0
    assert (
        tmp_path / "lib2" / "My Series" / "Chapter 3" / "p2.jpg"
    ).is_file()  # the PNG converted on the way
    assert not (tmp_path / "lib2" / "My Series" / "Chapter 3" / "p2.png").exists()

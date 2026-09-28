"""Tests for omniscan.importer.downloader (series folders written by manhwa-manga-downloader)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omniscan.importer.plan import ImportPlanError, ImportPlanItem, plan_import


def _chapter(series_dir: Path, folder: str, pages: int, suffix: str = ".jpg") -> Path:
    """One downloader chapter folder holding `pages` images named like the downloader's (`0001.jpg`, ...)."""
    path = series_dir / folder
    path.mkdir(parents=True)
    for index in range(1, pages + 1):
        (path / f"{index:04d}{suffix}").write_bytes(b"img" + str(index).encode())
    return path


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _chapters(plan_items: list[ImportPlanItem]) -> list[tuple[str, int]]:
    """(chapter, page count) per planned item."""
    return [(item.chapter, len(item.files)) for item in plan_items]


def test_numbered_folders_become_chapter_n_in_number_order(tmp_path: Path) -> None:
    src = tmp_path / "solo-leveling"
    _chapter(src, "num10_Chapter 10", 2)
    _chapter(src, "num2_Chapter 2", 3)
    _chapter(src, "num5.5_Chapter 5.5", 1)
    _write_json(src / "chapter_manifest.json", {"chapters": {"num10_Chapter 10": 2, "num2_Chapter 2": 3}})

    plan = plan_import(src)

    assert plan.series == "solo-leveling"
    assert _chapters(plan.items) == [("Chapter 2", 3), ("Chapter 5.5", 1), ("Chapter 10", 2)]
    assert [p.name for p in plan.items[0].files] == ["0001.jpg", "0002.jpg", "0003.jpg"]
    assert plan.warnings == []  # the bookkeeping JSON is not reported as a skipped file


def test_wfwf_style_names_and_series_override(tmp_path: Path) -> None:
    src = tmp_path / "wfwf504" / "1234"
    _chapter(src, "num1_chapter", 2)
    _chapter(src, "num2_chapter", 2)

    plan = plan_import(src, series="Solo Leveling")

    assert plan.series == "Solo Leveling"
    assert _chapters(plan.items) == [("Chapter 1", 2), ("Chapter 2", 2)]


def test_numeric_series_folder_needs_series(tmp_path: Path) -> None:
    src = tmp_path / "1234"  # wfwf names the series folder after the site's toon id
    _chapter(src, "num1_chapter", 1)

    with pytest.raises(ImportPlanError, match="--series"):
        plan_import(src)


def test_incomplete_chapter_is_skipped_with_warning(tmp_path: Path) -> None:
    src = tmp_path / "series"
    _chapter(src, "num1_Chapter 1", 2)
    _chapter(src, "num2_Chapter 2", 1)
    _write_json(
        src / "incomplete_chapters.json",
        {"chapters": [{"folder": "num2_Chapter 2", "downloaded": 1, "total": 4, "missing": 3}]},
    )

    plan = plan_import(src)

    assert _chapters(plan.items) == [("Chapter 1", 2)]
    assert plan.warnings == [
        "skipped num2_Chapter 2: incomplete download (1/4 pages) — finish it with the downloader, then import again"
    ]


def test_part_file_marks_an_interrupted_chapter(tmp_path: Path) -> None:
    src = tmp_path / "series"
    _chapter(src, "num1_Chapter 1", 2)
    folder = _chapter(src, "num2_Chapter 2", 2)
    (folder / "0003.jpg.part").write_bytes(b"half")

    plan = plan_import(src)

    assert _chapters(plan.items) == [("Chapter 1", 2)]
    assert len(plan.warnings) == 1
    assert "num2_Chapter 2: the download was interrupted" in plan.warnings[0]


def test_page_count_below_manifest_is_skipped(tmp_path: Path) -> None:
    src = tmp_path / "series"
    _chapter(src, "num1_Chapter 1", 2)
    _chapter(src, "num2_Chapter 2", 2)
    _write_json(src / "chapter_manifest.json", {"chapters": {"num1_Chapter 1": 2, "num2_Chapter 2": 5}})

    plan = plan_import(src)

    assert _chapters(plan.items) == [("Chapter 1", 2)]
    assert "2 page(s) on disk, the downloader finished it with 5" in plan.warnings[0]


@pytest.mark.parametrize("folder", ["num0_Chapter unknown", "num0_death-note", "numunknown_chapter"])
def test_unnumbered_fallback_folders_are_skipped(tmp_path: Path, folder: str) -> None:
    src = tmp_path / "series"
    _chapter(src, "num1_Chapter 1", 1)
    _chapter(src, folder, 1)

    plan = plan_import(src)

    assert _chapters(plan.items) == [("Chapter 1", 1)]
    assert plan.warnings == [
        f"skipped {folder}: the downloader recorded no chapter number — import it on its own with --series and --chapter"
    ]


def test_real_chapter_zero_is_kept(tmp_path: Path) -> None:
    src = tmp_path / "series"
    _chapter(src, "num0_Chapter 0", 1)
    _chapter(src, "num1_Chapter 1", 1)

    plan = plan_import(src)

    assert _chapters(plan.items) == [("Chapter 0", 1), ("Chapter 1", 1)]


def test_non_jpeg_pages_are_planned_for_conversion(tmp_path: Path) -> None:
    src = tmp_path / "series"
    _chapter(src, "num3_Chapter 3", 2, suffix=".webp")

    plan = plan_import(src)

    assert [p.name for p in plan.items[0].files] == ["0001.webp", "0002.webp"]


def test_nothing_importable_raises_with_reasons(tmp_path: Path) -> None:
    src = tmp_path / "series"
    _chapter(src, "num0_death-note", 1)

    with pytest.raises(ImportPlanError, match=r"nothing to import .*num0_death-note"):
        plan_import(src)


def test_two_folders_with_the_same_number_raise(tmp_path: Path) -> None:
    src = tmp_path / "series"
    _chapter(src, "num4_Chapter 4", 1)
    _chapter(src, "num4.0_Chapter 4", 1)

    with pytest.raises(ImportPlanError, match="both chapter 4"):
        plan_import(src)


def test_chapter_option_is_ambiguous(tmp_path: Path) -> None:
    src = tmp_path / "series"
    _chapter(src, "num1_Chapter 1", 1)

    with pytest.raises(ImportPlanError, match="--chapter"):
        plan_import(src, chapter="Chapter 1")


def test_unreadable_bookkeeping_is_ignored(tmp_path: Path) -> None:
    src = tmp_path / "series"
    _chapter(src, "num1_Chapter 1", 1)
    (src / "chapter_manifest.json").write_text("{not json", encoding="utf-8")
    _write_json(src / "incomplete_chapters.json", {"chapters": "oops"})

    plan = plan_import(src)

    assert _chapters(plan.items) == [("Chapter 1", 1)]
    assert plan.warnings == []


def test_other_loose_files_still_warn(tmp_path: Path) -> None:
    src = tmp_path / "series"
    _chapter(src, "num1_Chapter 1", 1)
    (src / "notes.txt").write_text("x", encoding="utf-8")

    plan = plan_import(src)

    assert plan.warnings == ["skipped non-image file: notes.txt"]


def test_ordinary_chapter_folders_keep_their_names(tmp_path: Path) -> None:
    # Only a folder whose every subfolder has the downloader's `num<N>_` name takes this path.
    src = tmp_path / "series"
    _chapter(src, "Chapter 1", 1)
    _chapter(src, "num2_Chapter 2", 1)

    plan = plan_import(src)

    assert _chapters(plan.items) == [("Chapter 1", 1), ("num2_Chapter 2", 1)]

"""Tests for omniscan.importer.plan."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.importer.plan import ImportPlanError, plan_import


def _write_files(folder: Path, names: list[str], content: bytes = b"img") -> None:
    """Create `folder` and one file per name with (optionally distinct) content."""
    folder.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        (folder / name).write_bytes(content + str(index).encode())


def test_case_a_natural_order(tmp_path: Path) -> None:
    src = tmp_path / "Chapter 7"
    _write_files(src, ["3.jpg", "1.jpg", "10.jpg", "2.jpg", "5.jpg"])

    plan = plan_import(src, series="Solo Leveling")

    assert plan.series == "Solo Leveling"
    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.chapter == "Chapter 7"
    assert [p.name for p in item.files] == ["1.jpg", "2.jpg", "3.jpg", "5.jpg", "10.jpg"]
    assert all(p.is_absolute() for p in item.files)
    assert plan.warnings == []


def test_case_a_folder_name_without_chapter_number(tmp_path: Path) -> None:
    # No --chapter, folder name "raws" doesn't parse, filenames carry no numbers either.
    src = tmp_path / "raws"
    _write_files(src, ["a.jpg", "b.jpg", "c.jpg"])

    with pytest.raises(ImportPlanError):
        plan_import(src, series="Solo Leveling")


def test_case_a_series_required(tmp_path: Path) -> None:
    src = tmp_path / "Chapter 7"
    _write_files(src, ["1.jpg", "2.jpg"])

    with pytest.raises(ImportPlanError):
        plan_import(src)  # chapter derivable from the folder name, but no series


def test_case_a_skips_non_image_with_warning(tmp_path: Path) -> None:
    src = tmp_path / "Chapter 3"
    _write_files(src, ["1.jpg", "2.jpg", "readme.txt"])

    plan = plan_import(src, series="Solo Leveling")

    assert plan.warnings == ["skipped non-image file: readme.txt"]
    assert [p.name for p in plan.items[0].files] == ["1.jpg", "2.jpg"]


def test_case_b_folder_of_folders_natural_order(tmp_path: Path) -> None:
    src = tmp_path / "My Manhwa"
    _write_files(src / "Chapter 10", ["2.jpg", "1.jpg"])
    _write_files(src / "Chapter 2", ["1.jpg", "2.jpg"])
    _write_files(src / "Chapter 1", ["1.jpg", "2.jpg"])

    plan = plan_import(src)

    assert plan.series == "My Manhwa"  # defaults to the source folder's own name
    assert [(item.chapter, len(item.files)) for item in plan.items] == [
        ("Chapter 1", 2),
        ("Chapter 2", 2),
        ("Chapter 10", 2),
    ]


def test_case_b_chapter_option_is_ambiguous(tmp_path: Path) -> None:
    src = tmp_path / "My Manhwa"
    _write_files(src / "Chapter 1", ["1.jpg", "2.jpg"])

    with pytest.raises(ImportPlanError):
        plan_import(src, series="Solo Leveling", chapter="Chapter 1")


def test_case_c_flat_dump_grouped_by_per_file_number(tmp_path: Path) -> None:
    src = tmp_path / "dump"  # no chapter in the folder name; numbers come from the filenames
    _write_files(src, ["S_Ch2_01.jpg", "S_Ch1_02.jpg", "S_Ch1_01.jpg"])

    plan = plan_import(src, series="Solo Leveling")

    assert [(item.chapter, [p.name for p in item.files]) for item in plan.items] == [
        ("Chapter 1", ["S_Ch1_01.jpg", "S_Ch1_02.jpg"]),
        ("Chapter 2", ["S_Ch2_01.jpg"]),
    ]


def test_case_c_unparseable_filename_raises(tmp_path: Path) -> None:
    src = tmp_path / "dump"
    _write_files(src, ["S_Ch1_01.jpg", "cover.jpg"])

    with pytest.raises(ImportPlanError, match=r"cover\.jpg"):
        plan_import(src, series="Solo Leveling")


def test_mixed_subfolder_and_loose_images(tmp_path: Path) -> None:
    src = tmp_path / "mixed"
    _write_files(src / "Chapter 1", ["1.jpg"])
    _write_files(src, ["a.jpg"])

    with pytest.raises(ImportPlanError):
        plan_import(src, series="Solo Leveling")


def test_nonexistent_source(tmp_path: Path) -> None:
    with pytest.raises(ImportPlanError, match="nope"):
        plan_import(tmp_path / "nope", series="Solo Leveling")


def test_empty_source(tmp_path: Path) -> None:
    src = tmp_path / "empty"
    src.mkdir()

    with pytest.raises(ImportPlanError, match="empty"):
        plan_import(src, series="Solo Leveling")


def test_case_c_wins_over_case_a_when_folder_has_no_chapter(tmp_path: Path) -> None:
    # Every filename parses a number while the folder name doesn't: flat dump, not "ask for --chapter".
    src = tmp_path / "raws"
    _write_files(src, ["3.jpg", "1.jpg"])

    plan = plan_import(src, series="Solo Leveling")

    assert [(item.chapter, len(item.files)) for item in plan.items] == [
        ("Chapter 1", 1),
        ("Chapter 3", 1),
    ]

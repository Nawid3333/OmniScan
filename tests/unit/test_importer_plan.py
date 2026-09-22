"""Tests for omniscan.importer.plan."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from omniscan.importer.plan import (
    ImportPlanError,
    ImportPlanItem,
    files_to_convert,
    plan_import,
)


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


def test_bare_page_numbers_do_not_trigger_case_c(tmp_path: Path) -> None:
    # Bare page numbers with no chapter keyword ("1.jpg", "3.jpg") are page numbers of ONE chapter, not
    # markers of three different one-page chapters — Case C requires an explicit "ch"/"chapter"/"ep" marker.
    # With no --chapter and an unparseable folder name, this must ask for --chapter (Case A), not silently
    # split into per-file chapters.
    src = tmp_path / "raws"
    _write_files(src, ["3.jpg", "1.jpg"])

    with pytest.raises(ImportPlanError, match="--chapter"):
        plan_import(src, series="Solo Leveling")


def test_bare_page_numbers_become_one_chapter_with_explicit_chapter(tmp_path: Path) -> None:
    src = tmp_path / "raws"
    _write_files(src, ["3.jpg", "1.jpg"])

    plan = plan_import(src, series="Solo Leveling", chapter="Chapter 7")

    assert len(plan.items) == 1
    assert plan.items[0].chapter == "Chapter 7"
    assert [p.name for p in plan.items[0].files] == ["1.jpg", "3.jpg"]


def test_case_c_requires_explicit_chapter_keyword(tmp_path: Path) -> None:
    # Mixed: one file has an explicit "ch" marker, the other is a bare number — still ambiguous, not Case C.
    src = tmp_path / "dump"
    _write_files(src, ["Ch1_01.jpg", "02.jpg"])

    with pytest.raises(ImportPlanError, match=r"02\.jpg"):
        plan_import(src, series="Solo Leveling")


# ---------------------------------------------------------------- archives


def _make_zip(path: Path, members: dict[str, bytes]) -> Path:
    """A .zip (or .cbz by name) holding the given member names and contents."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def test_zip_folder_of_folders(tmp_path: Path) -> None:
    src = _make_zip(tmp_path / "My Series.zip", {"Chapter 10/2.jpg": b"i", "Chapter 2/1.jpg": b"i"})

    plan = plan_import(src)

    assert plan.series == "My Series"  # the archive's own name without extension
    assert plan.archive == src
    assert [(item.chapter, [p.name for p in item.files]) for item in plan.items] == [
        ("Chapter 2", ["1.jpg"]),
        ("Chapter 10", ["2.jpg"]),
    ]
    assert plan.warnings == []


def test_cbz_single_chapter_folder(tmp_path: Path) -> None:
    src = _make_zip(tmp_path / "One Chapter.cbz", {"Chapter 5/2.jpg": b"i", "Chapter 5/1.jpg": b"i"})

    plan = plan_import(src)

    assert plan.series == "One Chapter"  # Case A still needs a series; the stem provides it
    assert len(plan.items) == 1
    assert plan.items[0].chapter == "Chapter 5"
    assert [p.name for p in plan.items[0].files] == ["1.jpg", "2.jpg"]


def test_zip_flat_dump_grouped_by_filename_marker(tmp_path: Path) -> None:
    src = _make_zip(tmp_path / "dump.zip", {"S_Ch2_01.jpg": b"i", "S_Ch1_02.jpg": b"i", "S_Ch1_01.jpg": b"i"})

    plan = plan_import(src)

    assert [(item.chapter, [p.name for p in item.files]) for item in plan.items] == [
        ("Chapter 1", ["S_Ch1_01.jpg", "S_Ch1_02.jpg"]),
        ("Chapter 2", ["S_Ch2_01.jpg"]),
    ]


def test_zip_wrapper_folder_is_unwrapped(tmp_path: Path) -> None:
    # Archive tools usually export a series as one wrapping folder; the plan descends into it.
    src = _make_zip(
        tmp_path / "My Manhwa.zip",
        {"My Manhwa/Chapter 1/1.jpg": b"i", "My Manhwa/Chapter 2/1.jpg": b"i"},
    )

    plan = plan_import(src)

    assert plan.series == "My Manhwa"
    assert [item.chapter for item in plan.items] == ["Chapter 1", "Chapter 2"]


def test_zip_series_option_overrides_stem(tmp_path: Path) -> None:
    src = _make_zip(tmp_path / "whatever.zip", {"Chapter 1/1.jpg": b"i"})

    plan = plan_import(src, series="Solo Leveling")

    assert plan.series == "Solo Leveling"


def test_zip_warns_about_non_image_members(tmp_path: Path) -> None:
    src = _make_zip(tmp_path / "My Series.zip", {"Chapter 1/1.jpg": b"i", "cover.txt": b"t"})

    plan = plan_import(src)

    assert plan.warnings == ["skipped non-image file: cover.txt"]


def test_corrupt_zip_raises_clear_error(tmp_path: Path) -> None:
    src = tmp_path / "broken.zip"
    src.write_bytes(b"this is not a zip file")

    with pytest.raises(ImportPlanError, match="can't read archive"):
        plan_import(src)


def test_missing_archive_raises(tmp_path: Path) -> None:
    with pytest.raises(ImportPlanError, match="source archive not found"):
        plan_import(tmp_path / "nope.cbz")


def test_empty_zip_raises(tmp_path: Path) -> None:
    src = _make_zip(tmp_path / "empty.zip", {})

    with pytest.raises(ImportPlanError, match="source folder is empty"):
        plan_import(src)


def test_zip_slip_member_name_is_refused(tmp_path: Path) -> None:
    src = _make_zip(tmp_path / "evil.zip", {"../escape.txt": b"x"})

    with pytest.raises(ImportPlanError, match="unsafe member name"):
        plan_import(src)


def test_plan_cleanup_removes_extraction(tmp_path: Path) -> None:
    src = _make_zip(tmp_path / "My Series.zip", {"Chapter 1/1.jpg": b"i"})

    plan = plan_import(src)
    extracted = plan.items[0].files[0]
    assert extracted.is_file()  # the plan's files point into the extraction
    assert plan.temp_dir is not None

    plan.cleanup()
    plan.cleanup()  # twice is fine (TemporaryDirectory cleanup is idempotent)

    assert not extracted.exists()


def test_edited_plan_keeps_the_extraction_alive(tmp_path: Path) -> None:
    from dataclasses import replace

    plan = plan_import(_make_zip(tmp_path / "My Series.zip", {"Chapter 1/1.jpg": b"i"}))
    assert plan.temp_dir is not None

    edited = replace(plan, items=[ImportPlanItem(chapter="Chapter 1", files=list(plan.items[0].files))])

    assert edited.temp_dir is plan.temp_dir  # dataclasses.replace preserves the extraction


def test_files_to_convert_lists_only_non_jpegs(tmp_path: Path) -> None:
    src = tmp_path / "mixed"
    _write_files(src, ["p1.jpg", "p2.png", "p3.bmp", "p4.jpeg"])

    plan = plan_import(src, series="Solo Leveling", chapter="Chapter 1")

    assert [p.name for p in files_to_convert(plan)] == ["p2.png", "p3.bmp"]
    all_jpeg = plan_import(_make_zip(tmp_path / "alljpg.zip", {"Chapter 1/1.jpg": b"i"}))
    assert files_to_convert(all_jpeg) == []

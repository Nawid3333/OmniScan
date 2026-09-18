"""Tests for omniscan.importer.execute."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.core.manifest import hash_file
from omniscan.importer.execute import execute_import
from omniscan.importer.plan import ImportPlanError, plan_import


def _write_files(folder: Path, names: list[str]) -> dict[str, bytes]:
    """Create `folder` with one distinctly-contented file per name; return name -> content."""
    folder.mkdir(parents=True, exist_ok=True)
    contents = {name: f"content of {name} in {folder.name}".encode() for name in names}
    for name, content in contents.items():
        (folder / name).write_bytes(content)
    return contents


def _case_a_plan(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    """Plan a single-chapter source folder; returns (source dir, per-file original content)."""
    src = tmp_path / "raws" / "Chapter 12"
    contents = _write_files(src, ["1.jpg", "2.jpg", "3.jpg"])
    return src, contents


def test_copies_files_byte_identical(tmp_path: Path) -> None:
    src, contents = _case_a_plan(tmp_path)
    plan = plan_import(src, series="Solo Leveling")
    library = tmp_path / "library"

    result = execute_import(plan, library)

    assert result.files_copied == 3
    assert result.files_skipped_duplicate == 0
    assert result.chapters_written == ["Chapter 12"]
    dest_dir = library / "Solo Leveling" / "Chapter 12"
    for name, content in contents.items():
        dest = dest_dir / name
        assert dest.is_file()
        assert hash_file(dest) == hash_file(src / name)
        assert dest.read_bytes() == content
        assert (src / name).is_file()  # move=False leaves the source alone


def test_move_deletes_source(tmp_path: Path) -> None:
    src, contents = _case_a_plan(tmp_path)
    plan = plan_import(src, series="Solo Leveling")
    library = tmp_path / "library"

    result = execute_import(plan, library, move=True)

    assert result.files_copied == 3
    dest_dir = library / "Solo Leveling" / "Chapter 12"
    for name, content in contents.items():
        assert not (src / name).exists()
        assert (dest_dir / name).read_bytes() == content


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    src, _ = _case_a_plan(tmp_path)
    plan = plan_import(src, series="Solo Leveling")
    library = tmp_path / "library"
    execute_import(plan, library)
    dest_dir = library / "Solo Leveling" / "Chapter 12"
    before = {p.name for p in dest_dir.iterdir()}

    second = execute_import(plan, library)

    assert second.files_copied == 0
    assert second.files_skipped_duplicate == len(plan.items[0].files)
    assert {p.name for p in dest_dir.iterdir()} == before  # no "file (1).jpg"-style renames


def test_conflicting_destination_raises(tmp_path: Path) -> None:
    src, _ = _case_a_plan(tmp_path)
    plan = plan_import(src, series="Solo Leveling")
    library = tmp_path / "library"
    dest = library / "Solo Leveling" / "Chapter 12" / "1.jpg"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"something else entirely")

    with pytest.raises(ImportPlanError) as excinfo:
        execute_import(plan, library)

    message = str(excinfo.value)
    assert str(src / "1.jpg") in message
    assert str(dest) in message
    assert (src / "1.jpg").read_bytes() != dest.read_bytes()  # nothing was overwritten


def test_chapters_written_in_plan_order_when_all_duplicates(tmp_path: Path) -> None:
    src = tmp_path / "raws" / "My Manhwa"
    _write_files(src / "Chapter 1", ["1.jpg", "2.jpg"])
    _write_files(src / "Chapter 10", ["1.jpg", "2.jpg"])
    plan = plan_import(src, series="Solo Leveling")
    library = tmp_path / "library"
    execute_import(plan, library)

    second = execute_import(plan, library)

    assert second.files_copied == 0
    assert second.chapters_written == [item.chapter for item in plan.items]


def test_move_with_existing_identical_destination_deletes_source(tmp_path: Path) -> None:
    src, _ = _case_a_plan(tmp_path)
    plan = plan_import(src, series="Solo Leveling")
    library = tmp_path / "library"
    execute_import(plan, library)
    dest = library / "Solo Leveling" / "Chapter 12" / "1.jpg"
    before = dest.read_bytes()

    result = execute_import(plan, library, move=True)

    assert result.files_copied == 0
    assert result.files_skipped_duplicate == len(plan.items[0].files)
    assert not (src / "1.jpg").exists()
    assert dest.read_bytes() == before

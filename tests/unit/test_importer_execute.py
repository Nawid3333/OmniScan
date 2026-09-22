"""Tests for omniscan.importer.execute."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from omniscan.core.manifest import hash_file
from omniscan.importer.execute import execute_import
from omniscan.importer.plan import ImportPlanError, plan_import
from omniscan.ingest.convert import needs_conversion


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


# ---------------------------------------------------------------- JPEG conversion


def _image(path: Path, mode: str = "RGB", color=(10, 20, 30), size=(2, 3)) -> Path:
    """One synthetic image file (the only image content in this suite that is a real image)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path)
    return path


def test_png_is_converted_to_jpeg(tmp_path: Path) -> None:
    src = tmp_path / "raws" / "Chapter 12"
    _image(src / "p1.png")
    _image(src / "p2.jpg", color=(1, 2, 3))
    plan = plan_import(src, series="Solo Leveling")
    library = tmp_path / "library"

    result = execute_import(plan, library)

    dest_dir = library / "Solo Leveling" / "Chapter 12"
    dest = dest_dir / "p1.jpg"  # the .png name becomes .jpg
    with Image.open(dest) as img:
        assert img.format == "JPEG"
        assert img.mode == "RGB"
        assert img.size == (2, 3)
    assert needs_conversion(dest) is False  # the ingest stage would accept it as-is
    assert (dest_dir / "p2.jpg").read_bytes() == (src / "p2.jpg").read_bytes()  # JPEGs stay byte-copies
    assert result.files_copied == 1
    assert result.files_converted == 1
    assert result.converted == ["p1.png"]
    assert result.files_skipped_duplicate == 0


def test_alpha_flattened_over_white(tmp_path: Path) -> None:
    # 16x16 so each half is whole 8x8 JPEG blocks: no DC sharing, flat colours survive q95 nearly exactly.
    src = tmp_path / "raws" / "Chapter 1"
    rgba_image = Image.new("RGBA", (16, 16), (10, 20, 30, 255))
    for y in range(8):  # the top half fully transparent -> must flatten to white
        for x in range(16):
            rgba_image.putpixel((x, y), (0, 0, 0, 0))
    src.mkdir(parents=True, exist_ok=True)
    rgba_image.save(src / "p1.png")
    plan = plan_import(src, series="Solo Leveling")

    result = execute_import(plan, tmp_path / "library")

    dest = tmp_path / "library" / "Solo Leveling" / "Chapter 1" / "p1.jpg"
    with Image.open(dest) as img:
        assert img.format == "JPEG" and img.mode == "RGB"
        top = img.getpixel((0, 0))
        bottom = img.getpixel((0, 15))
    assert all(abs(channel - 255) <= 2 for channel in top)  # the transparent half became white
    assert all(abs(channel - expected) <= 2 for channel, expected in zip(bottom, (10, 20, 30)))
    assert result.files_converted == 1


def test_gif_uses_first_frame(tmp_path: Path) -> None:
    src = tmp_path / "raws" / "Chapter 2"
    gif = src / "p1.gif"
    gif.parent.mkdir(parents=True)
    first = Image.new("RGB", (4, 4), (0, 0, 0))
    second = Image.new("RGB", (4, 4), (255, 255, 255))
    first.save(gif, save_all=True, append_images=[second], duration=100, loop=0)
    plan = plan_import(src, series="Solo Leveling")

    result = execute_import(plan, tmp_path / "library")

    dest = tmp_path / "library" / "Solo Leveling" / "Chapter 2" / "p1.jpg"
    with Image.open(dest) as img:
        assert img.format == "JPEG" and img.size == (4, 4)
        mean = sum(sum(img.getpixel((x, y))[:3]) / 3 for x in range(4) for y in range(4)) / 16
    assert mean < 32  # the black first frame, not the white second one
    assert result.converted == ["p1.gif"]


def test_bmp_converted_to_jpeg(tmp_path: Path) -> None:
    src = tmp_path / "raws" / "Chapter 3"
    _image(src / "p1.bmp")
    plan = plan_import(src, series="Solo Leveling")

    result = execute_import(plan, tmp_path / "library")

    with Image.open(tmp_path / "library" / "Solo Leveling" / "Chapter 3" / "p1.jpg") as img:
        assert img.format == "JPEG"
    assert result.files_converted == 1


def test_rerun_does_not_reconvert(tmp_path: Path) -> None:
    src = tmp_path / "raws" / "Chapter 4"
    _image(src / "p1.png")
    plan = plan_import(src, series="Solo Leveling")
    library = tmp_path / "library"
    execute_import(plan, library)
    dest = library / "Solo Leveling" / "Chapter 4" / "p1.jpg"
    before = dest.read_bytes()

    second = execute_import(plan, library)

    assert second.files_converted == 0
    assert second.files_copied == 0
    assert second.files_skipped_duplicate == 1  # the freshly encoded bytes match what is in place
    assert dest.read_bytes() == before


def test_move_converts_and_deletes_source(tmp_path: Path) -> None:
    src = tmp_path / "raws" / "Chapter 5"
    _image(src / "p1.png")
    plan = plan_import(src, series="Solo Leveling")

    result = execute_import(plan, tmp_path / "library", move=True)

    assert result.files_converted == 1
    assert not (src / "p1.png").exists()
    assert (tmp_path / "library" / "Solo Leveling" / "Chapter 5" / "p1.jpg").is_file()


def test_conflicting_converted_destination_raises(tmp_path: Path) -> None:
    src = tmp_path / "raws" / "Chapter 6"
    _image(src / "p1.png", color=(9, 9, 9))
    plan = plan_import(src, series="Solo Leveling")
    library = tmp_path / "library"
    dest = library / "Solo Leveling" / "Chapter 6" / "p1.jpg"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"a different jpeg")

    with pytest.raises(ImportPlanError) as excinfo:
        execute_import(plan, library)

    assert "destination exists with different content" in str(excinfo.value)
    assert dest.read_bytes() == b"a different jpeg"  # nothing was overwritten


def test_png_and_jpeg_same_stem_conflict_raises(tmp_path: Path) -> None:
    # p1.png and p1.jpg would both land as p1.jpg; different contents must fail fast, not overwrite.
    src = tmp_path / "raws" / "Chapter 6"
    _image(src / "p1.png", color=(1, 1, 1))
    _image(src / "p1.jpg", color=(250, 250, 250))
    plan = plan_import(src, series="Solo Leveling")

    with pytest.raises(ImportPlanError, match="destination exists with different content"):
        execute_import(plan, tmp_path / "library")


def test_corrupt_image_fails_fast_with_clear_message(tmp_path: Path) -> None:
    src = tmp_path / "raws" / "Chapter 7"
    (src / "p1.png").parent.mkdir(parents=True)
    (src / "p1.png").write_bytes(b"not an image")
    plan = plan_import(src, series="Solo Leveling")

    with pytest.raises(ImportPlanError, match="can't convert"):
        execute_import(plan, tmp_path / "library")


def test_progress_reports_every_file(tmp_path: Path) -> None:
    src = tmp_path / "raws" / "Chapter 7"
    _image(src / "p1.png")
    _image(src / "p2.jpg", color=(2, 2, 2))
    plan = plan_import(src, series="Solo Leveling")
    seen: list[tuple[int, int]] = []

    execute_import(plan, tmp_path / "library", on_progress=lambda done, total: seen.append((done, total)))

    assert seen == [(1, 2), (2, 2)]

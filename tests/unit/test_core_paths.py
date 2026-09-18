from pathlib import Path

from omniscan.core.config import Config, PathsConfig
from omniscan.core.paths import SeriesPaths, chapter_number, list_chapters, list_images, natural_key


def test_natural_key_orders_numbers() -> None:
    names = ["10.jpg", "2.jpg", "1.jpg", "001a.jpg"]
    assert sorted(names, key=natural_key) == ["1.jpg", "001a.jpg", "2.jpg", "10.jpg"]


def test_chapter_number_variants() -> None:
    assert chapter_number("Chapter 12") == 12.0
    assert chapter_number("Ch.12.5") == 12.5
    assert chapter_number("ep 7") == 7.0
    assert chapter_number("Prologue") is None


def test_list_chapters_reading_order_skips_special(tmp_path: Path) -> None:
    for name in ["Chapter 10", "Chapter 2", "Chapter 1", "_reference_en", "_filtered", "Prologue"]:
        (tmp_path / name).mkdir()
    assert [p.name for p in list_chapters(tmp_path)] == ["Chapter 1", "Chapter 2", "Chapter 10", "Prologue"]


def test_list_images_filters_and_sorts(tmp_path: Path) -> None:
    for name in ["10.jpg", "2.PNG", "notes.txt", "1.webp"]:
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in list_images(tmp_path)] == ["1.webp", "2.PNG", "10.jpg"]


def test_series_and_chapter_paths(tmp_path: Path) -> None:
    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )
    sp = SeriesPaths.from_config(cfg, "My Series")
    cp = sp.chapter("Chapter 3")
    assert cp.raw_dir == tmp_path / "lib" / "My Series" / "Chapter 3"
    assert cp.manifest == tmp_path / "work" / "My Series" / "Chapter 3" / "manifest.json"
    assert cp.filtered_dir == tmp_path / "out" / "My Series" / "_filtered" / "Chapter 3"
    assert sp.reference_dir == tmp_path / "lib" / "My Series" / "_reference_en"

"""Library service tests (Qt-free): series/chapter listing and chapter tile loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from omniscan.core.config import Config, PathsConfig
from omniscan.core.schemas import (
    ExportArtifact,
    ExportFile,
    IngestArtifact,
    SlicesArtifact,
    Slice,
    SourceFile,
)
from omniscan.gui.services import library


def _config(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
        )
    )


def _write_png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    config = _config(tmp_path)
    for sub in ("library", "work", "output"):
        (tmp_path / sub).mkdir()
    return config


# ---------------------------------------------------------------------- list_series / list_chapter_names


def test_list_series_unions_roots_and_skips_special_dirs(cfg: Config, tmp_path: Path) -> None:
    (tmp_path / "library" / "Series 10").mkdir()
    (tmp_path / "library" / "Series 2").mkdir()
    (tmp_path / "library" / "_reference_en").mkdir()
    (tmp_path / "library" / ".hidden").mkdir()
    (tmp_path / "output" / "Series 33").mkdir()  # only in output: still listed
    (tmp_path / "library" / "file.txt").write_text("not a dir")

    assert library.list_series(cfg) == ["Series 2", "Series 10", "Series 33"]


def test_list_series_missing_roots(tmp_path: Path) -> None:
    assert library.list_series(_config(tmp_path)) == []


def test_list_chapter_names_order_and_output_fallback(cfg: Config, tmp_path: Path) -> None:
    raw = tmp_path / "library" / "S"
    (raw / "Chapter 10").mkdir(parents=True)
    (raw / "Chapter 2").mkdir()
    assert library.list_chapter_names(cfg, "S") == ["Chapter 2", "Chapter 10"]

    (raw / "Chapter 2").rmdir()  # library now empty -> fall back to output folders
    (raw / "Chapter 10").rmdir()
    out = tmp_path / "output" / "S"
    (out / "Chapter 10").mkdir(parents=True)
    (out / "Chapter 2").mkdir()
    (out / "_filtered").mkdir()
    assert library.list_chapter_names(cfg, "S") == ["Chapter 2", "Chapter 10"]


# ---------------------------------------------------------------------- full artifacts


def _full_artifact_chapter(cfg: Config) -> None:
    """Build series 'S' / chapter 'Chapter 1' with ingest/slices/export artifacts and images."""
    sp = cfg.paths
    raw_dir = sp.library_root / "S" / "Chapter 1"
    out_dir = sp.output_root / "S" / "Chapter 1"
    work_dir = sp.work_root / "S" / "Chapter 1"
    for i in range(4):
        _write_png(raw_dir / f"page_{i:03d}.png", (800, 500), (20 * i, 0, 0))
    for i in (0, 1, 3):
        _write_png(out_dir / f"slice_{i:04d}.jpg", (800, 1000), (0, 20 * i, 0))
    IngestArtifact(
        series="S",
        chapter="Chapter 1",
        strip_width=800,
        strip_height=3000,
        files=[
            SourceFile(index=0, name="page_000.png", sha256="a", width=800, height=1000, y0=0, y1=1000),
            SourceFile(
                index=1, name="page_001.png", sha256="b", width=800, height=1500, y0=1000, y1=2500
            ),
            SourceFile(
                index=2, name="page_002.png", sha256="c", width=800, height=500, y0=2500, y1=3000
            ),
            SourceFile(
                index=3,
                name="page_003.png",
                sha256="d",
                width=800,
                height=500,
                y0=3000,
                y1=3500,
                filtered=True,
            ),
        ],
    ).save(work_dir / "ingest.json")
    SlicesArtifact(
        strip_width=800,
        strip_height=3000,
        bands=[],
        slices=[
            Slice(index=0, y0=0, y1=1200),
            Slice(index=1, y0=1200, y1=2400),
            Slice(index=2, y0=2400, y1=2700, filtered=True),
            Slice(index=3, y0=2700, y1=3000),
        ],
    ).save(work_dir / "slices.json")
    ExportArtifact(
        quality=90,
        subsampling="444",
        files=[
            ExportFile(name="slice_0000.jpg", slice_index=0, width=800, height=1200, bytes=1),
            ExportFile(name="slice_0001.jpg", slice_index=1, width=800, height=1200, bytes=1),
            ExportFile(name="slice_0003.jpg", slice_index=3, width=800, height=300, bytes=1),
        ],
    ).save(work_dir / "export.json")


def test_load_chapter_view_full_artifacts(cfg: Config) -> None:
    _full_artifact_chapter(cfg)
    view = library.load_chapter_view(cfg, "S", "Chapter 1")

    assert (view.series, view.chapter) == ("S", "Chapter 1")
    assert (view.strip_width, view.strip_height) == (800, 3000)
    assert view.has_output

    assert [t.kind for t in view.raw] == ["image", "image", "image"]
    assert [(t.y0, t.y1) for t in view.raw] == [(0, 1000), (1000, 2500), (2500, 3000)]
    assert [t.label for t in view.raw] == ["page_000.png", "page_001.png", "page_002.png"]
    assert all(t.path is not None and t.path.is_file() for t in view.raw)
    assert not any("page_003" in t.label for t in view.raw)  # filtered file skipped

    assert [t.kind for t in view.output] == ["image", "image", "filtered", "image"]
    assert [(t.y0, t.y1) for t in view.output] == [
        (0, 1200),
        (1200, 2400),
        (2400, 2700),
        (2700, 3000),
    ]
    assert [t.label for t in view.output] == [
        "slice_0000.jpg",
        "slice_0001.jpg",
        "filtered slice 2",
        "slice_0003.jpg",
    ]
    assert view.output[2].path is None
    assert all(t.path is not None and t.path.is_file() for t in view.output if t.kind == "image")


def test_missing_slice_kinds(cfg: Config) -> None:
    """A slice neither exported nor filtered is 'missing'; a vanished export file too."""
    _full_artifact_chapter(cfg)
    sp = cfg.paths
    work_dir = sp.work_root / "S" / "Chapter 1"
    slices = SlicesArtifact.load(work_dir / "slices.json")
    slices.slices[2].filtered = False  # now neither filtered nor exported
    slices.slices[3].filtered = False
    slices.save(work_dir / "slices.json")
    (sp.output_root / "S" / "Chapter 1" / "slice_0003.jpg").unlink()  # export entry exists, file gone

    view = library.load_chapter_view(cfg, "S", "Chapter 1")
    assert [t.kind for t in view.output] == ["image", "image", "missing", "missing"]
    assert [t.label for t in view.output] == [
        "slice_0000.jpg",
        "slice_0001.jpg",
        "missing slice 2",
        "missing slice 3",
    ]
    assert view.has_output


# ---------------------------------------------------------------------- fallbacks


def test_fallback_stacked_raw_images(cfg: Config, tmp_path: Path) -> None:
    raw_dir = tmp_path / "library" / "S" / "Chapter 1"
    _write_png(raw_dir / "1.png", (800, 1000), (255, 0, 0))
    _write_png(raw_dir / "2.png", (800, 600), (0, 255, 0))
    _write_png(raw_dir / "3.png", (400, 400), (0, 0, 255))  # scaled x2 -> 800 rows

    view = library.load_chapter_view(cfg, "S", "Chapter 1")
    assert [(t.y0, t.y1) for t in view.raw] == [(0, 1000), (1000, 1600), (1600, 2400)]
    assert (view.strip_width, view.strip_height) == (800, 2400)
    assert not view.has_output
    assert view.output == ()


def test_fallback_output_images_scaled_to_strip_width(cfg: Config, tmp_path: Path) -> None:
    raw_dir = tmp_path / "library" / "S" / "Chapter 1"
    out_dir = tmp_path / "output" / "S" / "Chapter 1"
    _write_png(raw_dir / "1.png", (800, 1000), (255, 0, 0))
    _write_png(out_dir / "1.jpg", (800, 400), (0, 255, 0))  # already strip width
    _write_png(out_dir / "2.jpg", (400, 300), (0, 0, 255))  # scaled x2 -> 600 rows

    view = library.load_chapter_view(cfg, "S", "Chapter 1")
    assert [(t.y0, t.y1) for t in view.output] == [(0, 400), (400, 1000)]
    assert (view.strip_width, view.strip_height) == (800, 1000)  # raw fallback wins
    assert view.has_output


def test_corrupt_ingest_behaves_like_missing(cfg: Config, tmp_path: Path) -> None:
    raw_dir = tmp_path / "library" / "S" / "Chapter 1"
    work_dir = tmp_path / "work" / "S" / "Chapter 1"
    _write_png(raw_dir / "1.png", (800, 1000), (255, 0, 0))
    work_dir.mkdir(parents=True)
    (work_dir / "ingest.json").write_text("{not json", encoding="utf-8")

    view = library.load_chapter_view(cfg, "S", "Chapter 1")
    assert [(t.y0, t.y1) for t in view.raw] == [(0, 1000)]
    assert (view.strip_width, view.strip_height) == (800, 1000)


def test_neither_dir_raises(cfg: Config) -> None:
    with pytest.raises(FileNotFoundError):
        library.load_chapter_view(cfg, "S", "Chapter 1")
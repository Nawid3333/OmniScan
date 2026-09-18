"""Tests for ingest_chapter end-to-end (fixtures written into tmp_path)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from tests.fixtures import images

from omniscan.core.paths import list_images
from omniscan.core.schemas import IngestArtifact
from omniscan.ingest import ingest_chapter


def test_uniform_chapter_layout(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    images.plain_jpeg(raw / "001.jpg", size=(400, 300))
    images.plain_jpeg(raw / "002.jpg", size=(400, 320))
    images.plain_jpeg(raw / "003.jpg", size=(400, 280))

    result = ingest_chapter(raw, "Series", "Chapter 1", tmp_path / "cache")
    art = result.artifact
    assert art.strip_width == 400
    assert all(f.converted_from is None for f in art.files)
    assert [f.y0 for f in art.files] == [0, 300, 620]
    assert [f.y1 for f in art.files] == [300, 620, 900]
    assert art.strip_height == 900
    assert all(f.scale == 1.0 for f in art.files)


def test_mixed_chapter_marks_converted_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    images.plain_jpeg(raw / "page_1.jpg", size=(400, 300))
    images.png_with_alpha(raw / "page_2.png", size=(400, 300))
    images.rotated_jpeg(raw / "page_3.jpg", size=(300, 400))

    result = ingest_chapter(raw, "Series", "Chapter 1", tmp_path / "cache")
    by_name = {f.name: f for f in result.artifact.files}
    assert by_name["page_1.jpg"].converted_from is None
    assert by_name["page_2.png"].converted_from == ".png"
    assert by_name["page_3.jpg"].converted_from == ".jpg"
    assert result.artifact.strip_width == 400


def test_file_order_matches_list_images_and_index_sequential(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    images.plain_jpeg(raw / "10.jpg", size=(400, 300))
    images.plain_jpeg(raw / "2.jpg", size=(400, 300))
    images.plain_jpeg(raw / "1.jpg", size=(400, 300))

    result = ingest_chapter(raw, "Series", "Chapter 1", tmp_path / "cache")
    expected = [p.name for p in list_images(raw)]
    assert [f.name for f in result.artifact.files] == expected
    assert [f.name for f in result.artifact.files] == ["1.jpg", "2.jpg", "10.jpg"]
    assert [f.index for f in result.artifact.files] == [0, 1, 2]


def test_artifact_roundtrip(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    images.plain_jpeg(raw / "001.jpg", size=(400, 300))
    images.png_with_alpha(raw / "002.png", size=(400, 300))

    result = ingest_chapter(raw, "Series", "Chapter 1", tmp_path / "cache")
    dest = tmp_path / "ingest.json"
    result.artifact.save(dest)
    assert IngestArtifact.load(dest) == result.artifact


def test_empty_chapter_raises(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(ValueError):
        ingest_chapter(raw, "Series", "Chapter 1", tmp_path / "cache")


def test_converted_paths_exist_and_are_jpeg(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    images.plain_jpeg(raw / "001.jpg", size=(400, 300))
    images.webp_image(raw / "002.webp", size=(400, 300))

    result = ingest_chapter(raw, "Series", "Chapter 1", tmp_path / "cache")
    assert len(result.converted_paths) == 2
    for path in result.converted_paths:
        assert path.exists()
        with Image.open(path) as img:
            assert img.format == "JPEG"
            img.verify()

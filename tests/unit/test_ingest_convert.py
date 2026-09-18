"""Tests for convert.py: needs_conversion / convert_to_jpeg."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from tests.fixtures import images

from omniscan.core.manifest import hash_file
from omniscan.ingest.convert import convert_to_jpeg, needs_conversion


def test_plain_jpeg_needs_no_conversion(tmp_path: Path) -> None:
    src = images.plain_jpeg(tmp_path / "page.jpg")
    assert src.exists()
    assert needs_conversion(src) is False
    out = convert_to_jpeg(src, tmp_path / "cache", 0)
    assert out.converted is False
    assert out.jpeg_path == src
    assert out.sha256 == hash_file(src)


def test_rotated_jpeg_is_upright_after_conversion(tmp_path: Path) -> None:
    src = images.rotated_jpeg(tmp_path / "page.jpg", size=(300, 400))
    assert needs_conversion(src) is True
    cache = tmp_path / "cache"
    out = convert_to_jpeg(src, cache, 3)
    assert out.converted is True
    assert out.jpeg_path != src
    assert out.jpeg_path.parent == cache
    assert (out.width, out.height) == (300, 400)
    with Image.open(out.jpeg_path) as img:
        assert img.getexif().get(0x0112, 1) == 1
        assert img.size == (300, 400)


def test_png_with_alpha_flattens_to_white(tmp_path: Path) -> None:
    src = images.png_with_alpha(tmp_path / "page.png")
    assert needs_conversion(src) is True
    out = convert_to_jpeg(src, tmp_path / "cache", 0)
    assert out.converted is True
    with Image.open(out.jpeg_path) as img:
        assert img.mode == "RGB"
        assert img.getpixel((0, 0)) == (255, 255, 255)


def test_cmyk_and_grayscale_convert_to_rgb(tmp_path: Path) -> None:
    for name, fn in (("cmyk", images.cmyk_jpeg), ("gray", images.grayscale_jpeg)):
        src = fn(tmp_path / f"{name}.jpg")
        assert needs_conversion(src) is True, name
        out = convert_to_jpeg(src, tmp_path / "cache", 0)
        assert out.converted is True, name
        with Image.open(out.jpeg_path) as img:
            assert img.mode == "RGB", name


def test_webp_suffix_triggers_conversion(tmp_path: Path) -> None:
    src = images.webp_image(tmp_path / "page.webp")
    assert needs_conversion(src) is True
    out = convert_to_jpeg(src, tmp_path / "cache", 0)
    assert out.converted is True
    assert (out.width, out.height) == (250, 180)


def test_rerun_after_source_change_writes_new_bytes(tmp_path: Path) -> None:
    src = images.rotated_jpeg(tmp_path / "page.jpg", color=(60, 200, 60))
    cache = tmp_path / "cache"
    first = convert_to_jpeg(src, cache, 0)
    images.rotated_jpeg(src, color=(200, 60, 60))
    second = convert_to_jpeg(src, cache, 0)
    assert second.jpeg_path == first.jpeg_path
    assert second.sha256 != first.sha256

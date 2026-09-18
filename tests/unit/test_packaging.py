"""Tests for omniscan.packaging and the `omniscan pack` CLI command."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile

import pytest
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, PathsConfig
from omniscan.packaging import pack_cbz, pack_pdf, safe_filename
from tests.fixtures.images import grayscale_jpeg, plain_jpeg, png_with_alpha

runner = CliRunner()


def _jpeg(path, size=(100, 80), color=(10, 200, 30)):
    """Write a distinct single-colour JPEG; the colour encodes the file's identity."""
    return plain_jpeg(path, size=size, color=color)


def _comic_info(cbz_path):
    with zipfile.ZipFile(cbz_path) as zf:
        return ET.fromstring(zf.read("ComicInfo.xml"))


def test_safe_filename_replaces_forbidden_characters() -> None:
    assert safe_filename('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"


def test_safe_filename_strips_spaces_and_dots() -> None:
    assert safe_filename("  .name. ") == "name"
    assert safe_filename("...") == "_"


def test_safe_filename_keeps_unicode() -> None:
    assert safe_filename("한국어 - Chapter 1") == "한국어 - Chapter 1"


def test_pack_cbz_entry_names_order_and_stored(tmp_path) -> None:
    images = [_jpeg(tmp_path / f"p{i}.jpg", color=(i * 40, 0, 0)) for i in range(3)]
    dest = tmp_path / "out.cbz"
    pack_cbz(images, dest)
    with zipfile.ZipFile(dest) as zf:
        assert zf.namelist() == ["0001.jpg", "0002.jpg", "0003.jpg", "ComicInfo.xml"]
        for image, name in zip(images, ("0001.jpg", "0002.jpg", "0003.jpg"), strict=True):
            assert zf.read(name) == image.read_bytes()
        assert all(info.compress_type == zipfile.ZIP_STORED for info in zf.infolist())
        assert zf.testzip() is None


def test_pack_cbz_comic_info_with_metadata(tmp_path) -> None:
    images = [_jpeg(tmp_path / f"p{i}.jpg") for i in range(3)]
    dest = tmp_path / "out.cbz"
    pack_cbz(images, dest, title="Chapter 1", series="S", number="1")
    root = _comic_info(dest)
    assert [child.tag for child in root] == ["Title", "Series", "Number", "PageCount", "LanguageISO"]
    page_count = root.find("PageCount")
    language = root.find("LanguageISO")
    assert page_count is not None and language is not None
    assert page_count.text == "3"
    assert language.text == "en"


def test_pack_cbz_comic_info_without_metadata(tmp_path) -> None:
    dest = tmp_path / "out.cbz"
    pack_cbz([_jpeg(tmp_path / "p.jpg")], dest)
    root = _comic_info(dest)
    assert [child.tag for child in root] == ["PageCount", "LanguageISO"]


def test_pack_cbz_preserves_order_and_lowercases_suffix(tmp_path) -> None:
    b = _jpeg(tmp_path / "b.jpg", color=(1, 2, 3))
    a = _jpeg(tmp_path / "X.JPG", color=(4, 5, 6))
    dest = tmp_path / "out.cbz"
    pack_cbz([b, a], dest)
    with zipfile.ZipFile(dest) as zf:
        assert zf.namelist()[:2] == ["0001.jpg", "0002.jpg"]
        assert zf.read("0001.jpg") == b.read_bytes()
        assert zf.read("0002.jpg") == a.read_bytes()


def test_pack_cbz_empty_raises_and_creates_nothing(tmp_path) -> None:
    dest = tmp_path / "out.cbz"
    with pytest.raises(ValueError, match="no images to pack"):
        pack_cbz([], dest)
    assert not dest.exists()


def test_pack_cbz_missing_input_leaves_no_tmp(tmp_path) -> None:
    good = _jpeg(tmp_path / "good.jpg")
    dest = tmp_path / "sub" / "out.cbz"
    with pytest.raises(FileNotFoundError):
        pack_cbz([good, tmp_path / "missing.jpg"], dest)
    assert not dest.exists()
    assert not dest.with_suffix(".cbz.tmp").exists()


def test_pack_cbz_twice_replaces_without_tmp(tmp_path) -> None:
    images = [_jpeg(tmp_path / "p.jpg")]
    dest = tmp_path / "out.cbz"
    pack_cbz(images, dest)
    pack_cbz(images, dest)
    assert dest.is_file()
    assert not dest.with_suffix(".cbz.tmp").exists()


def test_pack_pdf_page_count(tmp_path) -> None:
    images = [
        _jpeg(tmp_path / "a.jpg", size=(100, 80)),
        _jpeg(tmp_path / "b.jpg", size=(120, 60)),
        _jpeg(tmp_path / "c.jpg", size=(80, 140)),
    ]
    dest = tmp_path / "out.pdf"
    pack_pdf(images, dest)
    data = dest.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page(?![s\w])", data)) == 3


def test_pack_pdf_rgba_png_and_grayscale_jpeg(tmp_path) -> None:
    dest = tmp_path / "out.pdf"
    pack_pdf([png_with_alpha(tmp_path / "a.png"), grayscale_jpeg(tmp_path / "b.jpg")], dest)
    assert dest.read_bytes().startswith(b"%PDF")


def test_pack_pdf_empty_raises_and_creates_nothing(tmp_path) -> None:
    dest = tmp_path / "out.pdf"
    with pytest.raises(ValueError, match="no images to pack"):
        pack_pdf([], dest)
    assert not dest.exists()
    assert not dest.with_suffix(".pdf.tmp").exists()


def _config_with_output_root(tmp_path) -> Config:
    return Config(paths=PathsConfig(output_root=tmp_path / "output"))


def _make_output_tree(root) -> None:
    for files in {
        root / "S" / "Chapter 1": ("1.jpg", "2.jpg"),
        root / "S" / "Chapter 2": ("1.jpg",),
        root / "S" / "_filtered": ("x.jpg",),
    }.items():
        files[0].mkdir(parents=True)
        for name in files[1]:
            _jpeg(files[0] / name)


def test_cli_pack_default_all_chapters_cbz(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    _make_output_tree(output_root)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: _config_with_output_root(tmp_path))
    result = runner.invoke(app, ["pack", "S"])
    assert result.exit_code == 0
    packaged = output_root / "S" / "_packaged"
    assert sorted(p.name for p in packaged.iterdir()) == ["S - Chapter 1.cbz", "S - Chapter 2.cbz"]


def test_cli_pack_repeated_format_and_chapter(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    _make_output_tree(output_root)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: _config_with_output_root(tmp_path))
    result = runner.invoke(app, ["pack", "S", "-f", "pdf", "-f", "cbz", "-c", "Chapter 2"])
    assert result.exit_code == 0
    packaged = output_root / "S" / "_packaged"
    assert sorted(p.name for p in packaged.iterdir()) == ["S - Chapter 2.cbz", "S - Chapter 2.pdf"]


def test_cli_pack_no_output_folder_exits_2(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: _config_with_output_root(tmp_path))
    result = runner.invoke(app, ["pack", "S"])
    assert result.exit_code == 2
    assert "nothing to pack" in result.output


def test_cli_pack_number_formatting(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    for chapter in ("Chapter 12.5", "Chapter 3"):
        (output_root / "S" / chapter).mkdir(parents=True)
        _jpeg(output_root / "S" / chapter / "1.jpg")
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: _config_with_output_root(tmp_path))
    result = runner.invoke(app, ["pack", "S"])
    assert result.exit_code == 0
    numbers = {}
    for name in ("S - Chapter 12.5.cbz", "S - Chapter 3.cbz"):
        number = _comic_info(output_root / "S" / "_packaged" / name).find("Number")
        assert number is not None
        numbers[name] = number.text
    assert numbers == {"S - Chapter 12.5.cbz": "12.5", "S - Chapter 3.cbz": "3"}

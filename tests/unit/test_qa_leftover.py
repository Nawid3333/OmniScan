"""Tests for omniscan.qa.leftover — deciding which regions still show their original text (torch-free)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.core.config import Config, InpaintConfig, SfxConfig
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import BBox, ExportArtifact, ExportFile, QaArtifact, QaIssue, Region, RegionKind
from omniscan.export.segments import Segment
from omniscan.qa.leftover import (
    cleaned_kinds,
    leftover_issues,
    load_issues,
    regions_to_check,
    segment_rows,
    source_chars,
)


def region(rid: str, text: str, *, kind: RegionKind = "bubble_text", y0: int = 0) -> Region:
    return Region(id=rid, slice_index=0, kind=kind, bbox=BBox(x0=0, y0=y0, x1=100, y1=y0 + 40), text=text)


def test_source_chars_keeps_korean_japanese_and_chinese() -> None:
    assert source_chars("안녕, Jinwoo! ドン 漢字 ｱ") == "안녕ドン漢字ｱ"
    assert source_chars("WHERE ARE YOU GOING?!") == ""


def test_cleaned_kinds_follow_the_sfx_mode_and_the_watermark_switch() -> None:
    assert cleaned_kinds(Config()) == {"bubble_text", "free_text", "sfx", "watermark"}
    kept = Config(sfx=SfxConfig(mode="keep"), inpaint=InpaintConfig(remove_watermarks=False))
    assert cleaned_kinds(kept) == {"bubble_text", "free_text"}


def test_only_the_readable_original_text_is_an_issue() -> None:
    regions = [
        region("r0001", "여기가 어디지?"),
        region("r0002", "빨리 도망쳐!"),
        region("r0003", "쾅", kind="sfx"),
        region("r0004", "www.사이트.com", kind="watermark"),
        region("r0005", "가자"),
        region("r0006", "안녕하세요"),
    ]
    readings = {
        "r0001": ("여기가 어디지?", 0.93),  # the original text was never erased
        "r0002": ("のりけ", 0.9),  # English lettering misread as kana: not the original
        "r0003": ("쾅", 0.8),  # a one-character effect still there
        "r0004": ("www.사이트.com", 0.7),
        "r0005": ("가자", 0.2),  # too unsure
        # r0006: not read (nothing found there)
    }
    issues = leftover_issues(regions, readings)
    assert [(i.region_id, i.kind, i.read) for i in issues] == [
        ("r0001", "source_left", "여기가 어디지?"),
        ("r0003", "source_left", "쾅"),
        ("r0004", "watermark_left", "www.사이트.com"),
    ]
    assert issues[0].message == "the original text is still readable on the finished page: '여기가 어디지?'"
    assert issues[2].message.startswith("the watermark is still visible on the finished page")
    assert leftover_issues(regions, {"r0001": ("WHERE AM I?", 0.99)}) == []


def test_regions_to_check_are_cleaned_kinds_with_source_text_on_an_exported_page() -> None:
    regions = [
        region("r0001", "안녕", y0=10),
        region("r0002", "안녕", y0=700),  # in a filtered (not exported) slice
        region("r0003", "", y0=20),
        region("r0004", "HELLO", y0=30),  # nothing to look for
        region("r0005", "쾅", kind="sfx", y0=40),
    ]
    segments = [Segment(0, 600, 0)]
    assert [r.id for r in regions_to_check(regions, {"bubble_text"}, segments)] == ["r0001"]
    assert [r.id for r in regions_to_check(regions, {"bubble_text", "sfx"}, segments)] == ["r0001", "r0005"]


def export(*heights: int) -> ExportArtifact:
    files = [
        ExportFile(name=f"{i + 1:04d}.jpg", slice_index=i, width=100, height=h, bytes=1)
        for i, h in enumerate(heights)
    ]
    return ExportArtifact(quality=95, subsampling="444", files=files)


def test_segment_rows_pair_the_images_with_their_strip_rows() -> None:
    segments = [Segment(0, 600, 0), Segment(900, 1400, 2)]
    assert segment_rows(export(600, 500), segments) == [("0001.jpg", 0, 600), ("0002.jpg", 900, 1400)]
    with pytest.raises(ValueError, match="run export again"):
        segment_rows(export(600), segments)
    with pytest.raises(ValueError, match="run export again"):
        segment_rows(export(600, 400), segments)


def test_load_issues_reads_qa_json(tmp_path: Path) -> None:
    paths = ChapterPaths("S", "C", tmp_path / "r", tmp_path / "w", tmp_path / "o", tmp_path / "f")
    assert load_issues(paths) == []
    issue = QaIssue(region_id="r0001", kind="source_left", message="m", read="안녕")
    QaArtifact(checked=3, issues=[issue]).save(paths.artifact("qa.json"))
    assert load_issues(paths) == [issue]

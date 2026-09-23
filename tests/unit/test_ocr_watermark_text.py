"""Watermark text patterns: loading, matching and post-OCR region reclassification (card F2b)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.core.schemas import (
    BBox,
    IngestArtifact,
    Region,
    RegionKind,
    RegionsArtifact,
    SourceFile,
)
from omniscan.eval.score import score_chapter
from omniscan.eval.truth import TruthBox, TruthStats
from omniscan.ocr.watermark_text import (
    default_watermark_text_paths,
    load_watermark_patterns,
    matches_watermark_text,
    reclassify_watermark_regions,
)
from omniscan.translate.prompts import translatable
from tests.fixtures.korean_pages import KOREAN_LINES

AD_1 = '구글검색 "먹튀검증 스포위키"'  # found on Solo Leveling raw pages (card F2b)
AD_2 = "라이브스코어 스포츠중계 가상토토 전문가 정기/오목 웹툰"
SEED_PATTERNS = ("구글검색", "라이브스코어", "가상토토", "스포위키")


def write_patterns(path: Path, *patterns: str) -> Path:
    """A watermark_text.toml with `patterns` (created on demand)."""
    entries = ", ".join(f'"{pattern}"' for pattern in patterns)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"[watermark_text]\npatterns = [{entries}]\n", encoding="utf-8")
    return path


def region(kind: RegionKind, text: str, region_id: str = "r0001") -> Region:
    return Region(
        id=region_id,
        slice_index=0,
        kind=kind,
        bbox=BBox(x0=10, y0=50, x1=200, y1=90),
        text=text,
        confidence=0.95,
    )


# ---------------------------------------------------------------- matches_watermark_text


def test_the_real_ad_examples_match_their_patterns() -> None:
    assert matches_watermark_text(AD_1, list(SEED_PATTERNS))
    assert matches_watermark_text(AD_2, list(SEED_PATTERNS))
    assert matches_watermark_text(AD_1, ["구글검색", "스포위키"])
    assert matches_watermark_text(AD_2, ["라이브스코어", "가상토토"])


def test_matching_is_case_insensitive_and_a_substring_not_an_exact_match() -> None:
    assert matches_watermark_text("LiveScore Sports Broadcast Virtual Toto", ["livescore"])
    assert matches_watermark_text("LIVESCORE", ["LiveScore"])
    assert matches_watermark_text("라이브스코어", ["라이브스코어"])  # an exact text is its own substring
    assert not matches_watermark_text(
        "구글 검색 어디", ["구글검색"]
    )  # spacing differences survive the collapse


def test_a_pattern_spanning_a_line_break_in_the_ocr_text_matches() -> None:
    assert matches_watermark_text("먹튀검증\n스포위키 광고", ["먹튀검증 스포위키"])
    assert matches_watermark_text('구글검색\n"먹튀검증 스포위키"', ["스포위키"])


def test_empty_patterns_never_match() -> None:
    assert not matches_watermark_text(AD_1, [])
    assert not matches_watermark_text("", ["구글검색"])
    assert not matches_watermark_text("", [])


def test_ordinary_dialogue_never_matches_the_shipped_patterns() -> None:
    """Regression proof on the project's real Korean dialogue lines, not just a synthetic string."""
    shipped = load_watermark_patterns([default_watermark_text_paths()[0]])
    assert shipped == list(SEED_PATTERNS)
    for line in KOREAN_LINES:
        assert not matches_watermark_text(line, shipped), line


# ---------------------------------------------------------------- load_watermark_patterns


def test_the_shipped_file_loads_the_seed_patterns_in_order() -> None:
    shipped = default_watermark_text_paths()[0]
    assert shipped.name == "watermark_text.toml"
    assert load_watermark_patterns([shipped]) == list(SEED_PATTERNS)


def test_a_user_file_adds_to_the_shipped_patterns(tmp_path: Path) -> None:
    shipped = default_watermark_text_paths()[0]
    user = write_patterns(tmp_path / "watermark_text.toml", "디시인사이드", "구글검색")
    # both files contribute, order preserved, the duplicate is kept
    assert load_watermark_patterns([shipped, user]) == [*SEED_PATTERNS, "디시인사이드", "구글검색"]


def test_a_missing_file_in_the_list_is_skipped(tmp_path: Path) -> None:
    user = write_patterns(tmp_path / "watermark_text.toml", "스포위키")
    paths = [tmp_path / "missing.toml", user, tmp_path / "also-missing.toml"]
    assert load_watermark_patterns(paths) == ["스포위키"]


def test_an_empty_or_blank_pattern_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        load_watermark_patterns([write_patterns(tmp_path / "a.toml", "")])
    with pytest.raises(ValueError, match="empty"):
        load_watermark_patterns([write_patterns(tmp_path / "b.toml", "구글검색", "   ")])


def test_invalid_toml_raises_with_the_file_path(tmp_path: Path) -> None:
    path = tmp_path / "watermark_text.toml"
    path.write_text("[watermark_text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid TOML"):
        load_watermark_patterns([path])


def test_a_non_string_patterns_entry_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "watermark_text.toml"
    path.write_text("[watermark_text]\npatterns = [1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="list of strings"):
        load_watermark_patterns([path])


# ---------------------------------------------------------------- reclassify_watermark_regions


def test_matching_regions_become_watermarks_and_others_pass_through() -> None:
    ad_1 = region("bubble_text", AD_1, "r0001")
    ad_2 = region("free_text", AD_2, "r0002")
    line = region("bubble_text", KOREAN_LINES[0], "r0003")
    sfx = region("sfx", AD_2, "r0004")  # an sfx whose text happens to match stays sfx
    already = region("watermark", AD_1, "r0005")
    watermark = region("watermark", KOREAN_LINES[1], "r0006")  # stays a watermark either way

    out = reclassify_watermark_regions([ad_1, ad_2, line, sfx, already, watermark], list(SEED_PATTERNS))

    assert [r.kind for r in out] == [
        "watermark",
        "watermark",
        "bubble_text",
        "sfx",
        "watermark",
        "watermark",
    ]
    assert out[2] is line and out[3] is sfx and out[4] is already and out[5] is watermark
    assert out[0] is not ad_1 and out[1] is not ad_2  # reclassified regions are new copies


def test_only_the_kind_of_a_matched_region_changes() -> None:
    ad = region("bubble_text", AD_2, "r0007")
    (watermark,) = reclassify_watermark_regions([ad], list(SEED_PATTERNS))
    assert watermark.id == "r0007"
    assert watermark.text == AD_2
    assert watermark.bbox == ad.bbox
    assert watermark.confidence == 0.95
    assert watermark.slice_index == 0


def test_without_patterns_every_region_passes_through_by_identity() -> None:
    regions = [region("bubble_text", AD_1), region("free_text", KOREAN_LINES[0], "r0002")]
    out = reclassify_watermark_regions(regions, [])
    assert out[0] is regions[0] and out[1] is regions[1]


# ---------------------------------------------------------------- downstream (no new code needed)


def test_a_reclassified_region_is_excluded_from_translation() -> None:
    ad = region("bubble_text", AD_1, "r0001")
    line = region("bubble_text", KOREAN_LINES[0], "r0002")
    (watermark, kept) = reclassify_watermark_regions([ad, line], list(SEED_PATTERNS))
    assert watermark.kind == "watermark"
    assert [r.id for r in translatable([watermark, kept])] == ["r0002"]


def test_a_reclassified_region_is_excluded_from_scoring() -> None:
    """A watermark region inside a truth box counts neither as detected nor as a predicted region."""
    ingest = IngestArtifact(
        series="S",
        chapter="Chapter 1",
        strip_width=400,
        strip_height=600,
        files=[SourceFile(index=0, name="001.jpg", sha256="0" * 64, width=400, height=600, y0=0, y1=600)],
    )
    truth = TruthBox(page=1, bbox=BBox(x0=0, y0=0, x1=400, y1=600), lines=(AD_2,))
    (watermark,) = reclassify_watermark_regions([region("bubble_text", AD_2, "r0001")], ["라이브스코어"])
    report = score_chapter(
        "S",
        "Chapter 1",
        ingest,
        RegionsArtifact(regions=[watermark]),
        None,
        [truth],
        {},
        TruthStats(pages=1, pages_without_truth=0, dropped=0),
    )
    assert report.regions == 0  # the watermark region is not a predicted region
    assert report.detected_boxes == 0 and len(report.missed) == 1
    assert report.precision is None

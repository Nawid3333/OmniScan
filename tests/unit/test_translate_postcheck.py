"""Tests for omniscan.translate.postcheck (locked-term violations in the English output)."""

from __future__ import annotations

from typing import Literal

import pytest

from omniscan.core.schemas import BBox, GlossaryEntry, Region, TermType
from omniscan.translate.postcheck import (
    TermViolation,
    check_locked_terms,
    check_regions,
    normalize_for_check,
)


def region(
    rid: str,
    *,
    kind: Literal["bubble_text", "free_text", "sfx", "watermark"] = "bubble_text",
    text: str = "",
    slice_index: int = 0,
    reading_order: int = 0,
    lang: str = "ko",
) -> Region:
    return Region(
        id=rid,
        slice_index=slice_index,
        kind=kind,
        bbox=BBox(x0=0, y0=0, x1=10, y1=10),
        reading_order=reading_order,
        text=text,
        lang=lang,  # type: ignore[arg-type]
    )


def entry(
    eid: int | None,
    source: str,
    target: str,
    *,
    status: Literal["proposed", "locked", "rejected"] = "locked",
    aliases: list[str] | None = None,
    type_: TermType = "person",
) -> GlossaryEntry:
    return GlossaryEntry(
        id=eid, source=source, target=target, status=status, aliases=aliases or [], type=type_
    )


LOCKED = [entry(1, "성진", "Seong-jin"), entry(2, "게이트", "Gate", type_="place")]
SOURCE = "성진이가 게이트에 들어간 지"


def test_normalize_for_check_nfkc_casefold_only() -> None:
    assert normalize_for_check("Seong-Jin!") == "seong-jin!"
    assert normalize_for_check("Ｇａｔｅ") == "gate"
    assert normalize_for_check("성진") == "성진"
    assert normalize_for_check("a  b\tc") == "a  b\tc"  # whitespace untouched


def test_satisfied_target_no_violations() -> None:
    assert check_locked_terms(SOURCE, "Seong-jin went into the gate", LOCKED, lang="ko") == []


def test_missing_terms_in_source_order() -> None:
    assert check_locked_terms(SOURCE, "He went into the portal", LOCKED, lang="ko") == [
        TermViolation(1, "성진", "Seong-jin"),
        TermViolation(2, "게이트", "Gate"),
    ]


def test_only_second_term_missing() -> None:
    assert check_locked_terms(SOURCE, "Seong-jin went into the portal", LOCKED, lang="ko") == [
        TermViolation(2, "게이트", "Gate")
    ]


def test_proposed_and_rejected_ignored_and_no_locked_entries() -> None:
    entries = [
        entry(3, "성진", "Seong-jin", status="proposed"),
        entry(4, "게이트", "Gate", status="rejected"),
    ]
    assert check_locked_terms(SOURCE, "nothing applies", entries, lang="ko") == []
    assert check_locked_terms(SOURCE, "nothing applies", [], lang="ko") == []


def test_alias_hit_reports_entry_source() -> None:
    seongjin = entry(3, "성진", "Seong-jin", aliases=["진우"])
    assert check_locked_terms("진우가 왔다", "He came", [seongjin], lang="ko") == [
        TermViolation(3, "성진", "Seong-jin")
    ]


def test_duplicate_occurrence_one_violation_and_no_hit_even_with_empty_target() -> None:
    one = [entry(1, "성진", "Seong-jin")]
    assert check_locked_terms("성진이 성진을 불렀다", "", one, lang="ko") == [
        TermViolation(1, "성진", "Seong-jin")
    ]
    assert check_locked_terms("아무 용어 없는 문장", "", one, lang="ko") == []


def test_target_check_is_case_insensitive_and_nfkc() -> None:
    assert check_locked_terms("성진!", "SEONG-JIN!", [entry(1, "성진", "Seong-jin")], lang="ko") == []
    assert (
        check_locked_terms("게이트", "Ｇａｔｅ", [entry(2, "게이트", "Gate", type_="place")], lang="ko") == []
    )


def test_locked_entry_without_id_raises_on_hit_only() -> None:
    no_id = entry(None, "성진", "Seong-jin")
    with pytest.raises(ValueError, match="id"):
        check_locked_terms("성진이 갔다", "He left", [no_id], lang="ko")
    assert check_locked_terms("아무도 오지 않았다", "Nobody came", [no_id], lang="ko") == []


def test_check_regions_skips_watermark_and_missing_text() -> None:
    regions = [
        region("r0001", text="성진이가 왔다"),
        region("r0002", text="게이트"),
        region("r0003", kind="watermark", text="성진"),
        region("r0004", text="안녕"),
    ]
    texts = {"r0001": "He came", "r0002": "The gate", "r0003": "x"}
    assert check_regions(regions, texts, LOCKED) == {"r0001": [TermViolation(1, "성진", "Seong-jin")]}


def test_check_regions_follows_translatable_order_r10_after_r2() -> None:
    regions = [region("r10", text="성진이가 갔다"), region("r2", text="게이트가 열렸다")]
    texts = {"r10": "Someone left", "r2": "Opened"}
    result = check_regions(regions, texts, LOCKED)
    assert list(result) == ["r2", "r10"]
    assert result == {
        "r2": [TermViolation(2, "게이트", "Gate")],
        "r10": [TermViolation(1, "성진", "Seong-jin")],
    }


def test_check_locked_terms_japanese_particle() -> None:
    yujin = entry(5, "ユジン", "Yujin")
    assert check_locked_terms("ユジンは来た", "Yujin came", [yujin], lang="ja") == []
    assert check_locked_terms("ユジンは来た", "She came", [yujin], lang="ja") == [
        TermViolation(5, "ユジン", "Yujin")
    ]


def test_check_regions_japanese_region_passes_its_lang() -> None:
    regions = [region("r0001", text="ユジンは来た", lang="ja")]
    assert check_regions(regions, {"r0001": "She came"}, [entry(1, "ユジン", "Yujin")]) == {
        "r0001": [TermViolation(1, "ユジン", "Yujin")]
    }

"""Tests for omniscan.glossary.match (particle-stripping term matcher)."""

from __future__ import annotations

import pytest

from omniscan.core.schemas import GlossaryEntry
from omniscan.glossary.match import Match, find_terms, term_present


def entry(**overrides: object) -> GlossaryEntry:
    """A GlossaryEntry with a store-style id plus the given field overrides."""
    fields: dict[str, object] = {"id": 1, "source": "지훈", "target": "Jihoon"}
    fields.update(overrides)
    return GlossaryEntry(**fields)  # type: ignore[arg-type]


def test_term_and_particle_offsets() -> None:
    matches = find_terms(
        "지훈이 학교에 갔다", [entry(), entry(id=2, source="학교", target="school")], lang="ko"
    )
    assert matches == [
        Match(entry_id=1, source="지훈", start=0, end=3, particle="이"),
        Match(entry_id=2, source="학교", start=4, end=7, particle="에"),
    ]


def test_glued_pair_particle_taken_before_component() -> None:
    matches = find_terms("지훈이가 싫다고 했다", [entry()], lang="ko")
    assert matches == [Match(entry_id=1, source="지훈", start=0, end=4, particle="이가")]


def test_no_particle_before_space_or_end_of_string() -> None:
    assert find_terms("지훈 갔다", [entry()], lang="ko") == [
        Match(entry_id=1, source="지훈", start=0, end=2, particle=None)
    ]
    assert find_terms("지훈", [entry()], lang="ko") == [
        Match(entry_id=1, source="지훈", start=0, end=2, particle=None)
    ]


def test_alias_matches_like_source() -> None:
    matches = find_terms("훈이형이 왔다", [entry(aliases=["훈이형"])], lang="ko")
    assert matches == [Match(entry_id=1, source="훈이형", start=0, end=4, particle="이")]


def test_longer_candidate_wins_at_same_position() -> None:
    matches = find_terms(
        "학교에 갔다",
        [entry(id=1, source="학", target="study"), entry(id=2, source="학교", target="school")],
        lang="ko",
    )
    assert matches == [Match(entry_id=2, source="학교", start=0, end=3, particle="에")]


def test_entry_order_breaks_length_ties() -> None:
    matches = find_terms(
        "학교에 갔다",
        [entry(id=7, source="학교", target="first"), entry(id=8, source="학교", target="second")],
        lang="ko",
    )
    assert [m.entry_id for m in matches] == [7]


def test_matches_do_not_overlap_and_scan_resumes_after_end() -> None:
    matches = find_terms("지훈이지훈", [entry()], lang="ko")
    assert matches == [
        Match(entry_id=1, source="지훈", start=0, end=3, particle="이"),
        Match(entry_id=1, source="지훈", start=3, end=5, particle=None),
    ]


def test_no_matches_returns_empty_list() -> None:
    assert (
        find_terms("아무관련없는문장", [entry(), entry(id=2, source="학교", target="school")], lang="ko")
        == []
    )


def test_find_terms_requires_entry_id() -> None:
    with pytest.raises(ValueError, match="id"):
        find_terms("지훈이 갔다", [GlossaryEntry(source="지훈", target="Jihoon")], lang="ko")


def test_term_present_is_plain_substring_check() -> None:
    assert term_present("Jihoon went home", entry())
    assert term_present("Jihoon went home", entry(target="Ji"))
    assert not term_present("He went home", entry())
    assert not term_present("jihoon went home", entry())  # case-sensitive


def test_japanese_simple_particle() -> None:
    matches = find_terms("ユジンは学校に行った", [entry(source="ユジン", target="Yujin")], lang="ja")
    assert matches == [Match(entry_id=1, source="ユジン", start=0, end=4, particle="は")]


def test_japanese_compound_particle_taken_before_component() -> None:
    # とは must be consumed as one particle, not stop at its component と.
    matches = find_terms("ユジンとは違う", [entry(source="ユジン", target="Yujin")], lang="ja")
    assert matches == [Match(entry_id=1, source="ユジン", start=0, end=5, particle="とは")]


def test_no_particle_before_space_or_end_of_string_ja() -> None:
    je = entry(source="ユジン", target="Yujin")
    assert find_terms("ユジン 行った", [je], lang="ja") == [
        Match(entry_id=1, source="ユジン", start=0, end=3, particle=None)
    ]
    assert find_terms("ユジン", [je], lang="ja") == [
        Match(entry_id=1, source="ユジン", start=0, end=3, particle=None)
    ]


def test_chinese_strips_no_particles() -> None:
    # The particle-looking は right after the term is not consumed: zh has no particle set at all.
    assert find_terms("ユジンは行った", [entry(source="ユジン", target="Yujin")], lang="zh") == [
        Match(entry_id=1, source="ユジン", start=0, end=3, particle=None)
    ]


def test_english_strips_no_particles() -> None:
    assert find_terms("지훈이 갔다", [entry()], lang="en") == [
        Match(entry_id=1, source="지훈", start=0, end=2, particle=None)
    ]


def test_unknown_lang_strips_no_particles_and_does_not_raise() -> None:
    assert find_terms("지훈이 갔다", [entry()], lang="xx") == [
        Match(entry_id=1, source="지훈", start=0, end=2, particle=None)
    ]

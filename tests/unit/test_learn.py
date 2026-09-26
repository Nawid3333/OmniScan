"""Tests for omniscan.learn — corrections harvested from edits.json, memory.json, and the lessons applied."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from omniscan.core.config import LearnConfig
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import (
    BBox,
    ChapterEdits,
    FinalArtifact,
    FinalLine,
    LearnedRule,
    LearnKind,
    MemoryEntry,
    Region,
    RegionEdit,
    RegionKind,
    RegionsArtifact,
    SeriesMemory,
    Slice,
    SlicesArtifact,
)
from omniscan.edits import store
from omniscan.learn.apply import apply_to_regions, fix_words, similarity, translation_hints
from omniscan.learn.harvest import Corrections, harvest_chapter, harvest_series
from omniscan.learn.memory import (
    build_memory,
    current_memory,
    is_active,
    memory_path,
    rule_id,
    set_rule_enabled,
    word_changes,
)

CFG = LearnConfig()


def box(x0: int, y0: int, x1: int, y1: int) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def region(rid: str, text: str, y0: int = 0, *, kind: RegionKind = "bubble_text") -> Region:
    return Region(id=rid, slice_index=0, kind=kind, bbox=box(10, y0, 110, y0 + 50), text=text, confidence=0.9)


def rule(
    kind: LearnKind, wrong: str, right: str = "", count: int = 2, *, enabled: bool = True
) -> LearnedRule:
    return LearnedRule(
        id=rule_id(kind, wrong, right), kind=kind, wrong=wrong, right=right, count=count, enabled=enabled
    )


@pytest.fixture
def series(tmp_path: Path) -> SeriesPaths:
    return SeriesPaths(
        series="S",
        library_dir=tmp_path / "lib" / "S",
        work_dir=tmp_path / "work" / "S",
        output_dir=tmp_path / "out",
    )


def make_chapter(
    series: SeriesPaths, name: str, regions: list[Region], english: dict[str, str]
) -> ChapterPaths:
    """A processed chapter: raw folder, slices.json, ocr.json and a judge's final.json."""
    (series.library_dir / name).mkdir(parents=True)
    paths = series.chapter(name)
    SlicesArtifact(strip_width=200, strip_height=900, bands=[], slices=[Slice(index=0, y0=0, y1=900)]).save(
        paths.artifact("slices.json")
    )
    RegionsArtifact(regions=regions).save(paths.artifact("ocr.json"))
    lines = [FinalLine(region_id=rid, text=text, decision="pick") for rid, text in english.items()]
    FinalArtifact(judge_model="judge", lines=lines).save(paths.artifact("final.json"))
    return paths


# ---------------------------------------------------------------- building rules


def test_word_changes_counts_word_for_word_replacements() -> None:
    pairs = [
        ("Jlnwoo, run!", "Jinwoo, run!"),
        ("the Hunter guild", "the Hunters' Guild"),
        ("I am here", "I'm here"),  # two words became one: no word-for-word change
        ("so", "so what"),  # an insertion
    ]
    assert word_changes(pairs) == {("Jlnwoo", "Jinwoo"): 1, ("Hunter", "Hunters"): 1, ("guild", "Guild"): 1}
    assert word_changes(pairs, capitalised=True) == {("Jlnwoo", "Jinwoo"): 1, ("Hunter", "Hunters"): 1}


def test_build_memory_needs_evidence_and_keeps_the_switches() -> None:
    found = Corrections(
        ocr_fixes=[
            ("Jlnwoo, run!", "Jinwoo, run!"),
            ("Jlnwoo?", "Jinwoo?"),
            ("Hey Jlnwoo", "Hey Jinwoo"),
            ("Sung Jinwoo", "Sung Jin-woo"),  # once, while "Jinwoo" was kept three times: no rule
            ("the cat", "the cot"),
            ("a cat", "a cot"),
            ("my cat", "my cut"),  # "cat" -> "cot" wins two to one
        ],
        deleted=["www.site.com", "www.site.com", "x"],  # a one-character text is never a drop rule
        kinds=[("ⓒ STUDIO", "watermark"), ("쾅", "sfx")],
        translations=[
            ("안녕", "Hi", False, "1"),
            ("안녕", "Hello", True, "2"),
            ("가자", "Let's go", True, "2"),
        ],
        english_fixes=[
            ("The Hunter Association", "The Hunters Guild"),
            ("see you", "see ya"),
        ],  # lower case: no term
    )
    previous = SeriesMemory(rules=[rule("ocr_fix", "Jlnwoo", "Jinwoo", 3, enabled=False)])
    memory = build_memory(found, previous)
    assert [(r.kind, r.wrong, r.right, r.count, r.enabled) for r in memory.rules] == [
        ("ocr_fix", "Jlnwoo", "Jinwoo", 3, False),
        ("ocr_fix", "cat", "cot", 2, True),
        ("preferred_term", "Hunter", "Hunters", 1, True),
        ("preferred_term", "Association", "Guild", 1, True),
        ("drop_text", "www.site.com", "", 2, True),
        ("watermark_text", "ⓒ STUDIO", "", 1, True),
        ("sfx_text", "쾅", "", 1, True),
    ]
    assert memory.translations == [
        MemoryEntry(source="안녕", english="Hello", count=2, typed=True, chapter="2"),
        MemoryEntry(source="가자", english="Let's go", count=1, typed=True, chapter="2"),
    ]
    assert build_memory(Corrections()) == SeriesMemory()


def test_a_rule_acts_with_enough_evidence_while_switched_on() -> None:
    assert is_active(rule("ocr_fix", "a", "b", 2), CFG)
    assert not is_active(rule("ocr_fix", "a", "b", 1), CFG)
    assert is_active(rule("ocr_fix", "a", "b", 1), LearnConfig(min_count=1))
    assert is_active(rule("watermark_text", "ⓒ STUDIO", count=1), CFG)  # one label is enough
    assert not is_active(rule("drop_text", "www.site.com", count=5, enabled=False), CFG)


# ---------------------------------------------------------------- applying the lessons


def test_fix_words_replaces_whole_words_only() -> None:
    fixes = {"Jlnwoo": "Jinwoo"}
    assert fix_words("(Jlnwoo)  run,\nJlnwoo!", fixes) == "(Jinwoo)  run,\nJinwoo!"
    assert fix_words("Jlnwoo's xJlnwoo", fixes) == "Jlnwoo's xJlnwoo"
    assert fix_words("Jlnwoo", {}) == "Jlnwoo"


def test_apply_to_regions_fixes_drops_and_relabels() -> None:
    memory = SeriesMemory(
        rules=[
            rule("ocr_fix", "Jlnwoo", "Jinwoo"),
            rule("ocr_fix", "rn", "m", 1),  # not enough evidence yet
            rule("drop_text", "www.site.com"),
            rule("watermark_text", "ⓒ STUDIO", count=1),
            rule("sfx_text", "쾅", count=1),
        ]
    )
    untouched = region("r0006", "어디 가?", 500)
    regions = [
        region("r0001", "Jlnwoo, rn run!"),
        region("r0002", "www.site.com", 100, kind="free_text"),
        region("r0003", "ⓒ  STUDIO", 200),
        region("r0004", "쾅", 300),
        region("r0005", "www.site.com", 400, kind="watermark"),  # already a watermark: kept as one
        untouched,
    ]
    result, counts = apply_to_regions(regions, memory, CFG)
    assert [(r.id, r.kind, r.text, r.ocr_alt) for r in result] == [
        ("r0001", "bubble_text", "Jinwoo, rn run!", "Jlnwoo, rn run!"),
        ("r0003", "watermark", "ⓒ  STUDIO", None),
        ("r0004", "sfx", "쾅", None),
        ("r0005", "watermark", "www.site.com", None),
        ("r0006", "bubble_text", "어디 가?", None),
    ]
    assert result[-1] is untouched
    assert counts == {"learned_fixes": 1.0, "learned_drops": 1.0, "learned_kinds": 2.0}
    off = SeriesMemory(rules=[rule("drop_text", "www.site.com", enabled=False)])
    assert apply_to_regions(regions, off, CFG) == (
        regions,
        {"learned_fixes": 0.0, "learned_drops": 0.0, "learned_kinds": 0.0},
    )


def test_a_learned_fix_keeps_a_second_engines_reading() -> None:
    read = region("r0001", "Jlnwoo").model_copy(update={"ocr_alt": "Jinwco"})
    (fixed,), _ = apply_to_regions([read], SeriesMemory(rules=[rule("ocr_fix", "Jlnwoo", "Jinwoo")]), CFG)
    assert (fixed.text, fixed.ocr_alt) == ("Jinwoo", "Jinwco")


def test_translation_hints_pick_similar_memory_lines() -> None:
    memory = SeriesMemory(
        rules=[rule("preferred_term", "Hunter", "Hunters"), rule("preferred_term", "Mage", "Wizard", 1)],
        translations=[
            MemoryEntry(source="어디 가니?", english="Where to?", chapter="1"),
            MemoryEntry(source="안녕하세요", english="Good day", chapter="1"),
            MemoryEntry(source="어디 가요?", english="Where are you going?", chapter="2"),
            MemoryEntry(source="배고파", english="I'm hungry", chapter="2"),
        ],
    )
    hints = translation_hints(memory, LearnConfig(examples=2))
    assert hints.exact["어디 가요?"] == "Where are you going?"
    assert hints.preferences == (("Hunter", "Hunters"),)
    # the line itself is never its own example; unrelated lines are left out
    assert hints.examples_for(["어디  가요?"]) == [("어디 가니?", "Where to?")]
    assert hints.examples_for(["안녕하세요 형", "어디 가"]) == [
        ("안녕하세요", "Good day"),
        ("어디 가니?", "Where to?"),
    ]
    assert translation_hints(memory, LearnConfig(examples=0)).examples_for(["어디 가"]) == []
    assert similarity("같이 가자", "같이 가자!") == 0.75 and similarity("", "a") == 0.0


# ---------------------------------------------------------------- harvesting a chapter's edits


def test_harvest_reads_every_kind_of_correction(series: SeriesPaths) -> None:
    paths = make_chapter(
        series,
        "Chapter 1",
        [
            region("r0001", "Jlnwoo, run!"),
            region("r0002", "www.site.com", 200),
            region("r0003", "ⓒ STUDIO", 400),
        ],
        {"r0001": "Run, Jlnwoo! The Hunter Association"},
    )
    store.update_region(paths, "r0001", direction="ltr", text="Jinwoo, run!")
    store.delete_region(paths, "r0002", direction="ltr")
    store.update_region(paths, "r0003", direction="ltr", kind="watermark")
    store.add_region(paths, box(10, 600, 60, 650), direction="ltr", kind="sfx", text="쾅")
    store.set_translation(paths, "r0001", "Run, Jinwoo! The Hunters Guild", direction="ltr")
    found = harvest_chapter(paths)
    assert found.ocr_fixes == [("Jlnwoo, run!", "Jinwoo, run!")]
    assert found.deleted == ["www.site.com"]
    assert found.kinds == [("ⓒ STUDIO", "watermark"), ("쾅", "sfx")]
    assert found.translations == [("Jinwoo, run!", "Run, Jinwoo! The Hunters Guild", True, "Chapter 1")]
    assert found.english_fixes == [("Run, Jlnwoo! The Hunter Association", "Run, Jinwoo! The Hunters Guild")]

    # a re-run that already applies the lessons (the fix, the drop, the label) keeps every correction
    RegionsArtifact(
        regions=[region("r0001", "Jinwoo, run!"), region("r0003", "ⓒ STUDIO", 400, kind="watermark")]
    ).save(paths.artifact("ocr_auto.json"))
    FinalArtifact(
        judge_model="judge",
        lines=[FinalLine(region_id="r0001", text="Run, Jinwoo! The Hunters Guild", decision="pick")],
    ).save(paths.artifact("final_auto.json"))
    again = harvest_chapter(paths)
    assert (again.ocr_fixes, again.deleted, again.kinds, again.english_fixes) == (
        found.ocr_fixes,
        found.deleted,
        found.kinds,
        found.english_fixes,
    )


def test_a_kept_suggestion_is_memory_but_no_wording_lesson(series: SeriesPaths) -> None:
    paths = make_chapter(series, "Chapter 1", [region("r0001", "가자")], {"r0001": "Go"})
    store.set_translation(paths, "r0001", "Let's go!", direction="ltr", suggested_by="cloud")
    found = harvest_chapter(paths)
    assert found.translations == [("가자", "Let's go!", False, "Chapter 1")] and found.english_fixes == []


def test_an_edit_without_the_recorded_reading_is_compared_with_ocr_auto(series: SeriesPaths) -> None:
    paths = make_chapter(series, "Chapter 1", [region("r0001", "Jlnwoo")], {})
    store.update_region(paths, "r0001", direction="ltr", kind="free_text")  # creates ocr_auto.json
    legacy = RegionEdit(region_id="r0001", anchor=box(10, 0, 110, 50), text="Jinwoo")
    store.save_edits(paths, ChapterEdits(regions=[legacy]))
    assert harvest_chapter(paths).ocr_fixes == [("Jlnwoo", "Jinwoo")]
    assert harvest_chapter(make_chapter(series, "Chapter 2", [], {})) == Corrections()


# ---------------------------------------------------------------- memory.json


def test_memory_is_rebuilt_when_an_edit_changes_and_keeps_switches(series: SeriesPaths) -> None:
    assert current_memory(series) == SeriesMemory() and not memory_path(series).exists()
    one = make_chapter(series, "Chapter 1", [region("r0001", "Jlnwoo")], {})
    two = make_chapter(series, "Chapter 2", [region("r0001", "Jlnwoo?")], {})
    store.update_region(one, "r0001", direction="ltr", text="Jinwoo")
    memory = current_memory(series)
    assert [(r.wrong, r.right, r.count) for r in memory.rules] == [("Jlnwoo", "Jinwoo", 1)]
    assert SeriesMemory.load(memory_path(series)) == memory
    fix = memory.rules[0]
    assert set_rule_enabled(series, fix.id, False).enabled is False
    with pytest.raises(LookupError):
        set_rule_enabled(series, "nope", True)

    store.update_region(two, "r0001", direction="ltr", text="Jinwoo?")
    built = memory_path(series).stat().st_mtime_ns
    os.utime(two.artifact("edits.json"), ns=(built + 10**9, built + 10**9))  # edited after the last build
    rebuilt = current_memory(series)
    assert [(r.id, r.count, r.enabled) for r in rebuilt.rules] == [(fix.id, 2, False)]
    assert current_memory(series, rebuild=True) == rebuilt
    assert len(harvest_series(series).ocr_fixes) == 2

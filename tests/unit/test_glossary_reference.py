"""Unit tests for reference-mode glossary extraction (card GL1): pairing, extraction, aggregation, locking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omniscan.core.schemas import (
    BBox,
    GlossaryEntry,
    IngestArtifact,
    Region,
    RegionKind,
    SourceFile,
    TermType,
)
from omniscan.glossary.reference import (
    MAX_SOURCE_CHARS,
    MAX_TARGET_CHARS,
    AggregatedTerm,
    TermCandidate,
    aggregate,
    extract_candidates,
    merge_into_store,
    pair_chapter,
)
from omniscan.glossary.reference_prompts import TERMS_SCHEMA, PairedLine, parse_terms_reply
from omniscan.glossary.store import GlossaryStore
from omniscan.llm.ollama import ChatResponse
from omniscan.match.chapters import PagePair

# ---------------------------------------------------------------- synthetic chapters


def make_ingest(chapter: str, page_heights: list[int], width: int = 400) -> IngestArtifact:
    """An IngestArtifact whose pages tile the strip vertically (no gap)."""
    files: list[SourceFile] = []
    y = 0
    for index, height in enumerate(page_heights):
        files.append(
            SourceFile(
                index=index,
                name=f"{index:03d}.jpg",
                sha256=f"sha{chapter}-{index}",
                width=width,
                height=height,
                y0=y,
                y1=y + height,
            )
        )
        y += height
    return IngestArtifact(series="S", chapter=chapter, strip_width=width, strip_height=y, files=files)


def make_region(page: SourceFile, offset: int, order: int, text: str, *, kind: RegionKind = "bubble_text") -> Region:
    """A text region fully inside `page`'s strip-space y-range, `offset` px below the page top."""
    y0 = page.y0 + offset
    return Region(
        id=f"r{order:04d}",
        slice_index=page.index,
        kind=kind,
        bbox=BBox(x0=10, y0=y0, x1=200, y1=y0 + 40),
        reading_order=order,
        text=text,
        confidence=0.9,
    )


def pages(*pairs: tuple[int, int]) -> list[PagePair]:
    """Aligned page pairs of one chapter match, in reading order."""
    return [PagePair(page_a=a, page_b=b, similarity=0.9) for a, b in pairs]


# ---------------------------------------------------------------- pairing


def test_pair_chapter_pairs_regions_by_reading_order_rank_per_page() -> None:
    ingest_a = make_ingest("Chapter 001", [200, 200])
    ingest_b = make_ingest("ch_01", [200, 200])
    # ids deliberately oppose reading_order: pairing must follow reading order, not region id
    regions_a = [
        make_region(ingest_a.files[0], 10, 1, "민준이"),
        make_region(ingest_a.files[0], 100, 0, "안녕"),
    ]
    regions_b = [
        make_region(ingest_b.files[0], 12, 0, "Hello"),
        make_region(ingest_b.files[0], 102, 1, "Minjun"),
    ]
    result = pair_chapter(regions_a, regions_b, ingest_a, ingest_b, pages((0, 0)))
    assert [(line.source, line.target) for line in result.lines] == [("안녕", "Hello"), ("민준이", "Minjun")]
    assert all(line.page_a == 0 and line.page_b == 0 for line in result.lines)
    assert (result.pages_paired, result.pages_skipped) == (1, 0)


def test_pair_chapter_maps_regions_to_pages_by_strip_y_ranges() -> None:
    ingest_a = make_ingest("Chapter 001", [200, 200])
    ingest_b = make_ingest("ch_01", [200, 200])
    regions_a = [make_region(ingest_a.files[1], 10, 0, "민준이")]
    regions_b = [make_region(ingest_b.files[1], 10, 0, "Minjun")]
    result = pair_chapter(regions_a, regions_b, ingest_a, ingest_b, pages((0, 0), (1, 1)))
    assert [(line.source, line.target, line.page_a, line.page_b) for line in result.lines] == [
        ("민준이", "Minjun", 1, 1)
    ]


def test_pair_chapter_skips_page_pairs_whose_region_counts_diverge() -> None:
    ingest_a = make_ingest("Chapter 001", [200, 200])
    ingest_b = make_ingest("ch_01", [200, 200])
    regions_a = [make_region(ingest_a.files[0], 10, i, f"원문{i}") for i in range(3)]
    regions_b = [make_region(ingest_b.files[0], 10, 0, "One")]
    result = pair_chapter(regions_a, regions_b, ingest_a, ingest_b, pages((0, 0), (1, 1)))
    assert not result.lines and (result.pages_paired, result.pages_skipped) == (1, 1)


def test_pair_chapter_tolerates_one_stray_region_per_page() -> None:
    ingest_a = make_ingest("Chapter 001", [200])
    ingest_b = make_ingest("ch_01", [200])
    regions_a = [
        make_region(ingest_a.files[0], 10, 0, "민준이"),
        make_region(ingest_a.files[0], 100, 1, "서윤아"),
    ]
    regions_b = [make_region(ingest_b.files[0], 10, 0, "Minjun")]  # second bubble not detected on this side
    result = pair_chapter(regions_a, regions_b, ingest_a, ingest_b, pages((0, 0)))
    assert [(line.source, line.target) for line in result.lines] == [("민준이", "Minjun")]
    assert (result.pages_paired, result.pages_skipped) == (1, 0)


def test_pair_chapter_never_pairs_watermarks_or_empty_text() -> None:
    ingest_a = make_ingest("Chapter 001", [200])
    ingest_b = make_ingest("ch_01", [200])
    regions_a = [
        make_region(ingest_a.files[0], 10, 0, "SCAN SITE", kind="watermark"),
        make_region(ingest_a.files[0], 100, 1, "민준이"),
    ]
    regions_b = [
        make_region(ingest_b.files[0], 10, 0, ""),
        make_region(ingest_b.files[0], 100, 1, "Minjun"),
    ]
    result = pair_chapter(regions_a, regions_b, ingest_a, ingest_b, pages((0, 0)))
    assert [(line.source, line.target) for line in result.lines] == [("민준이", "Minjun")]


def test_pair_chapter_collapses_newlines_inside_a_region() -> None:
    ingest_a = make_ingest("Chapter 001", [200])
    ingest_b = make_ingest("ch_01", [200])
    result = pair_chapter(
        [make_region(ingest_a.files[0], 10, 0, "민준이는\n 말했다")],
        [make_region(ingest_b.files[0], 10, 0, "Minjun\n said")],
        ingest_a,
        ingest_b,
        pages((0, 0)),
    )
    assert (result.lines[0].source, result.lines[0].target) == ("민준이는 말했다", "Minjun said")


# ---------------------------------------------------------------- extraction


@dataclass
class FakeClient:
    """ChatClient double: scripted replies, every call recorded."""

    replies: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        cloud: bool = False,
        format: dict[str, Any] | str | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
        think: bool | None = None,
        max_retries: int = 5,
    ) -> ChatResponse:
        self.calls.append(
            {"model": model, "messages": messages, "cloud": cloud, "format": format, "options": options}
        )
        return ChatResponse(
            content=self.replies.pop(0),
            model=model,
            done=True,
            total_duration_ns=None,
            prompt_eval_count=None,
            eval_count=None,
            raw={},
        )


REPLY = '{"terms": [{"source": "민준", "target": "Minjun", "type": "person"}]}'


def test_extract_candidates_makes_no_request_without_lines() -> None:
    client = FakeClient()
    assert extract_candidates(client, "m", "Chapter 001", []) == []
    assert client.calls == []


def test_extract_candidates_sends_the_schema_prompt_and_tags_the_chapter() -> None:
    client = FakeClient([REPLY])
    lines = [PairedLine("민준", "Minjun", 0, 0)]
    terms = extract_candidates(client, "gemma4:31b-cloud", "Chapter 001", lines)
    assert [(t.source, t.target, t.type, t.chapter) for t in terms] == [
        ("민준", "Minjun", "person", "Chapter 001")
    ]
    call = client.calls[0]
    assert call["model"] == "gemma4:31b-cloud"
    assert call["cloud"] is False
    assert call["format"] == TERMS_SCHEMA
    assert call["options"] == {"temperature": 0.0}
    assert "Minjun" in call["messages"][1]["content"]


def test_parse_terms_reply_tolerates_malformed_replies() -> None:
    assert parse_terms_reply("not json at all") == []
    assert parse_terms_reply('{"nope": []}') == []
    assert parse_terms_reply("[]") == []  # a bare array holds no {"terms": ...} object
    assert parse_terms_reply('[{"source": "x"}]') == []  # missing target
    assert parse_terms_reply('{"terms": ["junk", {"source": 1, "target": "y"}]}') == []


def test_parse_terms_reply_defaults_and_strips() -> None:
    terms = parse_terms_reply(
        '{"terms": ['
        '{"source": " 민준 ", "target": "Minjun"},'
        '{"source": "마법사", "target": "Wizard", "type": "rank"},'
        '{"source": "화과", "target": "Whoosh", "type": "not-a-type"}'
        "]}"
    )
    assert [(t.source, t.target, t.type) for t in terms] == [
        ("민준", "Minjun", "other"),
        ("마법사", "Wizard", "rank"),
        ("화과", "Whoosh", "other"),
    ]


# ---------------------------------------------------------------- aggregation


def cand(source: str, target: str, chapter: str, type_: TermType = "person") -> TermCandidate:
    return TermCandidate(source=source, target=target, type=type_, chapter=chapter)


def test_aggregate_counts_distinct_chapters_not_occurrences() -> None:
    terms = aggregate(
        [
            cand("민준", "Minjun", "Chapter 001"),
            cand("민준", "Minjun", "Chapter 001"),  # twice within one chapter: one distinct chapter
            cand("민준", "Minjun", "Chapter 002"),
            cand("민준", "Minjun", "Chapter 003"),
        ]
    )
    assert len(terms) == 1
    assert (terms[0].chapters, terms[0].occurrences) == (3, 4)


def test_aggregate_keeps_disagreeing_targets_as_alternatives() -> None:
    terms = aggregate(
        [
            cand("민준", "Minjun", "Chapter 001"),
            cand("민준", "Minjun", "Chapter 002"),
            cand("민준", "Min-jun", "Chapter 003"),
        ]
    )
    assert (terms[0].target, terms[0].chapters) == ("Minjun", 2)
    assert terms[0].alternatives == ("Min-jun",)


def test_aggregate_folds_target_case_and_stores_the_most_frequent_spelling() -> None:
    terms = aggregate(
        [
            cand("민준", "Minjun", "Chapter 001"),
            cand("민준", "MINJUN", "Chapter 002"),
            cand("민준", "Minjun", "Chapter 003"),
        ]
    )
    assert terms[0].target == "Minjun" and terms[0].chapters == 3 and terms[0].alternatives == ()


def test_aggregate_drops_sentence_sized_terms() -> None:
    terms = aggregate(
        [
            cand("가" * (MAX_SOURCE_CHARS + 1), "Too Long", "Chapter 001"),
            cand("민준", "나" * (MAX_TARGET_CHARS + 1), "Chapter 001"),
            cand("민준", "Minjun", "Chapter 001"),
        ]
    )
    assert [(t.source, t.target) for t in terms] == [("민준", "Minjun")]


def test_aggregate_records_first_seen_chapter_and_type() -> None:
    terms = aggregate(
        [
            cand("민준", "Minjun", "Chapter 010", "person"),
            cand("민준", "Minjun", "Chapter 002", "person"),
            cand("민준", "Minjun", "Chapter 003", "other"),
        ]
    )
    assert terms[0].first_seen_chapter == 2.0 and terms[0].type == "person"


# ---------------------------------------------------------------- merging


def terms(source: str, target: str, chapters: int, occurrences: int | None = None) -> AggregatedTerm:
    return AggregatedTerm(
        source=source,
        target=target,
        type="person",
        chapters=chapters,
        occurrences=occurrences if occurrences is not None else chapters,
        first_seen_chapter=1.0,
        alternatives=(),
    )


def make_store(tmp_path: Path, *entries: dict[str, Any]) -> GlossaryStore:
    s = GlossaryStore(tmp_path / "series.db")
    for entry in entries:
        s.add(GlossaryEntry.model_validate(entry))
    return s


def test_merge_locks_at_threshold_and_proposes_below(tmp_path: Path) -> None:
    with make_store(tmp_path) as s:
        report = merge_into_store(s, [terms("민준", "Minjun", 3), terms("서윤", "Seoyun", 2)], min_locks=3)
        assert (report.locked, report.proposed, report.conflicts, report.rejected) == (1, 1, (), ())
        locked = s.find_by_source("민준")
        proposed = s.find_by_source("서윤")
    assert locked is not None and (locked.status, locked.origin, locked.count) == ("locked", "reference", 3)
    assert proposed is not None and (proposed.status, proposed.origin) == ("proposed", "reference")


def test_merge_agreement_with_locked_bumps_count_without_duplicating(tmp_path: Path) -> None:
    entry = {"source": "민준", "target": "Minjun", "status": "locked", "origin": "user", "count": 2}
    with make_store(tmp_path, entry) as s:
        report = merge_into_store(s, [terms("민준", "Minjun", 3, occurrences=5)])
        assert report.conflicts == () and report.locked == 1
        rows = s.list()
    assert len(rows) == 1
    assert (rows[0].target, rows[0].count, rows[0].status, rows[0].origin) == ("Minjun", 5, "locked", "user")


def test_merge_never_lowers_a_locked_count(tmp_path: Path) -> None:
    entry = {"source": "민준", "target": "Minjun", "status": "locked", "count": 9}
    with make_store(tmp_path, entry) as s:
        merge_into_store(s, [terms("민준", "Minjun", 3, occurrences=2)])
        found = s.find_by_source("민준")
        assert found is not None and found.count == 9


def test_merge_conflict_with_locked_flags_and_leaves_untouched(tmp_path: Path) -> None:
    entry = {"source": "민준", "target": "Minjun", "status": "locked", "origin": "user"}
    with make_store(tmp_path, entry) as s:
        report = merge_into_store(s, [terms("민준", "Min-jun", 4)])
        assert len(report.conflicts) == 1
        conflict = report.conflicts[0]
        assert (conflict.source, conflict.existing_target, conflict.extracted_target) == ("민준", "Minjun", "Min-jun")
        assert conflict.existing_status == "locked"
        row = s.find_by_source("민준")
    assert row is not None and (row.target, row.status) == ("Minjun", "locked")


def test_merge_rejected_entries_are_reported_and_untouched(tmp_path: Path) -> None:
    entry = {"source": "민준", "target": "Minjun", "status": "rejected"}
    with make_store(tmp_path, entry) as s:
        report = merge_into_store(s, [terms("민준", "Minjun", 5)])
        assert report.rejected == ("민준",) and report.locked == 0 and report.proposed == 0
        row = s.find_by_source("민준")
        assert row is not None and (row.status, row.count) == ("rejected", 0)


def test_merge_agrees_with_user_proposed_but_never_replaces_their_target(tmp_path: Path) -> None:
    agree_entry = {"source": "민준", "target": "Minjun", "status": "proposed", "origin": "user"}
    disagree_entry = {"source": "서윤", "target": "Seoyun", "status": "proposed", "origin": "user"}
    with make_store(tmp_path, agree_entry, disagree_entry) as s:
        agree = merge_into_store(s, [terms("민준", "MINJUN", 2, occurrences=4)])
        disagree = merge_into_store(s, [terms("서윤", "Seo-yun", 1)])
        assert agree.conflicts == () and len(disagree.conflicts) == 1
        assert disagree.conflicts[0].existing_status == "proposed"
        minjun = s.find_by_source("민준")
        seoyun = s.find_by_source("서윤")
    assert minjun is not None and minjun.count == 4
    assert seoyun is not None and seoyun.target == "Seoyun"


def test_merge_updates_its_own_proposed_in_place(tmp_path: Path) -> None:
    entry = {"source": "민준", "target": "Min-jun", "status": "proposed", "origin": "reference"}
    with make_store(tmp_path, entry) as s:
        report = merge_into_store(s, [terms("민준", "Minjun", 3)])
        rows = s.list()
    assert len(rows) == 1
    assert (rows[0].target, rows[0].status, rows[0].origin) == ("Minjun", "locked", "reference")
    assert rows[0].notes is not None and "Min-jun" in rows[0].notes
    assert report.conflicts == ()


def test_merge_reports_runner_up_targets_in_notes(tmp_path: Path) -> None:
    term = AggregatedTerm(
        source="민준",
        target="Minjun",
        type="person",
        chapters=2,
        occurrences=2,
        first_seen_chapter=1.0,
        alternatives=("Min-jun",),
    )
    with make_store(tmp_path) as s:
        merge_into_store(s, [term], min_locks=3)
        row = s.find_by_source("민준")
        assert row is not None and row.notes is not None and "Min-jun" in row.notes


def test_merge_write_false_computes_the_report_but_writes_nothing(tmp_path: Path) -> None:
    with make_store(tmp_path) as s:
        report = merge_into_store(s, [terms("민준", "Minjun", 3)], write=False)
        assert report.locked == 1
        assert s.list() == []


def test_merge_rerun_is_idempotent(tmp_path: Path) -> None:
    with make_store(tmp_path) as s:
        first = merge_into_store(s, [terms("민준", "Minjun", 3, occurrences=5)])
        second = merge_into_store(s, [terms("민준", "Minjun", 3, occurrences=5)])
        rows = s.list()
    assert (first.locked, second.locked) == (1, 1)
    assert len(rows) == 1 and rows[0].count == 5


def test_merge_custom_min_locks(tmp_path: Path) -> None:
    with make_store(tmp_path) as s:
        merge_into_store(s, [terms("민준", "Minjun", 2)], min_locks=2)
        row = s.find_by_source("민준")
        assert row is not None and row.status == "locked"
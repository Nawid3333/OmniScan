"""Tests for omniscan.glossary.proposals (hand-built ocr.json fixtures, fake client)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import BBox, GlossaryEntry, Region, RegionsArtifact
from omniscan.glossary.proposal_prompts import PROPOSAL_SYSTEM
from omniscan.glossary.proposals import (
    DEFAULT_MIN_CHAPTERS,
    chapter_lines,
    extract_proposals,
    run_proposals,
)
from omniscan.glossary.reference import TermCandidate, aggregate
from omniscan.glossary.reference_prompts import TERMS_SCHEMA
from omniscan.glossary.store import GlossaryStore
from omniscan.llm.ollama import ChatResponse
from omniscan.translate.prompts import source_text

MODEL = "proposal-model"


class FakeClient:
    """One scripted reply per chat call; records every call."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

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


def terms_reply(*terms: dict[str, str]) -> str:
    items = ", ".join(
        f'{{"source": "{t["source"]}", "target": "{t["target"]}", "type": "{t.get("type", "person")}"}}'
        for t in terms
    )
    return '{"terms": [' + items + "]}"


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
    )


def make_series(cfg: Config, *chapters: str) -> SeriesPaths:
    sp = SeriesPaths.from_config(cfg, "S")
    for name in chapters:
        (sp.library_dir / name).mkdir(parents=True, exist_ok=True)
    return sp


def write_ocr(cfg: Config, chapter: str) -> None:
    """One ocr.json whose region ids oppose reading_order (sort must follow the geometry)."""
    paths = SeriesPaths.from_config(cfg, "S").chapter(chapter)
    regions = [
        Region(
            id=rid,
            slice_index=index,
            kind=kind,
            bbox=BBox(x0=0, y0=0, x1=10, y1=10),
            reading_order=order,
            text=text,
        )
        for index, (rid, order, kind, text) in enumerate(
            [
                ("r0002", 1, "bubble_text", "민준이가"),
                ("r0001", 0, "bubble_text", "안녕"),
                ("r0003", 2, "watermark", "SCAN SITE"),
                ("r0004", 3, "bubble_text", "  \n "),
                ("r0005", 3, "sfx", "쿵!"),
            ]
        )
    ]
    RegionsArtifact(regions=regions).save(paths.artifact("ocr.json"))


def store(cfg: Config) -> GlossaryStore:
    return GlossaryStore(SeriesPaths.from_config(cfg, "S").db)


# ---------------------------------------------------------------- chapter_lines


def test_chapter_lines_matches_translatable_filter_and_order(cfg: Config) -> None:
    make_series(cfg, "Chapter 1")
    write_ocr(cfg, "Chapter 1")
    sp = SeriesPaths.from_config(cfg, "S")
    lines = chapter_lines(sp, "Chapter 1")
    assert lines == ["민준이가", "안녕", "쿵!"]  # watermark + blank dropped; sorted by slice, not by id
    artifact = RegionsArtifact.load(sp.chapter("Chapter 1").artifact("ocr.json"))
    assert [source_text(r) for r in artifact.regions] != lines  # the artifact's own order differs


def test_chapter_lines_missing_ocr_is_empty(cfg: Config) -> None:
    make_series(cfg, "Chapter 1")
    assert chapter_lines(SeriesPaths.from_config(cfg, "S"), "Chapter 1") == []


# ---------------------------------------------------------------- extract_proposals


def test_extract_proposals_makes_no_request_without_lines() -> None:
    client = FakeClient([])
    assert extract_proposals(client, MODEL, "Chapter 1", []) == []
    assert client.calls == []


def test_extract_proposals_sends_the_schema_prompt_and_tags_the_chapter() -> None:
    client = FakeClient([terms_reply({"source": "민준", "target": "Minjun", "type": "person"})])
    terms = extract_proposals(client, MODEL, "Chapter 1", ["민준이가 간다"])
    assert [(t.source, t.target, t.type, t.chapter) for t in terms] == [
        ("민준", "Minjun", "person", "Chapter 1")
    ]
    call = client.calls[0]
    assert (call["model"], call["cloud"], call["format"], call["options"]) == (
        MODEL,
        False,
        TERMS_SCHEMA,
        {"temperature": 0.0},
    )
    assert call["messages"][0]["content"] == PROPOSAL_SYSTEM
    assert "민준이가 간다" in call["messages"][1]["content"]


# ---------------------------------------------------------------- run_proposals


def test_run_proposals_threshold_and_never_locked(tmp_path: Path, cfg: Config) -> None:
    make_series(cfg, "Chapter 1", "Chapter 2")
    write_ocr(cfg, "Chapter 1")
    write_ocr(cfg, "Chapter 2")
    client = FakeClient(
        [
            terms_reply({"source": "민준", "target": "Minjun"}, {"source": "서윤", "target": "Seoyun"}),
            terms_reply({"source": "민준", "target": "Minjun"}),
        ]
    )
    summary = run_proposals(cfg, "S", client=client, model=MODEL)
    assert (summary.chapters_scanned, summary.chapters_with_lines, summary.lines_scanned) == (
        ("Chapter 1", "Chapter 2"),
        2,
        6,
    )
    assert summary.aggregated_above_threshold == 1  # 서윤 recurs in 1 chapter: below the default
    assert summary.merge.locked == 0 and summary.merge.proposed == 1
    with store(cfg) as s:
        rows = s.list()
    assert [(r.source, r.status, r.origin) for r in rows] == [("민준", "proposed", "llm")]
    assert rows[0].count == 2

    # min_chapters=1 would let 서윤 through, but still never as locked
    reply = terms_reply({"source": "서윤", "target": "Seoyun"})
    client = FakeClient([reply, reply])
    summary = run_proposals(cfg, "S", client=client, model=MODEL, min_chapters=1)
    assert summary.merge.locked == 0 and summary.merge.proposed == 1
    with store(cfg) as s:
        seoyun = s.find_by_source("서윤")
    assert seoyun is not None and (seoyun.status, seoyun.origin) == ("proposed", "llm")


def test_run_proposals_conflict_with_locked_leaves_it_untouched(tmp_path: Path, cfg: Config) -> None:
    make_series(cfg, "Chapter 1", "Chapter 2")
    write_ocr(cfg, "Chapter 1")
    write_ocr(cfg, "Chapter 2")
    with store(cfg) as s:
        s.add(GlossaryEntry(source="민준", target="Minjun", status="locked", origin="user"))
    reply = terms_reply({"source": "민준", "target": "Min-jun"})
    summary = run_proposals(cfg, "S", client=FakeClient([reply, reply]), model=MODEL)
    assert len(summary.merge.conflicts) == 1
    conflict = summary.merge.conflicts[0]
    assert (
        conflict.source,
        conflict.existing_target,
        conflict.extracted_target,
        conflict.existing_status,
    ) == (
        "민준",
        "Minjun",
        "Min-jun",
        "locked",
    )
    with store(cfg) as s:
        row = s.find_by_source("민준")
    assert row is not None and (row.target, row.status) == ("Minjun", "locked")


def test_run_proposals_updates_its_own_previous_proposal_in_place(tmp_path: Path, cfg: Config) -> None:
    make_series(cfg, "Chapter 1", "Chapter 2")
    write_ocr(cfg, "Chapter 1")
    write_ocr(cfg, "Chapter 2")
    reply = terms_reply({"source": "민준", "target": "Min-jun"})
    run_proposals(cfg, "S", client=FakeClient([reply, reply]), model=MODEL)
    with store(cfg) as s:
        first = s.find_by_source("민준")
        assert first is not None and first.target == "Min-jun"
    reply2 = terms_reply({"source": "민준", "target": "Minjun"})
    summary = run_proposals(cfg, "S", client=FakeClient([reply2, reply2]), model=MODEL)
    assert summary.merge.conflicts == () and summary.merge.proposed == 1
    with store(cfg) as s:
        rows = s.list()
    assert len(rows) == 1  # updated in place, never duplicated
    assert (rows[0].target, rows[0].status, rows[0].origin) == ("Minjun", "proposed", "llm")


def test_run_proposals_dry_run_writes_nothing(cfg: Config) -> None:
    make_series(cfg, "Chapter 1")
    write_ocr(cfg, "Chapter 1")
    reply = terms_reply({"source": "민준", "target": "Minjun"})
    summary = run_proposals(cfg, "S", client=FakeClient([reply]), model=MODEL, dry_run=True, min_chapters=1)
    assert summary.merge.proposed == 1  # the report is still computed
    sp = SeriesPaths.from_config(cfg, "S")
    assert not sp.db.exists() and not sp.glossary_yaml.exists()


def test_run_proposals_dry_run_on_existing_db_writes_nothing(tmp_path: Path, cfg: Config) -> None:
    make_series(cfg, "Chapter 1")
    write_ocr(cfg, "Chapter 1")
    with store(cfg) as s:
        s.add(GlossaryEntry(source="민준", target="Minjun", status="locked", count=7))
    reply = terms_reply({"source": "민준", "target": "Minjun"})
    summary = run_proposals(cfg, "S", client=FakeClient([reply]), model=MODEL, dry_run=True, min_chapters=1)
    assert summary.merge.locked == 1  # agreeing locked entry counted, but ...
    with store(cfg) as s:
        row = s.find_by_source("민준")
    assert row is not None and row.count == 7  # ... the count was not raised


def test_run_proposals_no_ocr_anywhere_is_an_empty_pass(tmp_path: Path, cfg: Config) -> None:
    make_series(cfg, "Chapter 1", "Chapter 2")
    client = FakeClient([])
    summary = run_proposals(cfg, "S", client=client, model=MODEL)
    assert (summary.chapters_scanned, summary.chapters_with_lines, summary.lines_scanned) == (
        ("Chapter 1", "Chapter 2"),
        0,
        0,
    )
    assert summary.aggregated_above_threshold == 0
    assert (
        summary.merge.locked,
        summary.merge.proposed,
        summary.merge.conflicts,
        summary.merge.rejected,
    ) == (
        0,
        0,
        (),
        (),
    )
    assert client.calls == []


def test_run_proposals_chapters_param_restricts(tmp_path: Path, cfg: Config) -> None:
    make_series(cfg, "Chapter 1", "Chapter 2")
    write_ocr(cfg, "Chapter 1")
    write_ocr(cfg, "Chapter 2")
    reply = terms_reply({"source": "민준", "target": "Minjun"})
    summary = run_proposals(cfg, "S", client=FakeClient([reply]), model=MODEL, chapters=["Chapter 2"])
    assert summary.chapters_scanned == ("Chapter 2",)
    assert summary.chapters_with_lines == 1


def test_aggregated_terms_reuse_reference_aggregation() -> None:
    candidates = [
        TermCandidate("민준", "Minjun", "person", "Chapter 1"),
        TermCandidate("민준", "Minjun", "person", "Chapter 2"),
    ]
    aggregated = aggregate(candidates)
    assert len(aggregated) == 1 and aggregated[0].chapters == 2
    assert DEFAULT_MIN_CHAPTERS == 2

"""Tests for omniscan.story.summarize (hand-built chapter artifacts, fake client)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import BBox, FinalArtifact, FinalLine, Region, RegionsArtifact
from omniscan.llm.ollama import ChatResponse
from omniscan.story.prompts import SUMMARY_SCHEMA
from omniscan.story.store import SummaryStore
from omniscan.story.summarize import chapter_final_lines, run_summarize, summarize_chapter

MODEL = "summary-model"


def region(rid: str, *, slice_index: int = 0, reading_order: int = 0) -> Region:
    return Region(
        id=rid,
        slice_index=slice_index,
        kind="bubble_text",
        bbox=BBox(x0=0, y0=0, x1=10, y1=10),
        reading_order=reading_order,
    )


def write_ocr(paths: ChapterPaths, *regions: dict[str, Any]) -> None:
    RegionsArtifact(regions=[region(**r) for r in regions]).save(paths.artifact("ocr.json"))


def write_final(paths: ChapterPaths, lines: list[tuple[str, str]]) -> None:
    FinalArtifact(
        judge_model="judge",
        lines=[FinalLine(region_id=rid, text=text, decision="pick", sources=["run"]) for rid, text in lines],
    ).save(paths.artifact("final.json"))


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
    """A series with empty raw chapter folders (reading order = chapter_number order)."""
    sp = SeriesPaths.from_config(cfg, "S")
    for name in chapters:
        (sp.library_dir / name).mkdir(parents=True, exist_ok=True)
    return sp


def paths_of(cfg: Config, chapter: str) -> ChapterPaths:
    return SeriesPaths.from_config(cfg, "S").chapter(chapter)


def ready(cfg: Config, chapter: str, lines: list[tuple[str, str]]) -> None:
    """Write final.json + ocr.json for one chapter."""
    paths = paths_of(cfg, chapter)
    write_ocr(
        paths, *[{"rid": rid, "slice_index": i, "reading_order": i} for i, (rid, _t) in enumerate(lines)]
    )
    write_final(paths, lines)


# ---------------------------------------------------------------- chapter_final_lines


def paths_bare(tmp_path: Path) -> ChapterPaths:
    return ChapterPaths(
        series="S",
        chapter="c",
        raw_dir=tmp_path / "r",
        work_dir=tmp_path / "w",
        output_dir=tmp_path / "o",
        filtered_dir=tmp_path / "o" / "f",
    )


def test_chapter_final_lines_order_by_ocr_geometry_not_file_order(tmp_path: Path) -> None:
    paths = paths_bare(tmp_path)
    # ocr.json ids oppose reading_order; final.json lists them backwards — neither decides the order
    write_ocr(
        paths,
        {"rid": "r0003", "slice_index": 1, "reading_order": 0},
        {"rid": "r0001", "slice_index": 0, "reading_order": 1},
        {"rid": "r0002", "slice_index": 0, "reading_order": 0},
    )
    write_final(paths, [("r0003", "three"), ("r0001", "one"), ("r0002", "two")])
    assert chapter_final_lines(paths) == ["two", "one", "three"]


def test_chapter_final_lines_drops_unknown_regions_and_blank_lines(tmp_path: Path) -> None:
    paths = paths_bare(tmp_path)
    write_ocr(paths, {"rid": "r0001"}, {"rid": "r0002"})
    write_final(paths, [("r0001", "kept"), ("r0002", "  \n "), ("r9999", "ghost")])
    assert chapter_final_lines(paths) == ["kept"]


def test_chapter_final_lines_missing_files_are_empty(tmp_path: Path) -> None:
    paths = paths_bare(tmp_path)
    write_final(paths, [("r0001", "kept")])
    assert chapter_final_lines(paths) == []  # no ocr.json
    write_ocr(paths, {"rid": "r0001"})
    paths.artifact("final.json").unlink()
    assert chapter_final_lines(paths) == []  # no final.json


# ---------------------------------------------------------------- summarize_chapter / run_summarize


def test_summarize_chapter_makes_no_request_without_lines() -> None:
    client = FakeClient([])
    assert summarize_chapter(client, MODEL, []) is None
    assert client.calls == []


def test_summarize_chapter_sends_schema_prompt_and_parses() -> None:
    client = FakeClient(['{"summary": "Minjun enters the gate."}'])
    summary = summarize_chapter(client, MODEL, ["첫 문장"])
    assert summary == "Minjun enters the gate."
    call = client.calls[0]
    assert (call["model"], call["cloud"], call["format"], call["options"]) == (
        MODEL,
        False,
        SUMMARY_SCHEMA,
        {"temperature": 0.0},
    )
    assert "첫 문장" in call["messages"][1]["content"]


def test_run_summarize_skips_and_recomputes(tmp_path: Path, cfg: Config) -> None:
    make_series(cfg, "Chapter 1", "Chapter 2", "Chapter 3")
    ready(cfg, "Chapter 1", [("r0001", "첫 문장")])
    ready(cfg, "Chapter 2", [("r0001", "둘째 문장")])
    sp = SeriesPaths.from_config(cfg, "S")
    with SummaryStore(sp.db) as store:  # Chapter 2 already summarized before the call
        store.set("Chapter 2", "Prior.", "old-model")

    client = FakeClient(['{"summary": "One."}'])
    summary = run_summarize(cfg, "S", client=client, model=MODEL)
    assert summary.done == ("Chapter 1",)
    assert summary.skipped_existing == ("Chapter 2",)
    assert summary.skipped_no_final == ("Chapter 3",)
    assert len(client.calls) == 1  # Chapter 2 skipped, Chapter 3 had nothing to summarize
    with SummaryStore(sp.db) as store:
        row = store.get("Chapter 1")
        assert row is not None and row.summary == "One."
        prior = store.get("Chapter 2")
        assert prior is not None and prior.model == "old-model"  # untouched

    # force re-summarizes the previously-existing one
    client = FakeClient(['{"summary": "One again."}', '{"summary": "Two again."}'])
    summary = run_summarize(cfg, "S", client=client, model=MODEL, force=True)
    assert summary.done == ("Chapter 1", "Chapter 2")
    assert summary.skipped_no_final == ("Chapter 3",)
    with SummaryStore(sp.db) as store:
        row = store.get("Chapter 2")
        assert row is not None and row.summary == "Two again."


def test_run_summarize_unusable_reply_counts_as_no_final(tmp_path: Path, cfg: Config) -> None:
    make_series(cfg, "Chapter 1")
    ready(cfg, "Chapter 1", [("r0001", "첫 문장")])
    client = FakeClient([""])  # an empty reply parses to None
    summary = run_summarize(cfg, "S", client=client, model=MODEL)
    assert summary.done == () and summary.skipped_no_final == ("Chapter 1",)


def test_run_summarize_given_chapters_restrict_and_keep_the_given_order(tmp_path: Path, cfg: Config) -> None:
    make_series(cfg, "Chapter 10", "Chapter 2")  # SeriesPaths order would be ["Chapter 2", "Chapter 10"]
    ready(cfg, "Chapter 2", [("r0001", "둘째")])
    ready(cfg, "Chapter 10", [("r0001", "열째")])
    client = FakeClient(['{"summary": "Ten."}', '{"summary": "Two."}'])
    summary = run_summarize(cfg, "S", client=client, model=MODEL, chapters=["Chapter 10", "Chapter 2"])
    assert summary.done == ("Chapter 10", "Chapter 2")
    assert "열째" in client.calls[0]["messages"][1]["content"]
    with SummaryStore(SeriesPaths.from_config(cfg, "S").db) as store:
        assert [row.summary for row in store.list()] == ["Ten.", "Two."]  # store.list: chapter order


def test_run_summarize_no_chapters_at_all(tmp_path: Path, cfg: Config) -> None:
    make_series(cfg)  # no chapter folders
    client = FakeClient([])
    summary = run_summarize(cfg, "S", client=client, model=MODEL)
    assert (summary.done, summary.skipped_no_final, summary.skipped_existing) == ((), (), ())
    assert client.calls == []


def test_run_summarize_stores_a_plain_json_reply(tmp_path: Path, cfg: Config) -> None:
    make_series(cfg, "Chapter 1")
    ready(cfg, "Chapter 1", [("r0001", "첫 문장")])
    client = FakeClient([json.dumps({"summary": "Fenced."})])
    summary = run_summarize(cfg, "S", client=client, model=MODEL)
    assert summary.done == ("Chapter 1",)
    with SummaryStore(SeriesPaths.from_config(cfg, "S").db) as store:
        row = store.get("Chapter 1")
        assert row is not None and row.summary == "Fenced."

"""Tests for incremental translation and judging: only regions whose inputs changed reach the model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import (
    BBox,
    Candidate,
    CandidateRun,
    FinalArtifact,
    GlossaryEntry,
    Region,
    RegionsArtifact,
)
from omniscan.llm.ollama import ChatResponse
from omniscan.translate.chapter import translate_chapter
from omniscan.translate.incremental import judge_key, translation_key
from omniscan.translate.judge_chapter import judge_chapter
from omniscan.translate.judge_config import JudgeConfig
from omniscan.translate.profiles import TranslationProfile

PROFILE = TranslationProfile(name="p", endpoint="local", model="m", style="chat_json")
JUDGE = JudgeConfig(model="judge-model")


def region(rid: str, text: str, order: int) -> Region:
    return Region(
        id=rid,
        slice_index=0,
        kind="bubble_text",
        bbox=BBox(x0=0, y0=order * 50, x1=10, y1=order * 50 + 10),
        reading_order=order,
        text=text,
    )


def write_ocr(paths: ChapterPaths, texts: dict[str, str]) -> None:
    regions = [region(rid, text, i) for i, (rid, text) in enumerate(texts.items())]
    RegionsArtifact(regions=regions).save(paths.artifact("ocr.json"))


class EchoClient:
    """chat_json: answers `<tag>:<source>` per region; judge: always picks candidate A. Records requests."""

    def __init__(self, tag: str = "EN") -> None:
        self.tag = tag
        self.requests: list[list[dict[str, Any]]] = []

    def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        self.requests.append(messages)
        user = messages[-1]["content"]
        items = json.loads(user.split("Regions (reading order):\n", 1)[1])
        if "judgements" in json.dumps(kwargs.get("format")):
            answer = {"judgements": [{"id": r["id"], "decision": "pick", "pick": "A"} for r in items]}
        else:
            answer = {"translations": [{"id": r["id"], "text": f"{self.tag}:{r['text']}"} for r in items]}
        return ChatResponse(json.dumps(answer, ensure_ascii=False), model, True, None, 1, 1, {})

    def sent_ids(self, index: int) -> list[str]:
        user = self.requests[index][-1]["content"]
        return [r["id"] for r in json.loads(user.split("Regions (reading order):\n", 1)[1])]


@pytest.fixture
def paths(tmp_path: Path) -> ChapterPaths:
    return ChapterPaths("S", "Chapter 1", tmp_path / "r", tmp_path / "w", tmp_path / "o", tmp_path / "f")


def run(paths: ChapterPaths) -> dict[str, Candidate]:
    return {c.region_id: c for c in CandidateRun.load(paths.artifact("translations/p.json")).candidates}


def test_keys_change_with_text_glossary_and_model_only() -> None:
    r = region("r0001", "철수가 왔다", 0)
    base = translation_key(r, [], PROFILE)
    assert translation_key(r.model_copy(update={"reading_order": 7}), [], PROFILE) == base
    assert translation_key(r.model_copy(update={"text": "영희가 왔다"}), [], PROFILE) != base
    assert translation_key(r.model_copy(update={"kind": "sfx"}), [], PROFILE) != base
    locked = GlossaryEntry(id=1, source="철수", target="Cheolsu", status="locked")
    unrelated = GlossaryEntry(id=2, source="영희", target="Younghee", status="locked")
    assert translation_key(r, [unrelated], PROFILE) == base  # a term this region does not contain
    assert translation_key(r, [locked], PROFILE) != base
    assert translation_key(r, [], PROFILE.model_copy(update={"model": "other"})) != base
    assert translation_key(r, [], PROFILE.model_copy(update={"enabled": False})) == base


def test_a_fixed_bubble_is_the_only_one_translated_again(paths: ChapterPaths) -> None:
    write_ocr(paths, {"r0001": "안녕", "r0002": "반가워", "r0003": "잘가"})
    first = EchoClient("v1")
    translate_chapter(first, paths, PROFILE, [], force=True, reuse=True)
    assert all(c.key for c in run(paths).values())
    write_ocr(paths, {"r0001": "안녕", "r0002": "반가워요", "r0003": "잘가"})  # one bubble fixed by hand
    second = EchoClient("v2")
    _status, result = translate_chapter(second, paths, PROFILE, [], force=True, reuse=True)
    assert len(second.requests) == 1 and second.sent_ids(0) == ["r0002"]
    assert {rid: c.text for rid, c in run(paths).items()} == {
        "r0001": "v1:안녕",
        "r0002": "v2:반가워요",
        "r0003": "v1:잘가",
    }
    assert result is not None and result.usage["reused"] == 2.0
    user = second.requests[0][-1]["content"]
    assert '"english": "v1:안녕"' in user and '"english": "v1:잘가"' in user  # the neighbours as context
    # nothing changed: nothing is sent at all
    third = EchoClient("v3")
    translate_chapter(third, paths, PROFILE, [], force=True, reuse=True)
    assert third.requests == []


def test_a_glossary_change_retranslates_the_regions_it_touches(paths: ChapterPaths) -> None:
    write_ocr(paths, {"r0001": "철수가 왔다", "r0002": "반가워"})
    translate_chapter(EchoClient("v1"), paths, PROFILE, [], force=True, reuse=True)
    locked = [GlossaryEntry(id=1, source="철수", target="Cheolsu", status="locked")]
    second = EchoClient("v2")
    translate_chapter(second, paths, PROFILE, locked, force=True, reuse=True)
    assert second.sent_ids(0) == ["r0001"]


def test_without_reuse_or_keys_everything_is_translated_again(paths: ChapterPaths) -> None:
    write_ocr(paths, {"r0001": "안녕", "r0002": "반가워"})
    translate_chapter(EchoClient("v1"), paths, PROFILE, [], force=True, reuse=True)
    full = EchoClient("v2")
    translate_chapter(full, paths, PROFILE, [], force=True)  # `omniscan translate --force`: a fresh run
    assert full.sent_ids(0) == ["r0001", "r0002"]
    legacy = CandidateRun(
        run_id="p", profile="p", model="m", candidates=[Candidate(region_id="r0001", text="old")]
    )  # written before keys existed
    legacy.save(paths.artifact("translations/p.json"))
    again = EchoClient("v3")
    translate_chapter(again, paths, PROFILE, [], force=True, reuse=True)
    assert again.sent_ids(0) == ["r0001", "r0002"]


def write_runs(paths: ChapterPaths, a: dict[str, str], b: dict[str, str]) -> None:
    for run_id, texts in (("runA", a), ("runB", b)):
        CandidateRun(
            run_id=run_id,
            profile=run_id,
            model="m",
            candidates=[Candidate(region_id=rid, text=text) for rid, text in texts.items()],
        ).save(paths.artifact(f"translations/{run_id}.json"))


def test_the_judge_only_judges_regions_whose_candidates_changed(paths: ChapterPaths) -> None:
    write_ocr(paths, {"r0001": "안녕", "r0002": "반가워"})
    write_runs(paths, {"r0001": "Hi", "r0002": "Nice"}, {"r0001": "Hello", "r0002": "Glad"})
    first = EchoClient()
    _s, artifact, stats = judge_chapter(first, paths, JUDGE, [], force=True, reuse=True)
    assert len(first.requests) == 1 and stats is not None and stats.reused == 0
    assert artifact is not None and all(line.key for line in artifact.lines)
    again = EchoClient()
    _s, artifact2, stats2 = judge_chapter(again, paths, JUDGE, [], force=True, reuse=True)
    assert again.requests == [] and stats2 is not None and (stats2.reused, stats2.regions) == (2, 2)
    assert artifact2 is not None and [x.text for x in artifact2.lines] == [x.text for x in artifact.lines]
    write_runs(paths, {"r0001": "Hi", "r0002": "Nice to see you"}, {"r0001": "Hello", "r0002": "Glad"})
    third = EchoClient()
    _s, artifact3, stats3 = judge_chapter(third, paths, JUDGE, [], force=True, reuse=True)
    assert len(third.requests) == 1 and '"id": "r0002"' in third.requests[0][-1]["content"]
    assert '"id": "r0001"' not in third.requests[0][-1]["content"]
    assert stats3 is not None and stats3.reused == 1
    assert artifact3 is not None and [x.region_id for x in artifact3.lines] == ["r0001", "r0002"]


def test_a_line_the_judge_failed_on_is_judged_again(paths: ChapterPaths) -> None:
    write_ocr(paths, {"r0001": "안녕"})
    write_runs(paths, {"r0001": "Hi"}, {"r0001": "Hello"})
    judge_chapter(EchoClient(), paths, JUDGE, [], force=True, reuse=True)
    auto = FinalArtifact.load(paths.artifact("final_auto.json"))
    failed = auto.model_copy(update={"lines": [auto.lines[0].model_copy(update={"flags": ["judge_failed"]})]})
    failed.save(paths.artifact("final_auto.json"))
    again = EchoClient()
    judge_chapter(again, paths, JUDGE, [], force=True, reuse=True)
    assert len(again.requests) == 1


def test_judge_key_follows_candidates_and_judge_settings() -> None:
    r = region("r0001", "안녕", 0)
    runs = {"a": {"r0001": "Hi"}, "b": {"r0001": "Hello"}}
    base = judge_key(r, runs, [], JUDGE)
    assert judge_key(r, {"b": {"r0001": "Hello"}, "a": {"r0001": "Hi"}}, [], JUDGE) == base  # order-free
    assert judge_key(r, {**runs, "a": {"r0001": "Hey"}}, [], JUDGE) != base
    assert judge_key(r, runs, [], JUDGE.model_copy(update={"model": "other"})) != base

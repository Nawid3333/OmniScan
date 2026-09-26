"""Tests for omniscan.translate.judge_chapter — hand-built chapter artifacts, fake client."""

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
    ChapterEdits,
    FinalArtifact,
    Region,
    RegionKind,
    RegionsArtifact,
    TranslationEdit,
)
from omniscan.llm.ollama import ChatResponse
from omniscan.translate.judge_chapter import judge_chapter
from omniscan.translate.judge_config import JudgeConfig

CFG = JudgeConfig(model="judge-model")


def region(rid: str, text: str, *, kind: RegionKind = "bubble_text") -> Region:
    return Region(id=rid, slice_index=0, kind=kind, bbox=BBox(x0=0, y0=0, x1=10, y1=10), text=text)


def write_ocr(paths: ChapterPaths) -> None:
    RegionsArtifact(
        regions=[
            region("r0001", "안녕"),
            region("r0002", "워터마크", kind="watermark"),
            region("r0003", "반가워"),
        ]
    ).save(paths.artifact("ocr.json"))


def write_run(paths: ChapterPaths, run_id: str, texts: dict[str, str]) -> None:
    CandidateRun(
        run_id=run_id,
        profile=run_id,
        model="run-model",
        candidates=[Candidate(region_id=rid, text=text) for rid, text in texts.items()],
    ).save(paths.artifact(f"translations/{run_id}.json"))


def judgements_reply(*judgements: dict[str, str]) -> str:
    return json.dumps({"judgements": list(judgements)}, ensure_ascii=False)


class FakeClient:
    """One scripted reply per chat call; counts calls."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

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
        self.calls += 1
        return ChatResponse(
            content=self.replies.pop(0),
            model=model,
            done=True,
            total_duration_ns=None,
            prompt_eval_count=1,
            eval_count=1,
            raw={},
        )


@pytest.fixture
def paths(tmp_path: Path) -> ChapterPaths:
    return ChapterPaths(
        series="S",
        chapter="Chapter 1",
        raw_dir=tmp_path / "library" / "S" / "Chapter 1",
        work_dir=tmp_path / "work" / "S" / "Chapter 1",
        output_dir=tmp_path / "output" / "S" / "Chapter 1",
        filtered_dir=tmp_path / "output" / "S" / "_filtered" / "Chapter 1",
    )


def test_done_writes_final_json_with_stats(paths: ChapterPaths) -> None:
    write_ocr(paths)
    write_run(paths, "run1", {"r0001": "Hello", "r0003": "Hi there"})
    client = FakeClient([])
    status, artifact, stats = judge_chapter(client, paths, CFG, [])
    assert status == "done"
    assert artifact is not None and stats is not None
    assert (stats.regions, stats.judged, stats.auto_picked, stats.untranslated) == (2, 0, 2, 0)
    assert stats.requests == 0
    loaded = FinalArtifact.load(paths.artifact("final.json"))
    assert loaded.judge_model == "judge-model"
    assert [(line.region_id, line.text, line.sources) for line in loaded.lines] == [
        ("r0001", "Hello", ["run1"]),
        ("r0003", "Hi there", ["run1"]),
    ]


def test_hand_written_lines_survive_a_rerun(paths: ChapterPaths) -> None:
    """edits.json's lines replace the judge's in final.json; the judge's own stay in final_auto.json."""
    write_ocr(paths)
    write_run(paths, "run1", {"r0001": "Hello", "r0003": "Hi there"})
    box = BBox(x0=0, y0=0, x1=10, y1=10)
    ChapterEdits(
        translations=[TranslationEdit(region_id="r0003", anchor=box, text="Nice to see you", source="반가워")]
    ).save(paths.artifact("edits.json"))
    status, artifact, _stats = judge_chapter(FakeClient([]), paths, CFG, [], force=True)
    assert status == "done" and artifact is not None
    final = FinalArtifact.load(paths.artifact("final.json"))
    assert [(line.region_id, line.text, line.decision) for line in final.lines] == [
        ("r0001", "Hello", "pick"),
        ("r0003", "Nice to see you", "manual"),
    ]
    assert artifact.lines == final.lines
    auto = FinalArtifact.load(paths.artifact("final_auto.json"))
    assert [line.text for line in auto.lines] == ["Hello", "Hi there"]


def test_existing_output_is_skipped_without_calls(paths: ChapterPaths) -> None:
    write_ocr(paths)
    write_run(paths, "run1", {"r0001": "Hello"})
    judge_chapter(FakeClient([]), paths, CFG, [])
    client = FakeClient([])
    assert judge_chapter(client, paths, CFG, []) == ("skipped", None, None)
    assert client.calls == 0


def test_force_reruns_and_judges_disagreeing_runs(paths: ChapterPaths) -> None:
    write_ocr(paths)
    write_run(paths, "run1", {"r0001": "Hello", "r0003": "Hi"})
    write_run(paths, "run2", {"r0001": "Goodbye", "r0003": "Hi"})
    judge_chapter(
        FakeClient([judgements_reply({"id": "r0001", "decision": "pick", "pick": "A"})]), paths, CFG, []
    )
    reply = judgements_reply({"id": "r0001", "decision": "pick", "pick": "B"})
    client = FakeClient([reply])
    status, artifact, stats = judge_chapter(client, paths, CFG, [], force=True)
    assert status == "done"
    assert client.calls == 1  # only the disagreeing region is judged
    assert artifact is not None and stats is not None
    assert stats.judged == 1
    assert artifact.lines[0].text == "Goodbye"
    assert artifact.lines[0].sources == ["run2"]
    assert artifact.lines[1].sources == ["run1"]


def test_run_ids_select_a_subset(paths: ChapterPaths) -> None:
    write_ocr(paths)
    write_run(paths, "run1", {"r0001": "Hello", "r0003": "Hi there"})
    write_run(paths, "run2", {"r0001": "Hey", "r0003": "Hello there"})
    client = FakeClient([])
    _status, artifact, _stats = judge_chapter(client, paths, CFG, [], run_ids=["run2"])
    assert artifact is not None
    assert [line.text for line in artifact.lines] == ["Hey", "Hello there"]
    assert [line.sources for line in artifact.lines] == [["run2"], ["run2"]]


def test_unknown_run_id_raises_naming_the_known_ones(paths: ChapterPaths) -> None:
    write_ocr(paths)
    write_run(paths, "run1", {"r0001": "Hello"})
    write_run(paths, "run2", {"r0001": "Hi"})
    with pytest.raises(ValueError, match=r"unknown run 'nope' \(known: run1, run2\)"):
        judge_chapter(FakeClient([]), paths, CFG, [], run_ids=["nope"])


def test_missing_ocr_takes_precedence_over_missing_runs(paths: ChapterPaths) -> None:
    with pytest.raises(FileNotFoundError, match=r"ocr\.json missing"):
        judge_chapter(FakeClient([]), paths, CFG, [])


def test_missing_runs_raise_with_the_hint(paths: ChapterPaths) -> None:
    write_ocr(paths)
    with pytest.raises(FileNotFoundError, match="no translation runs"):
        judge_chapter(FakeClient([]), paths, CFG, [])


def test_dot_prefixed_partial_files_are_not_runs(paths: ChapterPaths) -> None:
    write_ocr(paths)
    CandidateRun(
        run_id="run1",
        profile="run1",
        model="m",
        candidates=[Candidate(region_id="r0001", text="Hello")],
    ).save(paths.artifact("translations/.run1.partial.json"))
    with pytest.raises(FileNotFoundError, match="no translation runs"):
        judge_chapter(FakeClient([]), paths, CFG, [])


def test_empty_candidate_texts_are_dropped(paths: ChapterPaths) -> None:
    write_ocr(paths)
    write_run(paths, "run1", {"r0001": "Hello", "r0003": "   "})
    _status, artifact, stats = judge_chapter(FakeClient([]), paths, CFG, [])
    assert artifact is not None and stats is not None
    assert [(line.region_id, line.decision) for line in artifact.lines] == [
        ("r0001", "pick"),
        ("r0003", "manual"),
    ]
    assert artifact.lines[1].flags == ["untranslated"]
    assert stats.untranslated == 1


class RecordingClient:
    """FakeClient that also records the messages of every chat call."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.messages: list[list[dict[str, Any]]] = []

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
        self.messages.append(messages)
        return ChatResponse(
            content=self.replies.pop(0),
            model=model,
            done=True,
            total_duration_ns=None,
            prompt_eval_count=1,
            eval_count=1,
            raw={},
        )


def disagreeing_runs(paths: ChapterPaths) -> None:
    """Two runs for r0001 (the judge is asked) and one agreeing pair for r0003 (auto-picked)."""
    write_run(paths, "run1", {"r0001": "Hello", "r0003": "Hi"})
    write_run(paths, "run2", {"r0001": "Goodbye"})


def test_judge_chapter_passes_the_story_summary_to_the_model(paths: ChapterPaths) -> None:
    write_ocr(paths)
    disagreeing_runs(paths)
    client = RecordingClient([judgements_reply({"id": "r0001", "decision": "pick", "pick": "A"})])
    judge_chapter(client, paths, CFG, [], story_summary="ctx")
    assert "Story so far:\nctx" in client.messages[0][1]["content"]
    plain = RecordingClient([judgements_reply({"id": "r0001", "decision": "pick", "pick": "A"})])
    judge_chapter(plain, paths, CFG, [], force=True)  # no story summary
    assert "Story so far" not in plain.messages[0][1]["content"]


def test_judge_chapter_records_usage(paths: ChapterPaths) -> None:
    write_ocr(paths)
    disagreeing_runs(paths)
    client = FakeClient([judgements_reply({"id": "r0001", "decision": "pick", "pick": "A"})])
    status, artifact, stats = judge_chapter(client, paths, CFG, [])
    assert status == "done"
    assert artifact is not None and stats is not None
    assert (stats.requests, stats.repair_requests, stats.regions) == (1, 0, 2)
    loaded = FinalArtifact.load(paths.artifact("final.json"))
    assert list(loaded.usage) == [  # the pinned judge usage key set, in order
        "prompt_tokens",
        "completion_tokens",
        "requests",
        "repair_requests",
        "regions",
        "seconds",
        "rate_limited",
        "reused",  # lines kept from the previous run (judge_chapter reuse=True); summed by `omniscan usage`
    ]
    assert loaded.usage["prompt_tokens"] == 1.0  # FakeClient counts 1 prompt/completion token per call
    assert loaded.usage["completion_tokens"] == 1.0
    assert loaded.usage["requests"] == float(stats.requests)
    assert loaded.usage["repair_requests"] == float(stats.repair_requests)
    assert loaded.usage["regions"] == float(stats.regions)
    assert loaded.usage["seconds"] == pytest.approx(stats.seconds)
    assert loaded.usage["rate_limited"] == 0.0

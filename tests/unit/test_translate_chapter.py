"""Tests for omniscan.translate.chapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import BBox, Candidate, CandidateRun, Region, RegionsArtifact
from omniscan.llm.ollama import ChatResponse
from omniscan.translate.chapter import translate_chapter
from omniscan.translate.profiles import TranslationProfile

MODEL = "test-model"


def region(rid: str, text: str) -> Region:
    return Region(id=rid, slice_index=0, kind="bubble_text", bbox=BBox(x0=0, y0=0, x1=10, y1=10), text=text)


def write_ocr(paths: ChapterPaths) -> None:
    RegionsArtifact(regions=[region("r0001", "안녕"), region("r0002", "반가워")]).save(
        paths.artifact("ocr.json")
    )


def profile(**overrides: object) -> TranslationProfile:
    fields: dict[str, object] = {
        "name": "test-profile",
        "endpoint": "local",
        "model": MODEL,
        "style": "chat_json",
    }
    fields.update(overrides)
    return TranslationProfile(**fields)  # type: ignore[arg-type]


class FakeClient:
    """One scripted reply per chat call; records nothing else."""

    def __init__(self, replies: list[str | Exception]) -> None:
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
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResponse(
            content=reply,
            model=model,
            done=True,
            total_duration_ns=None,
            prompt_eval_count=1,
            eval_count=1,
            raw={},
        )


def json_reply(texts: dict[str, str]) -> str:
    return json.dumps(
        {"translations": [{"id": rid, "text": text} for rid, text in texts.items()]}, ensure_ascii=False
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


def test_translate_chapter_writes_run_and_removes_partial(paths: ChapterPaths) -> None:
    write_ocr(paths)
    client = FakeClient([json_reply({"r0001": "Hello", "r0002": "Hi"})])
    status, run = translate_chapter(client, paths, profile(), [])
    assert status == "done"
    assert run is not None
    output = paths.artifact("translations/test-profile.json")
    loaded = CandidateRun.load(output)
    assert loaded.run_id == "test-profile"
    assert loaded.profile == "test-profile"
    assert loaded.model == MODEL
    assert [c.region_id for c in loaded.candidates] == ["r0001", "r0002"]
    assert not paths.artifact("translations/.test-profile.partial.json").exists()

    client = FakeClient([])
    assert translate_chapter(client, paths, profile(), []) == ("skipped", None)
    assert client.calls == 0


def test_translate_chapter_force_reruns_and_overwrites(paths: ChapterPaths) -> None:
    write_ocr(paths)
    client = FakeClient([json_reply({"r0001": "First", "r0002": "run"})])
    translate_chapter(client, paths, profile(), [])
    client = FakeClient([json_reply({"r0001": "Second", "r0002": "run"})])
    status, run = translate_chapter(client, paths, profile(), [], force=True)
    assert status == "done"
    assert run is not None and [c.text for c in run.candidates] == ["Second", "run"]
    assert CandidateRun.load(paths.artifact("translations/test-profile.json")).candidates[0].text == "Second"


def test_translate_chapter_missing_ocr_raises(paths: ChapterPaths) -> None:
    client = FakeClient([])
    with pytest.raises(FileNotFoundError, match=r"ocr\.json missing"):
        translate_chapter(client, paths, profile(), [])
    assert client.calls == 0


def test_leftover_partial_is_used_when_not_forced(paths: ChapterPaths) -> None:
    write_ocr(paths)
    partial = paths.artifact("translations/.test-profile.partial.json")
    CandidateRun(
        run_id="test-profile",
        profile="test-profile",
        model=MODEL,
        candidates=[Candidate(region_id="r0001", text="kept")],
    ).save(partial)
    client = FakeClient([json_reply({"r0002": "Hi"})])
    status, run = translate_chapter(client, paths, profile(), [])
    assert status == "done"
    assert run is not None and [c.text for c in run.candidates] == ["kept", "Hi"]
    assert not partial.exists()  # removed after the run


def test_leftover_partial_is_ignored_and_removed_when_forced(paths: ChapterPaths) -> None:
    write_ocr(paths)
    partial = paths.artifact("translations/.test-profile.partial.json")
    CandidateRun(
        run_id="test-profile",
        profile="test-profile",
        model=MODEL,
        candidates=[Candidate(region_id="r0001", text="stale")],
    ).save(partial)
    client = FakeClient([json_reply({"r0001": "fresh", "r0002": "run"})])
    status, run = translate_chapter(client, paths, profile(), [], force=True)
    assert status == "done"
    assert run is not None and [c.text for c in run.candidates] == ["fresh", "run"]
    assert client.calls == 1  # both regions requested again
    assert not partial.exists()


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


def test_translate_chapter_passes_the_story_summary_to_the_model(paths: ChapterPaths) -> None:
    write_ocr(paths)
    client = RecordingClient([json_reply({"r0001": "Hello", "r0002": "Hi"})])
    translate_chapter(client, paths, profile(), [], story_summary="ctx")
    assert "Story so far:\nctx" in client.messages[0][1]["content"]
    plain = RecordingClient([json_reply({"r0001": "Hello", "r0002": "Hi"})])
    translate_chapter(plain, paths, profile(), [], force=True)  # no story summary
    assert "Story so far" not in plain.messages[0][1]["content"]

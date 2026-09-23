"""Tests for omniscan.translate.run — everything through a fake chat client, no network."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from omniscan.core.schemas import BBox, Candidate, CandidateRun, GlossaryEntry, Region
from omniscan.llm.ollama import ChatResponse, OllamaRateLimitError
from omniscan.translate.profiles import TranslationProfile
from omniscan.translate.prompts import (
    TRANSLATIONS_SCHEMA,
    source_text,
    substitute_binding,
    translategemma_prompt,
    translategemma_template,
)
from omniscan.translate.run import run_profile

MODEL = "test-model"


def region(
    rid: str,
    *,
    text: str = "",
    slice_index: int = 0,
    reading_order: int = 0,
    kind: str = "bubble_text",
    lang: str = "ko",
) -> Region:
    return Region(
        id=rid,
        slice_index=slice_index,
        kind=kind,  # type: ignore[arg-type]
        bbox=BBox(x0=0, y0=0, x1=10, y1=10),
        reading_order=reading_order,
        text=text,
        lang=lang,  # type: ignore[arg-type]
    )


def make_regions(n: int) -> list[Region]:
    """r0001..r{n:04d} in reading order."""
    return [region(f"r{i:04d}", text=f"문장 {i}") for i in range(1, n + 1)]


def profile(**overrides: object) -> TranslationProfile:
    fields: dict[str, object] = {
        "name": "test-profile",
        "endpoint": "local",
        "model": MODEL,
        "style": "chat_json",
    }
    fields.update(overrides)
    return TranslationProfile(**fields)  # type: ignore[arg-type]


def entry(eid: int, source: str, target: str) -> GlossaryEntry:
    return GlossaryEntry(id=eid, source=source, target=target, status="locked")  # type: ignore[arg-type]


def json_reply(ids: list[str]) -> str:
    return json.dumps({"translations": [{"id": rid, "text": f"T {rid}"} for rid in ids]}, ensure_ascii=False)


class FakeClient:
    """Records every chat call and replays scripted replies (str, (str, in, out) or an exception)."""

    def __init__(self, replies: list[str | tuple[str, int | None, int | None] | Exception]) -> None:
        self.replies: list[str | tuple[str, int | None, int | None] | Exception] = list(replies)
        self.calls: list[dict[str, Any]] = []
        self.on_chat: Callable[[], None] | None = None  # runs after the call is recorded, before replying

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
            {
                "model": model,
                "messages": messages,
                "cloud": cloud,
                "format": format,
                "options": options,
                "keep_alive": keep_alive,
                "think": think,
            }
        )
        if self.on_chat is not None:
            self.on_chat()
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content, prompt_count, eval_count = reply if isinstance(reply, tuple) else (reply, 10, 5)
        return ChatResponse(
            content=content,
            model=model,
            done=True,
            total_duration_ns=None,
            prompt_eval_count=prompt_count,
            eval_count=eval_count,
            raw={},
        )


def region_ids(call: dict[str, Any]) -> list[str]:
    """The ids listed in the 'Regions' part of a chat_json user message."""
    user = call["messages"][1]["content"]
    regions_part = user.split("Regions (reading order):\n", 1)[1]
    return [item["id"] for item in json.loads(regions_part)]


# ---------------------------------------------------------------- chat_json


def test_chat_json_chunks_of_chunk_regions() -> None:
    regions = make_regions(70)
    client = FakeClient(
        [
            json_reply([f"r{i:04d}" for i in range(1, 31)]),
            json_reply([f"r{i:04d}" for i in range(31, 61)]),
            json_reply([f"r{i:04d}" for i in range(61, 71)]),
        ]
    )
    run = run_profile(client, profile(), regions, [])
    assert [len(region_ids(call)) for call in client.calls] == [30, 30, 10]
    assert [c.region_id for c in run.candidates] == [f"r{i:04d}" for i in range(1, 71)]
    assert [c.text for c in run.candidates] == [f"T r{i:04d}" for i in range(1, 71)]


def test_chat_json_request_kwargs() -> None:
    regions = make_regions(2)
    client = FakeClient([json_reply(["r0001", "r0002"])])
    run_profile(client, profile(think=False), regions, [])
    call = client.calls[0]
    assert call["model"] == MODEL
    assert call["cloud"] is False
    assert call["format"] == TRANSLATIONS_SCHEMA
    assert call["options"] == {"temperature": 0.3}
    assert call["think"] is False

    client = FakeClient([json_reply(["r0001", "r0002"])])
    run_profile(client, profile(endpoint="cloud", think=None), regions, [])
    call = client.calls[0]
    assert call["cloud"] is True
    assert call["think"] is None


def test_chat_json_repair_requests_only_missing_and_merges() -> None:
    regions = make_regions(4)
    partial_reply = json.dumps(
        {"translations": [{"id": "r0001", "text": "T r0001"}, {"id": "r0002", "text": "T r0002"}]},
        ensure_ascii=False,
    )
    client = FakeClient([partial_reply, json_reply(["r0003", "r0004"])])
    run = run_profile(client, profile(), regions, [])
    assert len(client.calls) == 2
    assert region_ids(client.calls[1]) == ["r0003", "r0004"]
    assert [c.region_id for c in run.candidates] == ["r0001", "r0002", "r0003", "r0004"]
    assert run.usage["repair_requests"] == 1.0
    assert run.usage["requests"] == 2.0
    assert run.usage["missing"] == 0.0


def test_chat_json_ids_never_returned_are_missing_without_error() -> None:
    regions = make_regions(3)
    client = FakeClient([json_reply(["r0002"]), json_reply(["r0002"]), json_reply(["r0002"])])
    run = run_profile(client, profile(), regions, [], max_repair_rounds=2)
    assert len(client.calls) == 3  # 1 request + 2 repair rounds
    assert [c.region_id for c in run.candidates] == ["r0002"]
    assert run.usage["missing"] == 2.0
    assert run.usage["repair_requests"] == 2.0


def test_chat_json_usage_sums_with_injected_clock() -> None:
    regions = make_regions(2)
    # The first reply omits r0002, so the second request is a repair round.
    client = FakeClient([((json_reply(["r0001"])), 7, 3), ((json_reply(["r0002"])), None, None)])
    ticks = iter([100.0, 105.5])  # start, end (intermediate saves do not consume ticks)
    run = run_profile(client, profile(), regions, [], clock=lambda: next(ticks))
    assert run.usage["prompt_tokens"] == 7.0  # a None count is treated as 0
    assert run.usage["completion_tokens"] == 3.0
    assert run.usage["requests"] == 2.0
    assert run.usage["repair_requests"] == 1.0
    assert run.usage["regions"] == 2.0
    assert run.usage["missing"] == 0.0
    assert run.usage["seconds"] == pytest.approx(5.5)


def test_partial_with_two_candidates_skips_those_regions(tmp_path: Path) -> None:
    partial = tmp_path / "partial.json"
    regions = make_regions(3)
    CandidateRun(
        run_id="test-profile",
        profile="test-profile",
        model=MODEL,
        candidates=[Candidate(region_id="r0002", text="kept2"), Candidate(region_id="r0003", text="kept3")],
    ).save(partial)
    client = FakeClient([json_reply(["r0001"])])
    run = run_profile(client, profile(), regions, [], partial_path=partial)
    assert [c.region_id for c in run.candidates] == ["r0001", "r0002", "r0003"]
    assert [c.text for c in run.candidates] == ["T r0001", "kept2", "kept3"]
    assert len(client.calls) == 1
    assert region_ids(client.calls[0]) == ["r0001"]


def test_partial_of_another_model_is_ignored(tmp_path: Path) -> None:
    partial = tmp_path / "partial.json"
    regions = make_regions(2)
    CandidateRun(
        run_id="test-profile",
        profile="test-profile",
        model="other-model",
        candidates=[Candidate(region_id="r0001", text="stale")],
    ).save(partial)
    client = FakeClient([json_reply(["r0001", "r0002"])])
    run = run_profile(client, profile(), regions, [], partial_path=partial)
    assert [c.text for c in run.candidates] == ["T r0001", "T r0002"]
    assert len(client.calls) == 1


def test_partial_is_rewritten_after_every_chunk(tmp_path: Path) -> None:
    partial = tmp_path / "partial.json"
    regions = make_regions(70)
    client = FakeClient(
        [
            json_reply([f"r{i:04d}" for i in range(1, 31)]),
            json_reply([f"r{i:04d}" for i in range(31, 61)]),
            json_reply([f"r{i:04d}" for i in range(61, 71)]),
        ]
    )
    seen: list[int] = []

    def peek() -> None:
        if partial.is_file():
            seen.append(len(CandidateRun.load(partial).candidates))

    client.on_chat = peek
    run_profile(client, profile(), regions, [], partial_path=partial)
    # peek runs before each reply: no file before chunk 1, then 30 and 60 candidates saved so far.
    assert seen == [30, 60]


def test_rate_limit_on_second_chunk_propagates_and_partial_has_first_chunk(tmp_path: Path) -> None:
    partial = tmp_path / "partial.json"
    regions = make_regions(60)
    client = FakeClient(
        [json_reply([f"r{i:04d}" for i in range(1, 31)]), OllamaRateLimitError("429", status_code=429)]
    )
    with pytest.raises(OllamaRateLimitError):
        run_profile(client, profile(), regions, [], partial_path=partial)
    saved = CandidateRun.load(partial)
    assert [c.region_id for c in saved.candidates] == [f"r{i:04d}" for i in range(1, 31)]


# ---------------------------------------------------------------- translategemma


def test_translategemma_one_request_per_region_no_format() -> None:
    regions = make_regions(2)
    entries = [entry(1, "게이트", "Gate")]
    client = FakeClient(["Hello there", "Stop right there"])
    run = run_profile(client, profile(style="translategemma"), regions, entries)
    assert len(client.calls) == 2
    for call, reg in zip(client.calls, regions, strict=True):
        expected = translategemma_prompt(substitute_binding(source_text(reg), entries, reg.lang))
        assert call["messages"] == [{"role": "user", "content": expected}]
        assert call["format"] is None
        assert call["think"] is None
        assert call["cloud"] is False
    assert [c.text for c in run.candidates] == ["Hello there", "Stop right there"]


def test_translategemma_unwraps_quotes_and_skips_empty() -> None:
    regions = [
        region("r0001", text="너 여기 있구나"),
        region("r0002", text='"인용된 말"'),
        region("r0003", text="한 마디"),
        region("r0004", text="빈 응답"),
    ]
    client = FakeClient(['"You\'re here"', '"Quoted reply"', '"', ""])
    run = run_profile(client, profile(style="translategemma"), regions, [])
    assert [c.text for c in run.candidates] == ["You're here", '"Quoted reply"', '"']


def test_translategemma_sends_the_region_language_template() -> None:
    regions = [region("r0001", text="こんにちは", lang="ja")]
    client = FakeClient(["Hello"])
    run_profile(client, profile(style="translategemma"), regions, [])
    expected = translategemma_template("ja") + source_text(regions[0])  # acceptance 5
    assert client.calls[0]["messages"] == [{"role": "user", "content": expected}]


def test_translategemma_partial_saved_after_20_regions(tmp_path: Path) -> None:
    partial = tmp_path / "partial.json"
    regions = make_regions(20)
    client = FakeClient([f"Reply {i}" for i in range(1, 21)])
    run_profile(client, profile(style="translategemma"), regions, [], partial_path=partial)
    saved = CandidateRun.load(partial)
    assert [c.region_id for c in saved.candidates] == [f"r{i:04d}" for i in range(1, 21)]


@pytest.mark.parametrize("style", ["chat_json", "translategemma"])
def test_regions_without_letters_pass_through_unsent(style: str) -> None:
    regions = [region("r0001", text="!"), region("r0002", text="안녕"), region("r0003", text="……\n?!")]
    reply = json_reply(["r0002"]) if style == "chat_json" else "Hello"
    client = FakeClient([reply])
    run = run_profile(client, profile(style=style), regions, [])
    assert len(client.calls) == 1
    sent = client.calls[0]["messages"][-1]["content"]
    assert "안녕" in sent and "r0001" not in sent and "r0003" not in sent and "……" not in sent
    texts = {c.region_id: c.text for c in run.candidates}
    assert texts["r0001"] == "!"
    assert texts["r0003"] == "…… ?!"  # the whitespace-collapsed source text
    assert texts["r0002"] in ("T r0002", "Hello")
    assert run.usage["missing"] == 0.0


def test_a_region_with_any_letter_is_still_sent() -> None:
    client = FakeClient(["Pa!"])
    run = run_profile(client, profile(style="translategemma"), [region("r0001", text="파!")], [])
    assert len(client.calls) == 1
    assert run.candidates[0].text == "Pa!"

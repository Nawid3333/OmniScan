"""Tests for omniscan.translate.judge — everything through a fake chat client, no network."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from omniscan.core.schemas import BBox, FinalLine, GlossaryEntry, Region, RegionKind
from omniscan.llm.ollama import ChatResponse, OllamaRateLimitError
from omniscan.translate.judge import judge_regions
from omniscan.translate.judge_config import JudgeConfig
from omniscan.translate.judge_prompts import JUDGE_SCHEMA, JudgeItem, judge_messages

MODEL = "judge-model"


def region(rid: str, text: str, *, kind: RegionKind = "bubble_text") -> Region:
    return Region(id=rid, slice_index=0, kind=kind, bbox=BBox(x0=0, y0=0, x1=10, y1=10), text=text)


def make_regions(n: int) -> list[Region]:
    """r0001..r{n:04d} in reading order."""
    return [region(f"r{i:04d}", f"문장 {i}") for i in range(1, n + 1)]


def entry(eid: int, source: str, target: str) -> GlossaryEntry:
    return GlossaryEntry(id=eid, source=source, target=target, status="locked", type="person")  # type: ignore[arg-type]


def config(**overrides: object) -> JudgeConfig:
    fields: dict[str, object] = {"model": MODEL}
    fields.update(overrides)
    return JudgeConfig(**fields)  # type: ignore[arg-type]


def judgement(
    rid: str, *, decision: str = "pick", pick: str = "A", text: str = "", rationale: Any = ""
) -> dict[str, Any]:
    return {"id": rid, "decision": decision, "pick": pick, "text": text, "rationale": rationale}


def reply(*judgements: dict[str, Any]) -> str:
    return json.dumps({"judgements": list(judgements)}, ensure_ascii=False)


class FakeClient:
    """Records every chat call and replays scripted replies (str, (str, in, out) or an exception)."""

    def __init__(self, replies: Sequence[str | tuple[str, int | None, int | None] | Exception]) -> None:
        self.replies: list[str | tuple[str, int | None, int | None] | Exception] = list(replies)
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
        popped = self.replies.pop(0)
        if isinstance(popped, Exception):
            raise popped
        content, prompt_count, eval_count = popped if isinstance(popped, tuple) else (popped, 10, 5)
        return ChatResponse(
            content=content,
            model=model,
            done=True,
            total_duration_ns=None,
            prompt_eval_count=prompt_count,
            eval_count=eval_count,
            raw={},
        )


def request_ids(call: dict[str, Any]) -> list[str]:
    """The ids listed in the 'Regions' part of a judge user message."""
    user = call["messages"][1]["content"]
    return [item["id"] for item in json.loads(user.split("Regions (reading order):\n", 1)[1])]


def request_payload(call: dict[str, Any]) -> list[dict[str, Any]]:
    """The parsed region dicts of a judge user message."""
    user = call["messages"][1]["content"]
    return json.loads(user.split("Regions (reading order):\n", 1)[1])


# ---------------------------------------------------------------- deterministic paths


def test_agreeing_candidates_are_auto_picked_by_priority() -> None:
    regions = [region("r0001", "꺼져")]
    runs = {"a": {"r0001": "Get out!"}, "b": {"r0001": "get out"}}
    lines, stats = judge_regions(FakeClient([]), config(prefer=["b"]), regions, runs, [])
    assert lines == [
        FinalLine(
            region_id="r0001", text="get out", decision="pick", sources=["b"], rationale="candidates agree"
        )
    ]
    assert stats.requests == 0
    assert stats.auto_picked == 1
    assert stats.judged == 0


def test_single_run_rationale_is_single_candidate() -> None:
    regions = [region("r0001", "안녕")]
    lines, stats = judge_regions(FakeClient([]), config(), regions, {"a": {"r0001": "Hello"}}, [])
    assert lines[0].sources == ["a"]
    assert lines[0].rationale == "single candidate"
    assert stats.requests == 0


def test_region_without_any_candidate_is_manual_and_untranslated() -> None:
    regions = [region("r0001", "안녕"), region("r0002", "워터마크", kind="watermark"), region("r0003", "   ")]
    lines, stats = judge_regions(FakeClient([]), config(), regions, {}, [])
    assert lines == [
        FinalLine(
            region_id="r0001", text="", decision="manual", rationale="no candidate", flags=["untranslated"]
        )
    ]
    assert (stats.regions, stats.untranslated, stats.auto_picked, stats.judged) == (1, 1, 0, 0)


def test_run_missing_a_region_leaves_it_untranslated() -> None:
    regions = [region("r0001", "안녕"), region("r0002", "반가워")]
    lines, stats = judge_regions(FakeClient([]), config(), regions, {"a": {"r0001": "Hello"}}, [])
    assert [line.region_id for line in lines] == ["r0001", "r0002"]
    assert lines[0].rationale == "single candidate"
    assert lines[1].decision == "manual" and lines[1].flags == ["untranslated"]
    assert stats.untranslated == 1


def test_agreeing_candidates_that_all_violate_are_judged() -> None:
    regions = [region("r0001", "성진이가 간다")]
    entries = [entry(1, "성진", "Seong-jin")]
    runs = {"a": {"r0001": "He goes"}, "b": {"r0001": "He goes."}}
    client = FakeClient([reply(judgement("r0001", pick="A"))])
    lines, stats = judge_regions(client, config(max_repair_rounds=0), regions, runs, entries)
    assert len(client.calls) == 1  # judged: no candidate satisfies the locked term
    assert lines[0].text == "He goes" and lines[0].sources == ["a"]
    assert lines[0].flags == ["glossary_violation"]
    assert stats.violations_left == 1


def test_agreeing_candidates_second_satisfying_is_picked_without_request() -> None:
    regions = [region("r0001", "성진이가 간다")]
    entries = [entry(1, "성진", "Seong-jin")]
    runs = {"a": {"r0001": "Seong jin goes"}, "b": {"r0001": "Seong-jin goes"}}
    client = FakeClient([])
    lines, _stats = judge_regions(client, config(), regions, runs, entries)
    assert client.calls == []
    assert lines[0].text == "Seong-jin goes" and lines[0].sources == ["b"]
    assert lines[0].flags == []


def test_priority_alphabetical_without_prefer() -> None:
    regions = [region("r0001", "안녕")]
    runs = {"b": {"r0001": "Hello there!"}, "a": {"r0001": "Hello there"}}
    lines, _ = judge_regions(FakeClient([]), config(), regions, runs, [])
    assert lines[0].sources == ["a"] and lines[0].text == "Hello there"


def test_priority_prefer_listed_runs_first() -> None:
    regions = [region("r0001", "안녕")]
    runs = {"a": {"r0001": "Hello there"}, "b": {"r0001": "Hello there!"}}
    lines, _ = judge_regions(FakeClient([]), config(prefer=["b", "a"]), regions, runs, [])
    assert lines[0].sources == ["b"] and lines[0].text == "Hello there!"


def test_identical_normalised_candidates_are_shown_once() -> None:
    regions = [region("r0001", "안녕")]
    runs = {"r1": {"r0001": "Go away"}, "r2": {"r0001": "Leave now"}, "r3": {"r0001": "GO AWAY"}}
    client = FakeClient([reply(judgement("r0001", pick="B"))])
    lines, stats = judge_regions(client, config(always_judge=True), regions, runs, [])
    payload = request_payload(client.calls[0])
    assert payload[0]["candidates"] == {"A": "Go away", "B": "Leave now"}
    assert lines[0].text == "Leave now" and lines[0].sources == ["r2"]
    assert stats.requests == 1


def test_always_judge_sends_agreeing_regions_with_two_unique() -> None:
    regions = [region("r0001", "안녕")]
    runs = {"a": {"r0001": "Get out"}, "b": {"r0001": "Get outt"}}  # similar enough to agree
    client = FakeClient([reply(judgement("r0001", pick="A"))])
    lines, stats = judge_regions(client, config(always_judge=True), regions, runs, [])
    assert len(client.calls) == 1
    assert lines[0].sources == ["a"]
    assert stats.judged == 1
    without = FakeClient([])
    lines, stats = judge_regions(without, config(), regions, runs, [])
    assert without.calls == []  # without always_judge the agreeing pair is auto-picked
    assert lines[0].sources == ["a"]
    assert stats.auto_picked == 1


# ---------------------------------------------------------------- requests


def test_judged_request_messages_and_kwargs() -> None:
    regions = [region("r0001", "안녕")]
    runs = {"a": {"r0001": "Get out!"}, "b": {"r0001": "Leave now"}}
    entries = [entry(1, "성진", "Seong-jin")]  # no hit in this region -> no glossary sections
    client = FakeClient([reply(judgement("r0001", pick="A"))])
    judge_regions(client, config(), regions, runs, entries)
    items = [JudgeItem(region=regions[0], candidates={"A": "Get out!", "B": "Leave now"})]
    assert client.calls[0]["messages"] == judge_messages(items, entries)
    call = client.calls[0]
    assert call["model"] == MODEL
    assert call["cloud"] is False
    assert call["format"] == JUDGE_SCHEMA
    assert call["options"] == {"temperature": 0.2}
    assert call["think"] is False

    cloud_client = FakeClient([reply(judgement("r0001", pick="A"))])
    judge_regions(cloud_client, config(endpoint="cloud"), regions, runs, entries)
    assert cloud_client.calls[0]["cloud"] is True


def test_pick_merge_rewrite_resolutions() -> None:
    regions = [region("r0001", "첫째"), region("r0002", "둘째"), region("r0003", "셋째")]
    runs = {
        "a": {"r0001": "Hello", "r0002": "What is this", "r0003": "Stop it"},
        "b": {"r0001": "Goodbye", "r0002": "Who is that", "r0003": "Cut it out"},
    }
    client = FakeClient(
        [
            reply(
                judgement("r0001", pick="B", rationale="closer"),
                judgement("r0002", decision="merge", text="  What\n is   that?  ", rationale="combined"),
                judgement("r0003", decision="rewrite", text="Back off!", rationale="both wrong"),
            )
        ]
    )
    lines, stats = judge_regions(client, config(), regions, runs, [])
    assert lines[0].decision == "pick"
    assert (lines[0].text, lines[0].sources) == ("Goodbye", ["b"])
    assert lines[1].decision == "merge"
    assert (lines[1].text, lines[1].sources) == ("What is that?", ["a", "b"])
    assert lines[2].decision == "rewrite"
    assert (lines[2].text, lines[2].sources) == ("Back off!", [])
    assert [line.rationale for line in lines] == ["closer", "combined", "both wrong"]
    assert stats.judged == 3 and stats.requests == 1


def test_rationale_collapsed_cut_to_300_and_non_string_becomes_empty() -> None:
    regions = [region("r0001", "안녕"), region("r0002", "끝")]
    runs = {"a": {"r0001": "Hello", "r0002": "The end"}, "b": {"r0001": "Goodbye", "r0002": "It is over"}}
    rationale = "  word\n  " + "y" * 400  # whitespace-collapsed, then cut
    client = FakeClient(
        [reply(judgement("r0001", pick="A", rationale=rationale), judgement("r0002", pick="A", rationale=7))]
    )
    lines, _ = judge_regions(client, config(), regions, runs, [])
    assert lines[0].rationale == "word " + "y" * 295
    assert len(lines[0].rationale) == 300
    assert lines[1].rationale == ""


def test_chunking_of_judged_regions() -> None:
    regions = make_regions(45)
    runs = {
        "a": {f"r{i:04d}": f"Yes {i}" for i in range(1, 46)},
        "b": {f"r{i:04d}": f"No {i}" for i in range(1, 46)},
    }
    replies = [
        reply(*[judgement(f"r{i:04d}", pick="A") for i in range(1, 21)]),
        reply(*[judgement(f"r{i:04d}", pick="A") for i in range(21, 41)]),
        reply(*[judgement(f"r{i:04d}", pick="A") for i in range(41, 46)]),
    ]
    client = FakeClient(replies)
    lines, stats = judge_regions(client, config(chunk_regions=20), regions, runs, [])
    assert [request_ids(call) for call in client.calls] == [
        [f"r{i:04d}" for i in range(1, 21)],
        [f"r{i:04d}" for i in range(21, 41)],
        [f"r{i:04d}" for i in range(41, 46)],
    ]
    assert [line.text for line in lines] == [f"Yes {i}" for i in range(1, 46)]
    assert (stats.requests, stats.judged) == (3, 45)


def test_reply_tolerates_junk_unknown_ids_and_duplicate_ids_first_wins() -> None:
    regions = [region("r0001", "안녕")]
    runs = {"a": {"r0001": "Hello"}, "b": {"r0001": "Goodbye"}}
    content = reply(
        "junk",  # type: ignore[arg-type]
        judgement("r9999", pick="A"),
        judgement("r0001", pick="B"),
        judgement("r0001", pick="A"),
    )
    client = FakeClient([content])
    lines, stats = judge_regions(client, config(), regions, runs, [])
    assert lines[0].text == "Goodbye" and lines[0].sources == ["b"]
    assert (stats.requests, stats.repair_requests) == (1, 0)


def test_unresolved_items_get_one_repair_round_and_recover() -> None:
    regions = make_regions(4)
    runs = {
        "a": {f"r{i:04d}": f"Yes {i}" for i in range(1, 5)},
        "b": {f"r{i:04d}": f"No {i}" for i in range(1, 5)},
    }
    client = FakeClient(
        [
            reply(
                judgement("r0002", pick="Z"),  # unknown label
                judgement("r0003", decision="invent"),  # unknown decision
                judgement("r0004", decision="merge", text="   "),  # empty text
            ),
            reply(*[judgement(f"r{i:04d}", pick="A") for i in range(1, 5)]),
        ]
    )
    lines, stats = judge_regions(client, config(), regions, runs, [])
    assert len(client.calls) == 2
    assert request_ids(client.calls[1]) == ["r0001", "r0002", "r0003", "r0004"]
    payload = request_payload(client.calls[1])
    for item in payload:
        assert item["previous"] == ""
        assert item["problems"] == ["the previous answer was invalid or missing"]
    assert all("previous" not in item for item in request_payload(client.calls[0]))
    assert [line.text for line in lines] == [f"Yes {i}" for i in range(1, 5)]
    assert all(line.flags == [] for line in lines)
    assert (stats.requests, stats.repair_requests) == (2, 1)


def test_violating_resolution_gets_repair_with_problems_and_previous() -> None:
    regions = [region("r0001", "성진이가 간다")]
    entries = [entry(1, "성진", "Seong-jin")]
    runs = {"a": {"r0001": "He goes"}, "b": {"r0001": "Away he goes"}}
    client = FakeClient(
        [
            reply(judgement("r0001", decision="rewrite", text="He goes", rationale="shorter")),
            reply(judgement("r0001", decision="rewrite", text="Seong-jin goes", rationale="fixed")),
        ]
    )
    lines, stats = judge_regions(client, config(), regions, runs, entries)
    assert len(client.calls) == 2
    payload = request_payload(client.calls[1])
    assert payload[0]["previous"] == "He goes"
    assert payload[0]["problems"] == ["missing binding term: 성진 -> Seong-jin"]
    assert lines[0].decision == "rewrite"
    assert (lines[0].text, lines[0].sources, lines[0].flags) == ("Seong-jin goes", [], [])
    assert stats.violations_left == 0
    assert stats.repair_requests == 1


def test_violation_surviving_the_repair_round_is_kept_and_counted() -> None:
    regions = [region("r0001", "성진이가 간다")]
    entries = [entry(1, "성진", "Seong-jin")]
    runs = {"a": {"r0001": "He goes"}, "b": {"r0001": "Away he goes"}}
    client = FakeClient(
        [
            reply(judgement("r0001", decision="rewrite", text="He goes")),
            reply(judgement("r0001", decision="rewrite", text="He goes again")),
        ]
    )
    lines, stats = judge_regions(client, config(), regions, runs, entries)
    assert len(client.calls) == 2  # 1 request + exactly one repair round
    assert lines[0].text == "He goes again"
    assert lines[0].flags == ["glossary_violation"]
    assert stats.violations_left == 1


def test_max_repair_rounds_zero_makes_no_repair_requests() -> None:
    regions = [region("r0001", "성진이가 간다")]
    entries = [entry(1, "성진", "Seong-jin")]
    runs = {"a": {"r0001": "He goes"}, "b": {"r0001": "Away he goes"}}
    client = FakeClient([reply()])
    lines, stats = judge_regions(client, config(max_repair_rounds=0), regions, runs, entries)
    assert len(client.calls) == 1
    assert stats.repair_requests == 0
    # nothing resolved: the deterministic fallback keeps the first candidate, unrepaired
    assert (lines[0].text, lines[0].sources) == ("He goes", ["a"])
    assert lines[0].flags == ["judge_failed", "glossary_violation"]


def test_repair_items_are_chunked_like_the_first_round() -> None:
    regions = make_regions(45)
    runs = {
        "a": {f"r{i:04d}": f"Yes {i}" for i in range(1, 46)},
        "b": {f"r{i:04d}": f"No {i}" for i in range(1, 46)},
    }
    client = FakeClient([reply()] * 6)  # every round leaves everything unresolved
    lines, stats = judge_regions(client, config(chunk_regions=20), regions, runs, [])
    assert [len(request_ids(call)) for call in client.calls] == [20, 20, 5, 20, 20, 5]
    assert (stats.requests, stats.repair_requests) == (6, 3)
    assert all(line.decision == "pick" and line.rationale == "judge failed" for line in lines)


def test_still_unresolved_falls_back_to_first_clean_candidate() -> None:
    regions = [region("r0001", "성진이가 간다")]
    entries = [entry(1, "성진", "Seong-jin")]
    runs = {"a": {"r0001": "He goes"}, "b": {"r0001": "Seong-jin goes"}}
    client = FakeClient([reply(), reply()])
    lines, stats = judge_regions(client, config(), regions, runs, entries)
    assert len(client.calls) == 2
    assert lines[0] == FinalLine(
        region_id="r0001",
        text="Seong-jin goes",
        decision="pick",
        sources=["b"],
        rationale="judge failed",
        flags=["judge_failed"],
    )
    assert stats.violations_left == 0


def test_unresolved_fallback_that_violates_gets_both_flags() -> None:
    regions = [region("r0001", "성진이가 간다")]
    entries = [entry(1, "성진", "Seong-jin")]
    runs = {"a": {"r0001": "He goes"}, "b": {"r0001": "Gone already"}}
    client = FakeClient([reply(), reply()])
    lines, stats = judge_regions(client, config(), regions, runs, entries)
    assert lines[0].sources == ["a"]
    assert lines[0].flags == ["judge_failed", "glossary_violation"]
    assert stats.violations_left == 1


def test_usage_sums_and_injected_clock() -> None:
    regions = [region("r0001", "안녕"), region("r0002", "반가워")]
    runs = {
        "a": {"r0001": "Hello", "r0002": "Well met"},
        "b": {"r0001": "Goodbye", "r0002": "See you"},
    }
    client = FakeClient(
        [
            (reply(judgement("r0001", pick="A")), 7, 3),  # r0002 missing -> repair round
            (reply(judgement("r0002", pick="A")), None, None),  # None counts as 0
        ]
    )
    ticks = iter([100.0, 105.5])  # start, end (intermediate calls consume no ticks)
    _lines, stats = judge_regions(client, config(), regions, runs, [], clock=lambda: next(ticks))
    assert stats.prompt_tokens == 7
    assert stats.completion_tokens == 3
    assert (stats.requests, stats.repair_requests) == (2, 1)
    assert (stats.regions, stats.judged, stats.auto_picked, stats.untranslated) == (2, 2, 0, 0)
    assert stats.seconds == pytest.approx(5.5)


def test_stats_count_auto_single_and_manual() -> None:
    regions = [region("r0001", "안녕"), region("r0002", "혼자"), region("r0003", "끝")]
    runs = {"a": {"r0001": "Hello", "r0002": "Alone now", "r0003": ""}, "b": {"r0001": "Hello!"}}
    _lines, stats = judge_regions(FakeClient([]), config(), regions, runs, [])
    assert (stats.regions, stats.judged, stats.auto_picked, stats.untranslated) == (3, 0, 2, 1)
    assert stats.requests == 0


def test_rate_limit_error_propagates_unchanged() -> None:
    regions = [region("r0001", "안녕")]
    runs = {"a": {"r0001": "Hello"}, "b": {"r0001": "Goodbye"}}
    client = FakeClient([OllamaRateLimitError("429", status_code=429)])
    with pytest.raises(OllamaRateLimitError):
        judge_regions(client, config(), regions, runs, [])


def test_default_clock_reports_a_non_negative_seconds() -> None:
    regions = [region("r0001", "안녕")]
    runs = {"a": {"r0001": "Hello"}}
    lines, stats = judge_regions(FakeClient([]), config(), regions, runs, [])
    assert lines[0].text == "Hello"
    assert stats.seconds >= 0.0
    assert stats.requests == 0

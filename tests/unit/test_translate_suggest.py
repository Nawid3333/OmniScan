"""Tests for omniscan.translate.suggest — translating a few regions on demand (fake chat client)."""

from __future__ import annotations

import json
from typing import Any, Literal

import pytest

from omniscan.core.schemas import BBox, GlossaryEntry, MemoryEntry, Region, RegionKind
from omniscan.learn.apply import TranslationHints
from omniscan.llm.ollama import ChatResponse, OllamaRateLimitError
from omniscan.translate.profiles import TranslationProfile
from omniscan.translate.prompts import ContextLine, chat_json_messages, context_lines
from omniscan.translate.suggest import suggest


def region(rid: str, text: str, order: int, *, kind: RegionKind = "bubble_text") -> Region:
    return Region(
        id=rid,
        slice_index=0,
        kind=kind,
        bbox=BBox(x0=0, y0=order * 100, x1=50, y1=order * 100 + 40),
        reading_order=order,
        text=text,
    )


REGIONS = [
    region("r0001", "안녕", 0),
    region("r0002", "워터마크", 1, kind="watermark"),
    region("r0003", "어디 가?", 2),
    region("r0004", "집에.", 3),
    region("r0005", "같이 가자!", 4),
]
ENGLISH = {"r0001": "Hi", "r0003": "Where are you going?", "r0005": "Let's go together!"}


def profile(
    name: str = "cloud",
    *,
    style: Literal["chat_json", "translategemma"] = "chat_json",
    model: str = "gemma4:31b-cloud",
) -> TranslationProfile:
    return TranslationProfile(name=name, endpoint="local", model=model, style=style)


class FakeClient:
    """Answers chat_json with `text:<id>`-style lines; records the messages; can raise a rate limit once."""

    def __init__(self, *, rate_limited_models: tuple[str, ...] = ()) -> None:
        self.messages: list[list[dict[str, Any]]] = []
        self.models: list[str] = []
        self.rate_limited_models = rate_limited_models

    def chat(self, model: str, messages: list[dict[str, Any]], **_: Any) -> ChatResponse:
        self.messages.append(messages)
        self.models.append(model)
        if model in self.rate_limited_models:
            raise OllamaRateLimitError("limit", status_code=429)
        user = messages[-1]["content"]
        if messages[0]["role"] == "system":
            ids = json.loads(user.split("Regions (reading order):\n", 1)[1])
            content = json.dumps(
                {"translations": [{"id": r["id"], "text": f"{model}:{r['id']}"} for r in ids]}
            )
        else:
            content = f"{model}:plain"
        return ChatResponse(
            content=content,
            model=model,
            done=True,
            total_duration_ns=None,
            prompt_eval_count=1,
            eval_count=1,
            raw={},
        )


def test_context_lines_window_around_the_targets_without_the_targets() -> None:
    ordered = [r for r in REGIONS if r.kind != "watermark"]
    lines = context_lines(ordered, {"r0004"}, ENGLISH, before=2, after=1)
    assert lines == [
        ContextLine(id="r0001", text="안녕", english="Hi"),
        ContextLine(id="r0003", text="어디 가?", english="Where are you going?"),
        ContextLine(id="r0005", text="같이 가자!", english="Let's go together!"),
    ]
    assert context_lines(ordered, {"r0001"}, {}, before=2, after=1) == [
        ContextLine(id="r0003", text="어디 가?", english="")
    ]
    assert context_lines(ordered, {"r0009"}, ENGLISH) == []


def test_prompt_has_a_context_section_only_when_asked() -> None:
    targets = [REGIONS[3]]
    plain = chat_json_messages(targets, [])
    with_context = chat_json_messages(
        targets, [], context=[ContextLine(id="r0003", text="어디 가?", english="Where?")]
    )
    assert "Context" not in plain[1]["content"]
    user = with_context[1]["content"]
    assert user.index("Context (the lines around these regions") < user.index("Regions (reading order):")
    assert '"english": "Where?"' in user
    assert with_context[0] == plain[0]  # same system prompt


def test_suggest_returns_every_profiles_reading_with_context_and_glossary() -> None:
    client = FakeClient()
    entries = [GlossaryEntry(id=1, source="집", target="home", status="locked")]
    suggestions = suggest(
        client,
        [profile("a", model="m1"), profile("b", model="m2")],
        REGIONS,
        ["r0004"],
        entries,
        ENGLISH,
        story_summary="- Chapter 1: they met.",
    )
    assert [(s.region_id, s.profile, s.model, s.text) for s in suggestions] == [
        ("r0004", "a", "m1", "m1:r0004"),
        ("r0004", "b", "m2", "m2:r0004"),
    ]
    user = client.messages[0][1]["content"]
    assert "Story so far:\n- Chapter 1: they met." in user
    assert "집 -> home" in user
    assert '"english": "Where are you going?"' in user
    regions_sent = json.loads(user.split("Regions (reading order):\n", 1)[1])
    assert [r["id"] for r in regions_sent] == ["r0004"]


def test_suggest_uses_the_fallback_on_a_rate_limit() -> None:
    client = FakeClient(rate_limited_models=("m1",))
    fallback = profile("local", style="translategemma", model="tg")
    suggestions = suggest(
        client, [profile("a", model="m1")], REGIONS, ["r0004"], [], {}, fallbacks={"a": fallback}
    )
    assert [(s.profile, s.text) for s in suggestions] == [("local", "tg:plain")]
    with pytest.raises(OllamaRateLimitError):
        suggest(
            FakeClient(rate_limited_models=("m1",)), [profile("a", model="m1")], REGIONS, ["r0004"], [], {}
        )


def test_suggest_refuses_unknown_and_untranslatable_regions() -> None:
    with pytest.raises(ValueError, match="r0009"):
        suggest(FakeClient(), [profile()], REGIONS, ["r0009"], [], {})
    with pytest.raises(ValueError, match="watermark"):
        suggest(FakeClient(), [profile()], REGIONS, ["r0002"], [], {})


def test_suggest_asks_the_model_even_for_a_remembered_line() -> None:
    client = FakeClient()
    hints = TranslationHints(
        exact={"집에.": "Home."},
        entries=(
            MemoryEntry(source="집에.", english="Home.", chapter="1"),
            MemoryEntry(source="집에!", english="Home!", chapter="1"),
        ),
        preferences=(),
        examples=5,
    )
    suggestions = suggest(client, [profile()], REGIONS, ["r0004"], [], ENGLISH, hints=hints)
    assert [s.text for s in suggestions] == ["gemma4:31b-cloud:r0004"]
    user = client.messages[0][1]["content"]
    assert "- 집에! => Home!" in user and "=> Home." not in user  # a similar line, never the line itself

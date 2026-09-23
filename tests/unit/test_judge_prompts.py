"""Tests for omniscan.translate.judge_prompts."""

from __future__ import annotations

import json

import pytest

from omniscan.core.schemas import BBox, GlossaryEntry, Region, RegionKind
from omniscan.translate.judge_prompts import (
    JUDGE_SCHEMA,
    JUDGE_SYSTEM,
    JudgeItem,
    judge_messages,
    judge_system,
    label_for,
)

EXPECTED_SYSTEM = """You are the editor of an official English release of a Korean manhwa. For every numbered region you get the Korean source text and one or more candidate English translations labelled A, B, C. Decide per region: "pick" the best candidate unchanged; "merge" to combine the best parts of the candidates into one line; or "rewrite" to write a better line yourself when every candidate is wrong or unnatural. Judge the meaning against the Korean source, not by how many candidates agree. Write natural, idiomatic English suited to comic lettering: concise, in the character's voice, one continuous line without manual line breaks, no translator notes. Keep Korean honorific suffixes and titles romanized when they address a person (-nim, -ssi, hyung, noona, sunbae) unless that reads badly in English. For a region of kind "sfx" give a short English onomatopoeia. Entries under "Glossary (binding)" are mandatory: whenever a source term appears (with or without a particle such as 이/가/은/는/을/를/의), its target must appear in your English exactly as written. If a region has "problems", your earlier answer was rejected: fix exactly those problems (each missing binding target must appear verbatim in your text) and answer again. Answer with JSON only, in exactly this shape: {"judgements":[{"id":"r0001","decision":"pick","pick":"A","text":"","rationale":"short reason"}]} — for "pick" set "pick" to the label and leave "text" empty; for "merge" and "rewrite" set "text" to the final line and "pick" to ""; one entry for every input id, no extra ids, no commentary."""


def region(rid: str, text: str, *, kind: RegionKind = "bubble_text", lang: str = "ko") -> Region:
    return Region(
        id=rid,
        slice_index=0,
        kind=kind,
        bbox=BBox(x0=0, y0=0, x1=10, y1=10),
        text=text,
        lang=lang,  # type: ignore[arg-type]
    )


def entry(eid: int, source: str, target: str, status: str, type_: str = "person") -> GlossaryEntry:
    return GlossaryEntry(id=eid, source=source, target=target, status=status, type=type_)  # type: ignore[arg-type]


def regions_part(user: str) -> list[dict]:
    """The JSON list after the 'Regions (reading order):' heading of a user message."""
    return json.loads(user.split("Regions (reading order):\n", 1)[1])


def test_system_message_is_the_verbatim_card_text() -> None:
    assert JUDGE_SYSTEM == EXPECTED_SYSTEM


def test_schema_matches_the_documented_shape() -> None:
    assert JUDGE_SCHEMA == {
        "type": "object",
        "properties": {
            "judgements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "decision": {"type": "string", "enum": ["pick", "merge", "rewrite"]},
                        "pick": {"type": "string"},
                        "text": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["id", "decision"],
                },
            }
        },
        "required": ["judgements"],
    }


def test_label_for_a_to_z_and_error_beyond() -> None:
    assert label_for(0) == "A"
    assert label_for(1) == "B"
    assert label_for(25) == "Z"
    with pytest.raises(ValueError, match="26 labels"):
        label_for(26)
    with pytest.raises(ValueError):
        label_for(-1)


def test_messages_system_and_no_glossary_sections_without_hits() -> None:
    item = JudgeItem(region=region("r0001", "안녕"), candidates={"A": "Hello"})
    messages = judge_messages([item], [])
    assert messages[0] == {"role": "system", "content": JUDGE_SYSTEM}
    assert messages[1]["role"] == "user"
    user = messages[1]["content"]
    assert user.startswith("Regions (reading order):\n")
    assert "Glossary (binding)" not in user
    assert "Glossary (suggested)" not in user


def test_glossary_sections_only_for_hits_in_entry_order() -> None:
    locked = entry(1, "성진", "Seong-jin", "locked")
    proposed = entry(2, "헌터 협회", "Hunter Association", "proposed", type_="org")
    rejected = entry(3, "게이트", "Gate", "rejected")
    item = JudgeItem(region=region("r0001", "성진이가 헌터 협회에 간다"), candidates={"A": "Seong-jin goes"})
    user = judge_messages([item], [rejected, proposed, locked])[1]["content"]
    assert "Glossary (binding):\n- 성진 -> Seong-jin (person)" in user
    assert "Glossary (suggested):\n- 헌터 협회 -> Hunter Association (org)" in user
    assert "게이트" not in user  # rejected entries are never shown
    parts = user.split("\n\n")
    assert parts[0].startswith("Glossary (binding):")
    assert parts[1].startswith("Glossary (suggested):")
    assert parts[2].startswith("Regions (reading order):")


def test_regions_json_round_trips_with_keys_in_order() -> None:
    item = JudgeItem(region=region("r0001", "안녕\n  반가워"), candidates={"A": "Hello", "B": "Hi there"})
    user = judge_messages([item], [])[1]["content"]
    payload = regions_part(user)
    assert payload == [
        {
            "id": "r0001",
            "kind": "bubble_text",
            "source": "안녕 반가워",
            "candidates": {"A": "Hello", "B": "Hi there"},
        }
    ]
    assert list(payload[0]) == ["id", "kind", "source", "candidates"]
    assert "안녕 반가워" in user  # unescaped Korean in the raw message


def test_previous_and_problems_only_for_repair_items() -> None:
    normal = JudgeItem(region=region("r0001", "안녕"), candidates={"A": "Hello"})
    repair = JudgeItem(
        region=region("r0002", "간다"),
        candidates={"A": "He goes"},
        previous="He goes",
        problems=("missing binding term: 성진 -> Seong-jin",),
    )
    payload = regions_part(judge_messages([normal, repair], [])[1]["content"])
    assert list(payload[0]) == ["id", "kind", "source", "candidates"]
    assert list(payload[1]) == ["id", "kind", "source", "candidates", "previous", "problems"]
    assert payload[1]["previous"] == "He goes"
    assert payload[1]["problems"] == ["missing binding term: 성진 -> Seong-jin"]


def test_items_are_used_in_the_order_given() -> None:
    items = [
        JudgeItem(region=region("r0002", "간다"), candidates={"A": "He goes"}),
        JudgeItem(region=region("r0001", "안녕"), candidates={"A": "Hello"}),
    ]
    payload = regions_part(judge_messages(items, [])[1]["content"])
    assert [entry["id"] for entry in payload] == ["r0002", "r0001"]


def test_empty_items_still_produce_an_empty_regions_list() -> None:
    messages = judge_messages([], [])
    assert messages[1]["content"] == "Regions (reading order):\n[]"


def test_judge_messages_story_summary_is_the_first_part() -> None:
    item = JudgeItem(region=region("r0001", "성진이가 간다"), candidates={"A": "Seong-jin goes"})
    entries = [entry(1, "성진", "Seong-jin", "locked")]
    user = judge_messages([item], entries, story_summary="line 1")[1]["content"]
    parts = user.split("\n\n")
    assert parts[0] == "Story so far:\nline 1"
    assert parts[1].startswith("Glossary (binding):")
    assert parts[2].startswith("Regions (reading order):")


def test_judge_messages_none_and_empty_story_summary_are_identical_to_omitting_it() -> None:
    item = JudgeItem(region=region("r0001", "안녕"), candidates={"A": "Hello"})
    plain = judge_messages([item], [])
    assert judge_messages([item], [], story_summary=None) == plain
    assert judge_messages([item], [], story_summary="") == plain


def test_judge_messages_use_the_items_regions_language() -> None:
    item = JudgeItem(region=region("r0001", "こんにちは", lang="ja"), candidates={"A": "Hello"})
    messages = judge_messages([item], [])
    assert messages[0] == {"role": "system", "content": judge_system("ja")}
    assert messages[0]["content"] != JUDGE_SYSTEM  # acceptance 5: not the Korean prompt

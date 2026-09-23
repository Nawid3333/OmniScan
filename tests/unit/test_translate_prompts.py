"""Tests for omniscan.translate.prompts."""

from __future__ import annotations

import json
from typing import Literal

from omniscan.core.schemas import BBox, GlossaryEntry, Region, TermType
from omniscan.translate.prompts import (
    CHAT_JSON_SYSTEM,
    TRANSLATEGEMMA_TEMPLATE,
    chat_json_messages,
    chat_json_system,
    glossary_subset,
    source_text,
    substitute_binding,
    translatable,
    translategemma_prompt,
)


def region(
    rid: str,
    *,
    kind: Literal["bubble_text", "free_text", "sfx", "watermark"] = "bubble_text",
    text: str = "",
    slice_index: int = 0,
    reading_order: int = 0,
    lang: str = "ko",
) -> Region:
    return Region(
        id=rid,
        slice_index=slice_index,
        kind=kind,
        bbox=BBox(x0=0, y0=0, x1=10, y1=10),
        reading_order=reading_order,
        text=text,
        lang=lang,  # type: ignore[arg-type]
    )


def entry(
    eid: int,
    source: str,
    target: str,
    *,
    status: Literal["proposed", "locked", "rejected"] = "locked",
    aliases: list[str] | None = None,
    type_: TermType = "person",
) -> GlossaryEntry:
    return GlossaryEntry(
        id=eid, source=source, target=target, status=status, aliases=aliases or [], type=type_
    )


def test_source_text_collapses_whitespace() -> None:
    r = region("r1", text="이건 그냥\n   꿈이겠지...?   ")
    assert source_text(r) == "이건 그냥 꿈이겠지...?"


def test_translatable_drops_watermarks_and_empty_sorts_naturally() -> None:
    regions = [
        region("r10", text="두 번째", slice_index=1, reading_order=0),
        region("r2", text="첫 번째", slice_index=0, reading_order=0),
        region("r1", text="물", kind="watermark", slice_index=0, reading_order=1),
        region("r3", text="  \n ", slice_index=0, reading_order=2),
        region("r4", text="쿵!", kind="sfx", slice_index=0, reading_order=3),
        region("r11", text="여기", slice_index=0, reading_order=1),
    ]
    result = translatable(regions)
    assert [r.id for r in result] == ["r2", "r11", "r4", "r10"]
    assert regions[0].id == "r10"  # input not mutated


def test_glossary_subset_hits_aliases_and_particles_excludes_rejected_and_missing() -> None:
    regions = [region("r1", text="성진이가 헌터 협회에 갔다"), region("r2", text="쿵! 소리")]
    entries = [
        entry(1, "성진", "Seong-jin", status="locked"),
        entry(2, "성진우", "Seong-woo", status="locked", aliases=["성진"]),  # alias hit
        entry(3, "협회", "Association", status="proposed"),
        entry(4, "헌터", "Hunter", status="rejected"),
        entry(5, "게이트", "Gate", status="locked"),  # no hit
        entry(6, "쿵", "Thud", status="locked"),
    ]
    subset = glossary_subset(regions, entries)
    assert [e.id for e in subset] == [1, 2, 3, 6]  # 협회 is a proposed entry with a hit inside "헌터 협회"


def test_glossary_subset_japanese_particle_hit() -> None:
    regions = [region("r1", text="ユジンは来た", lang="ja")]
    subset = glossary_subset(regions, [entry(1, "ユジン", "Yujin"), entry(2, "성진", "Seong-jin")])
    assert [e.id for e in subset] == [1]


def test_chat_json_messages_sections_and_regions_list() -> None:
    regions = [region("r0001", text="성진이가\n게이트에"), region("r0002", text="쿵!", kind="sfx")]
    entries = [
        entry(1, "성진", "Seong-jin", status="locked"),
        entry(2, "게이트", "Gate", status="locked"),
        entry(3, "쿵", "Thud", status="proposed"),
        entry(4, "안 나옴", "Never", status="locked"),
    ]
    messages = chat_json_messages(regions, entries)
    assert messages[0] == {"role": "system", "content": CHAT_JSON_SYSTEM}
    assert messages[1]["role"] == "user"
    user = messages[1]["content"]
    assert "Glossary (binding):\n- 성진 -> Seong-jin (person)\n- 게이트 -> Gate (person)" in user
    assert "Glossary (suggested):\n- 쿵 -> Thud (person)" in user
    regions_part = user.split("Regions (reading order):\n", 1)[1]
    parsed = json.loads(regions_part)
    assert parsed == [
        {"id": "r0001", "kind": "bubble_text", "text": "성진이가 게이트에"},
        {"id": "r0002", "kind": "sfx", "text": "쿵!"},
    ]


def test_chat_json_messages_omits_empty_glossary_sections() -> None:
    messages = chat_json_messages(
        [region("r1", text="안녕")], [entry(1, "성진", "Seong-jin", status="locked")]
    )
    user = messages[1]["content"]
    assert "Glossary" not in user
    assert user.startswith("Regions (reading order):\n")


def test_substitute_binding_replaces_locked_terms_only() -> None:
    entries = [
        entry(1, "성진", "Seong-jin", status="locked"),
        entry(2, "게이트", "Gate", status="locked"),
        entry(3, "지훈", "Jihoon", status="proposed"),
        entry(4, "없는말", "Never", status="locked"),
    ]
    text = "성진이가 게이트에 들어간 지"
    assert substitute_binding(text, entries, lang="ko") == "Seong-jin이가 Gate에 들어간 지"
    assert substitute_binding("지훈이가 갔다", entries, lang="ko") == "지훈이가 갔다"  # proposed: untouched
    assert substitute_binding("아무 용어 없는 문장", entries, lang="ko") == "아무 용어 없는 문장"


def test_substitute_binding_japanese_particle_stays_put() -> None:
    entries = [entry(1, "ユジン", "Yujin", status="locked")]
    assert substitute_binding("ユジンは来た", entries, lang="ja") == "Yujinは来た"
    # compound particle: the replacement span must not eat or leave behind part of とは
    assert substitute_binding("ユジンとは違う", entries, lang="ja") == "Yujinとは違う"


def test_translategemma_prompt_appends_text_to_probe_template() -> None:
    assert translategemma_prompt("안녕") == TRANSLATEGEMMA_TEMPLATE + "안녕"
    assert TRANSLATEGEMMA_TEMPLATE.endswith(":\n\n\n")


def test_chat_json_messages_story_summary_is_the_first_part() -> None:
    regions = [region("r0001", text="성진이가 게이트에")]
    entries = [entry(1, "성진", "Seong-jin", status="locked")]
    user = chat_json_messages(regions, entries, story_summary="line 1")[1]["content"]
    parts = user.split("\n\n")
    assert parts[0] == "Story so far:\nline 1"
    assert parts[1].startswith("Glossary (binding):")
    assert parts[2].startswith("Regions (reading order):")


def test_chat_json_messages_none_and_empty_story_summary_are_identical_to_omitting_it() -> None:
    regions = [region("r0001", text="성진이가 게이트에"), region("r0002", text="쿵!", kind="sfx")]
    entries = [entry(1, "성진", "Seong-jin", status="locked")]
    plain = chat_json_messages(regions, entries)
    assert chat_json_messages(regions, entries, story_summary=None) == plain
    assert chat_json_messages(regions, entries, story_summary="") == plain


def test_chat_json_messages_regions_language_selects_the_system_prompt() -> None:
    messages = chat_json_messages([region("r0001", text="你好", lang="zh")], [])
    assert messages[0] == {"role": "system", "content": chat_json_system("zh")}
    assert messages[0]["content"] != CHAT_JSON_SYSTEM  # acceptance 5: not the Korean prompt

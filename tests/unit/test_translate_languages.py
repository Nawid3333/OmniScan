"""Tests for omniscan.translate.languages and the byte-identical Korean prompts of every module."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import BBox, Region, RegionsArtifact
from omniscan.glossary.proposal_prompts import PROPOSAL_SYSTEM, proposal_system
from omniscan.glossary.reference_prompts import EXTRACT_SYSTEM, extract_system
from omniscan.story.prompts import SUMMARY_SYSTEM, summary_system
from omniscan.translate.judge_prompts import JUDGE_SYSTEM, judge_system
from omniscan.translate.languages import (
    SourceLanguage,
    chapter_language,
    region_language,
    source_language,
)
from omniscan.translate.prompts import (
    CHAT_JSON_SYSTEM,
    TRANSLATEGEMMA_TEMPLATE,
    chat_json_system,
    translategemma_template,
)

# Literal copies of the prompts as they were before this card (`git show main:<file>`): the "ko"
# renderings are tuned and measured (docs/benchmarks/translation-probe.md) and must not change.
KOREAN_CHAT_JSON_SYSTEM = 'You are the translator for an official English release of a Korean manhwa. Translate every numbered region from Korean into natural, idiomatic English suited to comic lettering: concise, in the character\'s voice, with no translator notes. Write each translation as one continuous line without manual line breaks (the letterer re-wraps it). Keep Korean honorific suffixes and titles romanized when they address a person (-nim, -ssi, hyung, noona, sunbae) unless that reads badly in English. For a region of kind "sfx" give a short English onomatopoeia. Entries under "Glossary (binding)" are mandatory: whenever a source term appears (with or without a particle such as 이/가/은/는/을/를/의), its target must appear in your English exactly as written. Entries under "Glossary (suggested)" are preferred spellings but not mandatory. Answer with JSON only, in exactly this shape: {"translations":[{"id":"r0001","text":"..."}]} — one entry for every input id, no extra ids, no commentary.'

KOREAN_TRANSLATEGEMMA_TEMPLATE = "You are a professional Korean (ko) to English (en) translator. Your goal is to accurately convey the meaning and nuances of the original Korean text while adhering to English grammar, vocabulary, and cultural sensitivities.\nProduce only the English translation, without any additional explanations or commentary. Please translate the following Korean text into English:\n\n\n"

KOREAN_JUDGE_SYSTEM = 'You are the editor of an official English release of a Korean manhwa. For every numbered region you get the Korean source text and one or more candidate English translations labelled A, B, C. Decide per region: "pick" the best candidate unchanged; "merge" to combine the best parts of the candidates into one line; or "rewrite" to write a better line yourself when every candidate is wrong or unnatural. Judge the meaning against the Korean source, not by how many candidates agree. Write natural, idiomatic English suited to comic lettering: concise, in the character\'s voice, one continuous line without manual line breaks, no translator notes. Keep Korean honorific suffixes and titles romanized when they address a person (-nim, -ssi, hyung, noona, sunbae) unless that reads badly in English. For a region of kind "sfx" give a short English onomatopoeia. Entries under "Glossary (binding)" are mandatory: whenever a source term appears (with or without a particle such as 이/가/은/는/을/를/의), its target must appear in your English exactly as written. If a region has "problems", your earlier answer was rejected: fix exactly those problems (each missing binding target must appear verbatim in your text) and answer again. Answer with JSON only, in exactly this shape: {"judgements":[{"id":"r0001","decision":"pick","pick":"A","text":"","rationale":"short reason"}]} — for "pick" set "pick" to the label and leave "text" empty; for "merge" and "rewrite" set "text" to the final line and "pick" to ""; one entry for every input id, no extra ids, no commentary.'

KOREAN_SUMMARY_SYSTEM = 'You are the continuity writer for an official English release of a Korean manhwa. You get one chapter\'s translated text as its lines in reading order. Write a short English summary of what happens in this chapter: 3 to 5 sentences, third person, present tense, naming the characters and places involved and what changed in this chapter only — nothing about earlier chapters, no meta-commentary about the art or the lettering, no chapter numbers. Answer with JSON only, in exactly this shape: {"summary":"..."} — the summary text and nothing else.'

KOREAN_PROPOSAL_SYSTEM = 'You are building the glossary of an English release of a Korean manhwa from its raw Korean chapters — no official translation exists. You get one chapter\'s text as its lines in reading order. List every term a glossary should bind: character names, place names, organisations, ranks and titles, recurring items, skills or techniques. For each term give the Korean source exactly as it appears in a line with any attached particle (이, 가, 은, 는, 을, 를, 의, 도, 아, 야, 에, 에게, 에서) removed; and your own best English rendering — romanise names as the release would spell them, translate everything else as the release would. Give one type: person, place, org, skill, item, rank, title, honorific, sfx or other. A term is one to a few syllables, never a whole sentence or phrase. If you saw the same term more than once, list it once. Answer with JSON only, in exactly this shape: {"terms":[{"source":"민준","target":"Minjun","type":"person"}]} — one entry per distinct term, no commentary.'

KOREAN_EXTRACT_SYSTEM = 'You are building the glossary of an official English release of a Korean manhwa from its own translated chapters. You get one chapter\'s text as paired lines: \'source\' is the Korean original, \'target\' is the official English translation of that same line. List every term a glossary should bind: character names, place names, organisations, ranks and titles, recurring items, skills or techniques. For each term give the Korean source exactly as it appears in a source line with any attached particle (이, 가, 은, 는, 을, 를, 의, 도, 아, 야, 에, 에게, 에서) removed; the English target exactly as the paired target lines spell it; and one type: person, place, org, skill, item, rank, title, honorific, sfx or other. Only list terms whose English you can actually see in the target lines — never invent or re-romanise a translation. A term is one to a few syllables, never a whole sentence or phrase. If you saw the same term with different English spellings, list the spelling that appeared most. Answer with JSON only, in exactly this shape: {"terms":[{"source":"민준","target":"Minjun","type":"person"}]} — one entry per distinct term, no commentary.'

SYSTEMS: dict[str, Callable[[str], str]] = {
    "chat_json": chat_json_system,
    "judge": judge_system,
    "summary": summary_system,
    "proposal": proposal_system,
    "extract": extract_system,
}


def region(rid: str, lang: str) -> Region:
    return Region(
        id=rid,
        slice_index=0,
        kind="bubble_text",
        bbox=BBox(x0=0, y0=0, x1=10, y1=10),
        lang=lang,  # type: ignore[arg-type]
    )


def paths(tmp_path: Path) -> ChapterPaths:
    return ChapterPaths(
        series="S",
        chapter="c",
        raw_dir=tmp_path / "r",
        work_dir=tmp_path / "w",
        output_dir=tmp_path / "o",
        filtered_dir=tmp_path / "o" / "f",
    )


# ---------------------------------------------------------------- Korean pins (acceptance 1)


def test_korean_prompts_are_byte_identical_to_the_pre_card_text() -> None:
    assert chat_json_system("ko") == KOREAN_CHAT_JSON_SYSTEM
    assert CHAT_JSON_SYSTEM == KOREAN_CHAT_JSON_SYSTEM
    assert translategemma_template("ko") == KOREAN_TRANSLATEGEMMA_TEMPLATE
    assert TRANSLATEGEMMA_TEMPLATE == KOREAN_TRANSLATEGEMMA_TEMPLATE
    assert judge_system("ko") == KOREAN_JUDGE_SYSTEM
    assert JUDGE_SYSTEM == KOREAN_JUDGE_SYSTEM
    assert summary_system("ko") == KOREAN_SUMMARY_SYSTEM
    assert SUMMARY_SYSTEM == KOREAN_SUMMARY_SYSTEM
    assert proposal_system("ko") == KOREAN_PROPOSAL_SYSTEM
    assert PROPOSAL_SYSTEM == KOREAN_PROPOSAL_SYSTEM
    assert extract_system("ko") == KOREAN_EXTRACT_SYSTEM
    assert EXTRACT_SYSTEM == KOREAN_EXTRACT_SYSTEM


# ---------------------------------------------------------------- ja/zh renderings (acceptance 2)


@pytest.mark.parametrize("code", ["ja", "zh"])
@pytest.mark.parametrize("name", list(SYSTEMS), ids=list(SYSTEMS))
def test_system_prompt_names_its_language(name: str, code: str) -> None:
    sl = source_language(code)
    text = SYSTEMS[name](code)
    assert sl.name in text and sl.work in text
    assert "Korean" not in text and "manhwa" not in text and "이/가" not in text
    assert "  " not in text  # no double space where the honorifics sentence or hint was removed


def test_ja_has_honorifics_sentence_and_zh_has_none() -> None:
    for build in SYSTEMS.values():
        # only chat_json and judge carry the honorifics sentence in the Korean original
        if "romanized when they address a person" in build("ko"):
            assert "-san" in build("ja")
        assert "romanized when they address a person" not in build("zh")


@pytest.mark.parametrize("code", ["ja", "zh"])
def test_translategemma_template_names_the_language(code: str) -> None:
    sl = source_language(code)
    template = translategemma_template(code)
    assert f"professional {sl.name} ({sl.code}) to English (en) translator" in template
    assert f"the following {sl.name} text into English" in template
    assert "Korean" not in template and "  " not in template


# ---------------------------------------------------------------- SourceLanguage table


def test_source_language_values_follow_the_card_definitions() -> None:
    ko, ja = source_language("ko"), source_language("ja")
    zh, en = source_language("zh"), source_language("en")
    assert all(isinstance(sl, SourceLanguage) for sl in (ko, ja, zh, en))
    assert (ko.code, ko.name, ko.work) == ("ko", "Korean", "manhwa")
    assert ko.honorifics.endswith("unless that reads badly in English. ")
    assert ko.particle_hint == " (with or without a particle such as 이/가/은/는/을/를/의)"
    assert (ja.code, ja.name, ja.work) == ("ja", "Japanese", "manga")
    assert ja.honorifics.endswith("unless that reads badly in English. ") and "-san" in ja.honorifics
    assert ja.particle_hint == " (with or without a particle such as は/が/を/に/の)"
    assert (zh.code, zh.name, zh.work) == ("zh", "Chinese", "manhua")
    assert (en.code, en.name, en.work) == ("en", "English", "comic")
    assert zh.honorifics == zh.particle_hint == en.honorifics == en.particle_hint == ""


def test_source_language_unknown_code_raises() -> None:
    with pytest.raises(ValueError, match="unknown source language"):
        source_language("fr")


# ---------------------------------------------------------------- region_language (acceptance 3)


def test_region_language_majority_wins() -> None:
    regions = [region("r1", "ko"), region("r2", "ja"), region("r3", "ja")]
    assert region_language(regions) == "ja"


def test_region_language_tie_goes_to_the_first_region() -> None:
    regions = [region("r1", "ja"), region("r2", "ko"), region("r3", "ko"), region("r4", "ja")]
    assert region_language(regions) == "ja"


def test_region_language_empty_is_ko() -> None:
    assert region_language([]) == "ko"


# ---------------------------------------------------------------- chapter_language (acceptance 4)


def test_chapter_language_reads_the_ocr_json_regions(tmp_path: Path) -> None:
    cp = paths(tmp_path)
    RegionsArtifact(regions=[region("r1", "ja"), region("r2", "ja")]).save(cp.artifact("ocr.json"))
    assert chapter_language(cp) == "ja"


def test_chapter_language_missing_ocr_json_is_ko(tmp_path: Path) -> None:
    assert chapter_language(paths(tmp_path)) == "ko"

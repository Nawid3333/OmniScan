"""Prompt building for the translation runner: chat_json batching and translategemma per-region prompts.

Evidence for every choice lives in docs/benchmarks/translation-probe.md: source text is sent with
collapsed whitespace (cloud models leak source line breaks), glossary entries are split into binding
(proposed/locked) sections, and translategemma gets locked terms pre-substituted because it ignores
prompt glossaries. Prompts are functions of the regions' source language (`languages.py`); the "ko"
renderings are the tuned originals.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from omniscan.core.paths import natural_key
from omniscan.core.schemas import GlossaryEntry, Region
from omniscan.glossary.match import find_terms
from omniscan.translate.languages import region_language, source_language


def chat_json_system(lang: str) -> str:
    """The chat_json system prompt for `lang`'s source language; "ko" is the tuned original."""
    sl = source_language(lang)
    return (
        f"You are the translator for an official English release of a {sl.name} {sl.work}. Translate "
        f"every numbered region from {sl.name} into natural, idiomatic English suited to comic "
        "lettering: concise, in the character's voice, with no translator notes. Write each "
        "translation as one continuous line without manual line breaks (the letterer re-wraps it). "
        f'{sl.honorifics}For a region of kind "sfx" (a sound effect) give the English sound effect an '
        "official release would letter: one short, punchy onomatopoeia such as BOOM, THUD, WHOOSH or "
        "BA-DUMP, never a description of the sound; a repeated sound stays repeated. Entries under "
        '"Glossary (binding)" are mandatory: whenever a source term appears'
        f"{sl.particle_hint}, its target must appear in your English exactly as written. Entries "
        'under "Glossary (suggested)" are preferred spellings but not mandatory. Answer with JSON '
        "only, "
        'in exactly this shape: {"translations":[{"id":"r0001","text":"..."}]} — one entry for every '
        "input id, no extra ids, no commentary."
    )


CHAT_JSON_SYSTEM = chat_json_system("ko")

TRANSLATIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["id", "text"],
            },
        }
    },
    "required": ["translations"],
}


# From scripts/probe_translation.py (tg_style) — translategemma's own training template, ending with
# three newlines before the text.
def translategemma_template(lang: str) -> str:
    """The translategemma training template naming `lang`'s source language; "ko" is the original."""
    sl = source_language(lang)
    return (
        f"You are a professional {sl.name} ({sl.code}) to English (en) translator. Your goal is to "
        "accurately convey the meaning and nuances of the original "
        f"{sl.name} text while adhering to English grammar, vocabulary, and cultural sensitivities.\n"
        "Produce only the English translation, without any additional explanations or commentary. "
        f"Please translate the following {sl.name} text into English:\n\n\n"
    )


TRANSLATEGEMMA_TEMPLATE = translategemma_template("ko")


def source_text(region: Region) -> str:
    """The region's text with every run of whitespace (newlines included) collapsed to one space."""
    return " ".join(region.text.split())


def translatable(regions: Sequence[Region]) -> list[Region]:
    """The regions worth translating (no watermarks, non-empty text) in reading order."""
    targets = [r for r in regions if r.kind != "watermark" and source_text(r)]
    return sorted(targets, key=lambda r: (r.slice_index, r.reading_order, natural_key(r.id)))


def glossary_subset(regions: Sequence[Region], entries: Sequence[GlossaryEntry]) -> list[GlossaryEntry]:
    """Locked/proposed entries whose source term or an alias occurs in any region's source text."""
    lang = region_language(regions)
    texts = [source_text(r) for r in regions]
    subset: list[GlossaryEntry] = []
    for entry in entries:
        if entry.status == "rejected":
            continue
        if any(find_terms(text, [entry], lang) for text in texts):
            subset.append(entry)
    return subset


@dataclass(frozen=True, slots=True)
class ContextLine:
    """A neighbouring line shown to the model for reference when only a few regions are translated."""

    id: str
    text: str  # source text
    english: str  # its current English line ("" when it has none yet)


def chat_json_messages(
    regions: Sequence[Region],
    entries: Sequence[GlossaryEntry],
    *,
    story_summary: str | None = None,
    context: Sequence[ContextLine] = (),
) -> list[dict[str, str]]:
    """The chat_json prompt: system message plus story context, glossary sections and the regions list.

    `context` (translating a few regions on demand) adds the neighbouring lines, source and English, before
    the regions list; without it the prompt is exactly the whole-chapter one."""
    subset = glossary_subset(regions, entries)
    parts: list[str] = []
    if story_summary:
        parts.append(f"Story so far:\n{story_summary}")
    for title, status in (("Glossary (binding)", "locked"), ("Glossary (suggested)", "proposed")):
        section = [e for e in subset if e.status == status]
        if section:
            lines = "\n".join(f"- {e.source} -> {e.target} ({e.type})" for e in section)
            parts.append(f"{title}:\n{lines}")
    if context:
        context_json = json.dumps(
            [{"id": line.id, "text": line.text, "english": line.english} for line in context],
            ensure_ascii=False,
            indent=1,
        )
        parts.append(
            "Context (the lines around these regions, already lettered; for reference only, do not "
            f"translate them):\n{context_json}"
        )
    regions_json = json.dumps(
        [{"id": r.id, "kind": r.kind, "text": source_text(r)} for r in regions],
        ensure_ascii=False,
        indent=1,
    )
    parts.append(f"Regions (reading order):\n{regions_json}")
    return [
        {"role": "system", "content": chat_json_system(region_language(regions))},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def substitute_binding(text: str, entries: Sequence[GlossaryEntry], lang: str) -> str:
    """Replace locked glossary terms with their targets (right-to-left); particles stay untouched.

    `lang` selects the particle set as in `find_terms`.
    """
    locked = [e for e in entries if e.status == "locked"]
    if not locked:
        return text
    by_id = {e.id: e for e in locked}
    result = text
    for match in reversed(find_terms(result, locked, lang)):
        entry = by_id[match.entry_id]
        result = result[: match.start] + entry.target + result[match.start + len(match.source) :]
    return result


def translategemma_prompt(text: str, lang: str = "ko") -> str:
    """The one-region translategemma prompt (its own template, glossary pre-substituted upstream)."""
    return translategemma_template(lang) + text

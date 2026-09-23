"""Prompt building for the translation runner: chat_json batching and translategemma per-region prompts.

Evidence for every choice lives in docs/benchmarks/translation-probe.md: source text is sent with
collapsed whitespace (cloud models leak source line breaks), glossary entries are split into binding
(proposed/locked) sections, and translategemma gets locked terms pre-substituted because it ignores
prompt glossaries.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from omniscan.core.paths import natural_key
from omniscan.core.schemas import GlossaryEntry, Region
from omniscan.glossary.match import find_terms

CHAT_JSON_SYSTEM = (
    "You are the translator for an official English release of a Korean manhwa. Translate every "
    "numbered region from Korean into natural, idiomatic English suited to comic lettering: concise, "
    "in the character's voice, with no translator notes. Write each translation as one continuous "
    "line without manual line breaks (the letterer re-wraps it). Keep Korean honorific suffixes and "
    "titles romanized when they address a person (-nim, -ssi, hyung, noona, sunbae) unless that reads "
    'badly in English. For a region of kind "sfx" give a short English onomatopoeia. Entries under '
    '"Glossary (binding)" are mandatory: whenever a source term appears (with or without a particle '
    "such as 이/가/은/는/을/를/의), its target must appear in your English exactly as written. Entries "
    'under "Glossary (suggested)" are preferred spellings but not mandatory. Answer with JSON only, '
    'in exactly this shape: {"translations":[{"id":"r0001","text":"..."}]} — one entry for every '
    "input id, no extra ids, no commentary."
)

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
TRANSLATEGEMMA_TEMPLATE = (
    "You are a professional Korean (ko) to English (en) translator. Your goal is to accurately convey "
    "the meaning and nuances of the original Korean text while adhering to English grammar, vocabulary, "
    "and cultural sensitivities.\nProduce only the English translation, without any additional "
    "explanations or commentary. Please translate the following Korean text into English:\n\n\n"
)


def source_text(region: Region) -> str:
    """The region's text with every run of whitespace (newlines included) collapsed to one space."""
    return " ".join(region.text.split())


def translatable(regions: Sequence[Region]) -> list[Region]:
    """The regions worth translating (no watermarks, non-empty text) in reading order."""
    targets = [r for r in regions if r.kind != "watermark" and source_text(r)]
    return sorted(targets, key=lambda r: (r.slice_index, r.reading_order, natural_key(r.id)))


def glossary_subset(regions: Sequence[Region], entries: Sequence[GlossaryEntry]) -> list[GlossaryEntry]:
    """Locked/proposed entries whose source term or an alias occurs in any region's source text."""
    texts = [source_text(r) for r in regions]
    subset: list[GlossaryEntry] = []
    for entry in entries:
        if entry.status == "rejected":
            continue
        if any(find_terms(text, [entry]) for text in texts):
            subset.append(entry)
    return subset


def chat_json_messages(
    regions: Sequence[Region], entries: Sequence[GlossaryEntry], *, story_summary: str | None = None
) -> list[dict[str, str]]:
    """The chat_json prompt: system message plus story context, glossary sections and the regions list."""
    subset = glossary_subset(regions, entries)
    parts: list[str] = []
    if story_summary:
        parts.append(f"Story so far:\n{story_summary}")
    for title, status in (("Glossary (binding)", "locked"), ("Glossary (suggested)", "proposed")):
        section = [e for e in subset if e.status == status]
        if section:
            lines = "\n".join(f"- {e.source} -> {e.target} ({e.type})" for e in section)
            parts.append(f"{title}:\n{lines}")
    regions_json = json.dumps(
        [{"id": r.id, "kind": r.kind, "text": source_text(r)} for r in regions],
        ensure_ascii=False,
        indent=1,
    )
    parts.append(f"Regions (reading order):\n{regions_json}")
    return [
        {"role": "system", "content": CHAT_JSON_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def substitute_binding(text: str, entries: Sequence[GlossaryEntry]) -> str:
    """Replace locked glossary terms with their targets (right-to-left); particles stay untouched."""
    locked = [e for e in entries if e.status == "locked"]
    if not locked:
        return text
    by_id = {e.id: e for e in locked}
    result = text
    for match in reversed(find_terms(result, locked)):
        entry = by_id[match.entry_id]
        result = result[: match.start] + entry.target + result[match.start + len(match.source) :]
    return result


def translategemma_prompt(text: str) -> str:
    """The one-region translategemma prompt (its own template, glossary pre-substituted upstream)."""
    return TRANSLATEGEMMA_TEMPLATE + text

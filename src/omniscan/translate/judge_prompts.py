"""Prompt building for the judge: system instructions, the answer schema and per-chunk user messages.

The judge sees only what it must — the glossary entries that actually occur in the chunk's regions,
each region's Korean source and its labelled candidates — to keep prompts small. Repair rounds add
the rejected answer and the problems found with it. Every choice follows the translation-probe
evidence (docs/benchmarks/translation-probe.md): tolerant parsing upstream, `think` controllable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from omniscan.core.schemas import GlossaryEntry, Region
from omniscan.translate.languages import region_language, source_language
from omniscan.translate.prompts import glossary_subset, source_text


def judge_system(lang: str) -> str:
    """The judge system prompt for `lang`'s source language; "ko" is the tuned original."""
    sl = source_language(lang)
    return (
        f"You are the editor of an official English release of a {sl.name} {sl.work}. For every "
        f"numbered region you get the {sl.name} source text and one or more candidate English "
        'translations labelled A, B, C. Decide per region: "pick" the best candidate unchanged; '
        '"merge" to combine the best parts of the candidates into one line; or "rewrite" to write '
        "a better line yourself when every candidate is wrong or unnatural. Judge the meaning "
        f"against the {sl.name} source, not by how many candidates agree. Write natural, idiomatic "
        "English suited to comic lettering: concise, in the character's voice, one continuous line "
        f"without manual line breaks, no translator notes. {sl.honorifics}For a region of "
        'kind "sfx" (a sound effect) give the English sound effect an official release would letter: '
        "one short, punchy onomatopoeia such as BOOM, THUD, WHOOSH or BA-DUMP, never a description of "
        'the sound; a repeated sound stays repeated. Entries under "Glossary (binding)" are '
        "mandatory: whenever a source term appears"
        f"{sl.particle_hint}, its target must appear in your English exactly as written. If a region "
        'has "problems", your earlier answer was rejected: fix exactly those problems (each missing '
        "binding target must appear verbatim in your text) and answer again. Answer with JSON only, in "
        'exactly this shape: {"judgements":[{"id":"r0001","decision":"pick","pick":"A",'
        '"text":"","rationale":"short reason"}]} — for "pick" set "pick" to the label and '
        'leave "text" empty; for "merge" and "rewrite" set "text" to the final line and '
        '"pick" to ""; one entry for every input id, no extra ids, no commentary.'
    )


JUDGE_SYSTEM = judge_system("ko")

JUDGE_SCHEMA: dict[str, Any] = {
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


@dataclass(frozen=True, slots=True)
class JudgeItem:
    """One region shown to the judge model: the Korean source and the labelled candidates."""

    region: Region
    candidates: dict[str, str]  # label ("A", "B", ...) -> text, in label order
    previous: str | None = None  # repair rounds: the rejected final line
    problems: tuple[str, ...] = ()  # repair rounds: what was wrong with it


def label_for(index: int) -> str:
    """The candidate label at `index`: 0 -> "A", 25 -> "Z"; more than 26 candidates raise."""
    if not 0 <= index < 26:
        raise ValueError(f"candidate index {index} out of range (26 labels A..Z)")
    return chr(ord("A") + index)


def judge_messages(
    items: Sequence[JudgeItem], entries: Sequence[GlossaryEntry], *, story_summary: str | None = None
) -> list[dict[str, str]]:
    """The judge prompt: system message plus story context, glossary sections and the regions list."""
    subset = glossary_subset([item.region for item in items], entries)
    parts: list[str] = []
    if story_summary:
        parts.append(f"Story so far:\n{story_summary}")
    for title, status in (("Glossary (binding)", "locked"), ("Glossary (suggested)", "proposed")):
        section = [e for e in subset if e.status == status]
        if section:
            lines = "\n".join(f"- {e.source} -> {e.target} ({e.type})" for e in section)
            parts.append(f"{title}:\n{lines}")
    regions_json = json.dumps([_region_dict(item) for item in items], ensure_ascii=False, indent=1)
    parts.append(f"Regions (reading order):\n{regions_json}")
    return [
        {"role": "system", "content": judge_system(region_language([item.region for item in items]))},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _region_dict(item: JudgeItem) -> dict[str, Any]:
    """The JSON dict for one region; `previous`/`problems` only once a rejected answer exists."""
    region_dict: dict[str, Any] = {
        "id": item.region.id,
        "kind": item.region.kind,
        "source": source_text(item.region),
        "candidates": item.candidates,
    }
    if item.previous is not None:
        region_dict["previous"] = item.previous
        region_dict["problems"] = list(item.problems)
    return region_dict

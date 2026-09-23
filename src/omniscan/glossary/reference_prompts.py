"""Prompt building and reply parsing for reference-mode glossary extraction (`glossary/reference.py`).

The extraction model sees one matched chapter pair's paired lines (Korean source, official English)
and returns the proper-noun-like terms it can see translated. Message building and tolerant reply
parsing live here, in the style of translate/prompts.py / judge_prompts.py; the client call and the
aggregation/locking policy live in glossary/reference.py.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast, get_args

from omniscan.core.schemas import TermType
from omniscan.translate.languages import source_language


def extract_system(lang: str) -> str:
    """The extraction system prompt for `lang`'s source language; "ko" is the original."""
    sl = source_language(lang)
    return (
        f"You are building the glossary of an official English release of a {sl.name} {sl.work} from "
        "its own translated chapters. You get one chapter's text as paired lines: 'source' is the "
        f"{sl.name} original, 'target' is the official English translation of that same line. List "
        "every term a glossary should bind: character names, place names, organisations, ranks and "
        "titles, recurring items, skills or techniques. For each term give the "
        f"{sl.name} source exactly as it appears in a source line with any attached particle (이, 가, "
        "은, 는, 을, 를, 의, 도, 아, 야, 에, 에게, 에서) removed; the English target exactly as the "
        "paired target lines spell it; and one type: person, "
        "place, org, skill, item, rank, title, honorific, sfx or other. Only list terms whose English "
        "you can actually see in the target lines — never invent or re-romanise a translation. A term "
        "is one to a few syllables, never a whole sentence or phrase. If you saw the same term with "
        "different English spellings, list the spelling that appeared most. Answer with JSON only, in "
        'exactly this shape: {"terms":[{"source":"민준","target":"Minjun","type":"person"}]} — one '
        "entry per distinct term, no commentary."
    )


EXTRACT_SYSTEM = extract_system("ko")

TERMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string", "enum": list(get_args(TermType))},
                },
                "required": ["source", "target"],
            },
        }
    },
    "required": ["terms"],
}

_TERM_TYPES = frozenset(get_args(TermType))


@dataclass(frozen=True, slots=True)
class PairedLine:
    """One paired text line of a matched chapter pair (reading-order rank within its page)."""

    source: str
    target: str
    page_a: int  # 0-based page index in the raw chapter
    page_b: int  # 0-based page index in the reference chapter


@dataclass(frozen=True, slots=True)
class RawTerm:
    """One term exactly as the extraction model returned it (unknown or missing types fall back)."""

    source: str
    target: str
    type: TermType = "other"


def terms_messages(lines: Sequence[PairedLine], lang: str = "ko") -> list[dict[str, str]]:
    """The extraction prompt: system message plus the chapter's paired lines as one JSON list."""
    pairs_json = json.dumps(
        [{"source": line.source, "target": line.target} for line in lines], ensure_ascii=False, indent=1
    )
    return [
        {"role": "system", "content": extract_system(lang)},
        {"role": "user", "content": f"Paired lines (reading order):\n{pairs_json}"},
    ]


def parse_terms_reply(content: str) -> list[RawTerm]:
    """The usable terms of an extraction reply; tolerant of malformed JSON and entries."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    items = data.get("terms") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    terms: list[RawTerm] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source, target = item.get("source"), item.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if not source.strip() or not target.strip():
            continue
        terms.append(RawTerm(source.strip(), target.strip(), _term_type(item.get("type"))))
    return terms


def _term_type(value: object) -> TermType:
    """`value` as a TermType; anything unknown or missing falls back to 'other'."""
    if isinstance(value, str) and value in _TERM_TYPES:
        return cast(TermType, value)
    return "other"

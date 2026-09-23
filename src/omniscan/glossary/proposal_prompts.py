"""Prompt building for single-language glossary proposals from OCR text (`glossary/proposals.py`).

Reference mode (reference_prompts.py) learns terms from an official translation; this pass runs
when none exists, so the model must both identify the Korean term and invent its own best English
rendering — which is why everything it proposes is written `proposed`, never auto-locked.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

PROPOSAL_SYSTEM = (
    "You are building the glossary of an English release of a Korean manhwa from its raw Korean "
    "chapters — no official translation exists. You get one chapter's text as its lines in reading "
    "order. List every term a glossary should bind: character names, place names, organisations, "
    "ranks and titles, recurring items, skills or techniques. For each term give the Korean source "
    "exactly as it appears in a line with any attached particle (이, 가, 은, 는, 을, 를, 의, 도, 아, "
    "야, 에, 에게, 에서) removed; and your own best English rendering — romanise names as the "
    "release would spell them, translate everything else as the release would. Give one type: "
    "person, place, org, skill, item, rank, title, honorific, sfx or other. A term is one to a few "
    "syllables, never a whole sentence or phrase. If you saw the same term more than once, list it "
    "once. Answer with JSON only, in exactly this shape: "
    '{"terms":[{"source":"민준","target":"Minjun","type":"person"}]} — one entry per distinct term, '
    "no commentary."
)


def proposal_messages(lines: Sequence[str]) -> list[dict[str, str]]:
    """The proposal prompt: system message plus the chapter's source lines as one JSON list."""
    lines_json = json.dumps(list(lines), ensure_ascii=False, indent=1)
    return [
        {"role": "system", "content": PROPOSAL_SYSTEM},
        {"role": "user", "content": f"Chapter lines (reading order):\n{lines_json}"},
    ]

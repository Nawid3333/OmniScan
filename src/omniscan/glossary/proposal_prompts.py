"""Prompt building for single-language glossary proposals from OCR text (`glossary/proposals.py`).

Reference mode (reference_prompts.py) learns terms from an official translation; this pass runs
when none exists, so the model must both identify the Korean term and invent its own best rendering in
the release language (`[translate] target_lang`, English by default) — which is why everything it proposes is
written `proposed`, never auto-locked.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from omniscan.translate.languages import source_language, target_language


def proposal_system(lang: str, target: str = "en") -> str:
    """The proposal system prompt for `lang`'s source language and a release in `target`; "ko" into English is
    the original."""
    sl, tl = source_language(lang), target_language(target)
    article = "an" if tl.name[0] in "AEIOU" else "a"
    return (
        f"You are building the glossary of {article} {tl.name} release of a {sl.name} {sl.work} from its raw "
        f"{sl.name} chapters — no official translation exists. You get one chapter's text as its "
        "lines in reading order. List every term a glossary should bind: character names, place "
        "names, organisations, ranks and titles, recurring items, skills or techniques. For each "
        f"term give the {sl.name} source exactly as it appears in a line with any attached particle "
        f"(이, 가, 은, 는, 을, 를, 의, 도, 아, 야, 에, 에게, 에서) removed; and your own best {tl.name} "
        "rendering — romanise names as the release would spell them, translate everything else as "
        "the release would. Give one type: person, place, org, skill, item, rank, title, honorific, "
        "sfx or other. A term is one to a few syllables, never a whole sentence or phrase. If you "
        "saw the same term more than once, list it once. Answer with JSON only, in exactly this "
        "shape: "
        '{"terms":[{"source":"민준","target":"Minjun","type":"person"}]} — one entry per distinct term, '
        "no commentary."
    )


PROPOSAL_SYSTEM = proposal_system("ko")


def proposal_messages(lines: Sequence[str], lang: str = "ko", target: str = "en") -> list[dict[str, str]]:
    """The proposal prompt: system message plus the chapter's source lines as one JSON list."""
    lines_json = json.dumps(list(lines), ensure_ascii=False, indent=1)
    return [
        {"role": "system", "content": proposal_system(lang, target)},
        {"role": "user", "content": f"Chapter lines (reading order):\n{lines_json}"},
    ]

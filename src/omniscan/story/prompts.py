"""Prompt building and reply parsing for chapter summarisation (`story/summarize.py`).

In the style of glossary/reference_prompts.py: the system message states the task and the JSON
answer shape, and reply parsing is tolerant of the fences and prose chat models add — a
"write a summary" prompt in particular sometimes gets answered with plain text instead of JSON.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from omniscan.translate.languages import source_language

MAX_SUMMARY_CHARS = 800

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


def summary_system(lang: str) -> str:
    """The summarisation system prompt for `lang`'s source language; "ko" is the original."""
    sl = source_language(lang)
    return (
        f"You are the continuity writer for an official English release of a {sl.name} {sl.work}. "
        "You get one chapter's translated text as its lines in reading order. Write a short English "
        "summary of what happens in this chapter: 3 to 5 sentences, third person, present tense, "
        "naming the characters and places involved and what changed in this chapter only — nothing "
        "about earlier chapters, no meta-commentary about the art or the lettering, no chapter "
        "numbers. Answer with JSON only, in "
        'exactly this shape: {"summary":"..."} — the summary text and nothing else.'
    )


SUMMARY_SYSTEM = summary_system("ko")


def summary_messages(lines: Sequence[str], lang: str = "ko") -> list[dict[str, str]]:
    """The summarisation prompt: system message plus the chapter's lines as one user message."""
    return [
        {"role": "system", "content": summary_system(lang)},
        {"role": "user", "content": "Chapter lines (reading order):\n" + "\n".join(lines)},
    ]


def parse_summary_reply(content: str) -> str | None:
    """The usable summary of a chat reply; a JSON `"summary"` first, the raw prose as the fallback."""
    text = content.strip()
    if text.startswith("```"):
        text = _strip_fence(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        value = data.get("summary")
        if isinstance(value, str) and value.strip():
            return value.strip()[:MAX_SUMMARY_CHARS]
    return text[:MAX_SUMMARY_CHARS] or None  # bare prose is still a summary; an empty reply is not


def _strip_fence(text: str) -> str:
    """The body of a ```-fenced reply, minus a leading `json` language line; empty when degenerate."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith("```"):
        return text
    lines = lines[1:]
    if lines and lines[0].strip() == "json":
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

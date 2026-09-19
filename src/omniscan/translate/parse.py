"""Tolerant parsing of chat-model translation answers (see docs/benchmarks/translation-probe.md).

Cloud models ignore Ollama's `format` JSON schema: they wrap the JSON in a ```json fence, put prose
around or before it, or answer with two JSON objects. The extractor tries plain JSON, fenced blocks
and finally raw-decodes every `{` left to right, and only accepts an object whose top-level key is a
list — `extract_list(content, key)` for any key, `extract_translations` for "translations".
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?[^\S\n]*\n?(.*?)```", re.DOTALL)


@dataclass(frozen=True, slots=True)
class ParseResult:
    """What a model reply yielded against the expected region ids."""

    texts: dict[str, str]  # id -> stripped non-empty text, only ids from `expected_ids`
    missing: list[str]  # expected ids without a usable text, in `expected_ids` order


def extract_list(content: str, key: str) -> list[Any] | None:
    """The first `key` list found in `content` (plain JSON, fenced blocks, then raw decode)."""
    attempts: list[Any] = []
    with contextlib.suppress(json.JSONDecodeError):
        attempts.append(json.loads(content.strip()))
    attempts.extend(fence.group(1) for fence in _FENCE_RE.finditer(content))
    decoder = json.JSONDecoder()
    for start, char in enumerate(content):
        if char == "{":
            try:
                attempts.append(decoder.raw_decode(content, start)[0])
            except json.JSONDecodeError:
                continue
    for attempt in attempts:
        if isinstance(attempt, str):
            try:
                attempt = json.loads(attempt)
            except json.JSONDecodeError:
                continue
        if isinstance(attempt, dict) and isinstance(attempt.get(key), list):
            return attempt[key]
    return None


def extract_translations(content: str) -> list[Any] | None:
    """The first `translations` list found in `content` (plain JSON, fenced blocks, then raw decode)."""
    return extract_list(content, "translations")


def parse_translations(content: str, expected_ids: Sequence[str]) -> ParseResult:
    """Usable id -> text pairs from a model reply; anything malformed, unknown or empty is skipped."""
    items = extract_translations(content)
    texts: dict[str, str] = {}
    if items is not None:
        expected = set(expected_ids)
        for item in items:
            if not isinstance(item, dict):
                continue
            region_id = item.get("id")
            text = item.get("text")
            if not isinstance(region_id, str) or not isinstance(text, str):
                continue
            if region_id not in expected or region_id in texts:
                continue
            stripped = text.strip()
            if stripped:
                texts[region_id] = stripped
    missing = [region_id for region_id in expected_ids if region_id not in texts]
    return ParseResult(texts=texts, missing=missing)

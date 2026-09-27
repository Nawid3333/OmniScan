"""Keys of what produced a translation or a judged line, so a re-run redoes only what changed.

A key hashes everything the model saw for one region that the stage's own invalidation does not already
cover per region: the source text, kind and language, the glossary entries that match it, and the model
settings. When one bubble's text is fixed by hand, only that bubble is translated and judged again; the
chapter's other lines are kept word for word (and cost no tokens). The story summary is context, not
content, and is left out: a new summary of an earlier chapter does not re-translate this one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from omniscan.core.schemas import GlossaryEntry, Region
from omniscan.translate.judge_config import JudgeConfig
from omniscan.translate.profiles import TranslationProfile
from omniscan.translate.prompts import glossary_subset, source_text

REJUDGE_FLAGS = frozenset({"judge_failed"})  # a line with one of these flags is judged again next run


def _digest(payload: Mapping[str, Any]) -> str:
    """sha256 of a canonical JSON encoding."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _terms(region: Region, entries: Sequence[GlossaryEntry]) -> list[list[str]]:
    """The glossary entries that match the region's text, as sorted (source, target, status) triples."""
    return sorted([e.source, e.target, e.status] for e in glossary_subset([region], entries))


def translation_key(region: Region, entries: Sequence[GlossaryEntry], profile: TranslationProfile) -> str:
    """Key of one region's translation by `profile`."""
    return _digest(
        {
            "text": source_text(region),
            "kind": region.kind,
            "lang": region.lang,
            "glossary": _terms(region, entries),
            "profile": profile.model_dump(exclude={"enabled", "fallback", "chunk_regions"}),
        }
    )


def judge_key(
    region: Region,
    runs: Mapping[str, Mapping[str, str]],
    entries: Sequence[GlossaryEntry],
    cfg: JudgeConfig,
) -> str:
    """Key of one region's judged line: its source, every run's candidate for it, glossary and judge."""
    return _digest(
        {
            "text": source_text(region),
            "kind": region.kind,
            "lang": region.lang,
            "candidates": sorted(
                [run_id, texts[region.id]] for run_id, texts in runs.items() if region.id in texts
            ),
            "glossary": _terms(region, entries),
            "judge": cfg.model_dump(),
        }
    )

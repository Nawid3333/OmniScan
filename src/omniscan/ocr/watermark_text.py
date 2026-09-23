"""Watermark text patterns (promo filter tier 1, card F2b): reclassify ad regions after OCR.

A detected region whose OCR'd text contains one of the configured patterns is a source-injected
ad/watermark: it is reclassified as kind="watermark", which every downstream consumer already
excludes (`translatable`, `score_chapter`, inpaint — docs/OPEN_QUESTIONS.md E4: a watermark stays
on the page, it is only kept out of the translation). Patterns come from the shipped repo file
`config/watermark_text.toml` plus the per-user `~/.config/omniscan/watermark_text.toml`: there is
no name key here, so the two files' pattern lists are simply concatenated in load order (a later
file adds to — never replaces — the earlier one's patterns).
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from pathlib import Path

from omniscan.core.config import DEFAULT_TOML, USER_TOML
from omniscan.core.schemas import Region

# sfx text is short and would collide with patterns easily, and an already-watermark region needs
# no copy — only these kinds are ever reclassified
_RECLASSIFY_KINDS = frozenset({"bubble_text", "free_text"})


def default_watermark_text_paths() -> list[Path]:
    """Shipped repo file, then the per-user one (both contribute; see the module docstring)."""
    return [DEFAULT_TOML.parent / "watermark_text.toml", USER_TOML.parent / "watermark_text.toml"]


def load_watermark_patterns(paths: Sequence[Path]) -> list[str]:
    """Every pattern from every existing file in `paths`, in order, duplicates kept."""
    patterns: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{path}: invalid TOML: {exc}") from exc
        table = data.get("watermark_text", {})
        if not isinstance(table, dict):
            raise ValueError(f"{path}: expected a [watermark_text] table")
        entries = table.get("patterns", [])
        if not isinstance(entries, list) or not all(isinstance(p, str) for p in entries):
            raise ValueError(f"{path}: watermark_text.patterns must be a list of strings")
        if any(not pattern.strip() for pattern in entries):
            raise ValueError(f"{path}: watermark_text.patterns must not contain empty patterns")
        patterns.extend(entries)
    return patterns


def matches_watermark_text(text: str, patterns: Sequence[str]) -> bool:
    """True if any pattern is a case-insensitive substring of `text`. Empty `patterns` -> always False.

    Both sides are whitespace-collapsed the same way `translate.prompts.source_text` does, so a
    pattern still matches when the OCR put a line break inside it."""
    collapsed = " ".join(text.split()).casefold()
    return any(" ".join(pattern.split()).casefold() in collapsed for pattern in patterns)


def reclassify_watermark_regions(regions: Sequence[Region], patterns: Sequence[str]) -> list[Region]:
    """Regions whose OCR text matches a pattern get kind="watermark" (via model_copy); others unchanged.

    Same object, not just equal — callers may rely on identity for regions that pass through. Only
    bubble_text/free_text regions are reclassified; every other field of a matched region is untouched."""
    return [
        region.model_copy(update={"kind": "watermark"})
        if region.kind in _RECLASSIFY_KINDS and matches_watermark_text(region.text, patterns)
        else region
        for region in regions
    ]

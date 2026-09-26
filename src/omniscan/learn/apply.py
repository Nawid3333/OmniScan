"""Learned rules and translation memory applied to a new chapter: OCR readings and translation requests."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from omniscan.core.config import LearnConfig
from omniscan.core.schemas import LearnKind, MemoryEntry, Region, SeriesMemory
from omniscan.learn.harvest import norm
from omniscan.learn.memory import PUNCTUATION, core, is_active

_SPLIT = re.compile(r"(\s+)")


def _active(memory: SeriesMemory, cfg: LearnConfig, kind: LearnKind) -> list[tuple[str, str]]:
    """(wrong, right) of the active rules of one kind."""
    return [(rule.wrong, rule.right) for rule in memory.rules if rule.kind == kind and is_active(rule, cfg)]


def fix_words(text: str, fixes: dict[str, str]) -> str:
    """`text` with every whole word listed in `fixes` replaced (punctuation around words and spacing kept)."""
    if not fixes:
        return text
    parts = _SPLIT.split(text)
    for i, part in enumerate(parts):
        word = core(part)
        if word in fixes:
            start = len(part) - len(part.lstrip(PUNCTUATION))
            parts[i] = part[:start] + fixes[word] + part[start + len(word) :]
    return "".join(parts)


def apply_to_regions(
    regions: Sequence[Region], memory: SeriesMemory, cfg: LearnConfig
) -> tuple[list[Region], dict[str, float]]:
    """The OCR's regions with the series' lessons applied: learned word fixes (the reading before the fix
    kept as `ocr_alt` unless a second engine's reading is already there), texts the editor keeps deleting dropped, texts the editor marks as a watermark or a
    sound effect re-kinded; plus counts of each."""
    fixes = dict(_active(memory, cfg, "ocr_fix"))
    drops = {wrong for wrong, _ in _active(memory, cfg, "drop_text")}
    watermarks = {wrong for wrong, _ in _active(memory, cfg, "watermark_text")}
    effects = {wrong for wrong, _ in _active(memory, cfg, "sfx_text")}
    result: list[Region] = []
    counts = {"learned_fixes": 0.0, "learned_drops": 0.0, "learned_kinds": 0.0}
    for region in regions:
        fixed = fix_words(region.text, fixes)
        texts = {norm(region.text), norm(fixed)}
        if region.kind != "watermark" and texts & drops:
            counts["learned_drops"] += 1
            continue
        update: dict[str, object] = {}
        if fixed != region.text:
            update.update(text=fixed, ocr_alt=region.ocr_alt or region.text)
            counts["learned_fixes"] += 1
        if region.kind != "watermark" and texts & watermarks:
            update["kind"] = "watermark"
            counts["learned_kinds"] += 1
        elif region.kind in ("bubble_text", "free_text") and texts & effects:
            update["kind"] = "sfx"
            counts["learned_kinds"] += 1
        result.append(region.model_copy(update=update) if update else region)
    return result, counts


def _bigrams(text: str) -> set[str]:
    """Character bigrams of `text` without spaces (a language-neutral similarity basis for CJK)."""
    flat = "".join(text.split())
    return {flat[i : i + 2] for i in range(len(flat) - 1)} or ({flat} if flat else set())


def similarity(a: str, b: str) -> float:
    """Jaccard similarity of two texts' character bigrams, 0..1."""
    x, y = _bigrams(a), _bigrams(b)
    return len(x & y) / len(x | y) if x and y else 0.0


MIN_SIMILARITY = 0.3  # a memory line at least this similar to a line being translated is shown as an example


@dataclass(frozen=True, slots=True)
class TranslationHints:
    """What the translation requests of a chapter get from the series' memory."""

    exact: dict[str, str]  # source line (whitespace collapsed) -> the editor's English: used as is
    entries: tuple[MemoryEntry, ...]
    preferences: tuple[tuple[str, str], ...]  # (machine word, the editor's word)
    examples: int

    def examples_for(self, texts: Sequence[str]) -> list[tuple[str, str]]:
        """Up to `examples` (source, English) memory lines most similar to `texts` (and not one of them)."""
        if self.examples <= 0 or not self.entries:
            return []
        wanted = {norm(text) for text in texts}
        scored = sorted(
            (
                (max(similarity(entry.source, text) for text in wanted), entry)
                for entry in self.entries
                if entry.source not in wanted
            ),
            key=lambda pair: -pair[0],
        )
        return [(entry.source, entry.english) for score, entry in scored if score >= MIN_SIMILARITY][
            : self.examples
        ]


def translation_hints(memory: SeriesMemory, cfg: LearnConfig) -> TranslationHints:
    """The translation memory and preferred wording of a series, ready for the translation runner."""
    return TranslationHints(
        exact={entry.source: entry.english for entry in memory.translations},
        entries=tuple(memory.translations),
        preferences=tuple(_active(memory, cfg, "preferred_term")),
        examples=cfg.examples,
    )

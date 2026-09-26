"""memory.json: the rules and translation memory a series' corrections add up to, rebuilt when edits change.

A rule needs evidence: an OCR word or English word is replaced only after `learn.min_count` corrections
made the same change and none kept the old word; a region text is dropped after as many deletions. The
editor sees every rule with its count and switches any of them off; a switch survives rebuilds.
"""

from __future__ import annotations

import hashlib
import string
import threading
from collections import Counter
from collections.abc import Iterable
from difflib import SequenceMatcher
from pathlib import Path

from omniscan.core.config import LearnConfig
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import LearnedRule, LearnKind, MemoryEntry, SeriesMemory
from omniscan.edits.store import EDITS_FILE
from omniscan.learn.harvest import Corrections, harvest_series

MEMORY_FILE = "memory.json"
PUNCTUATION = string.punctuation + "…·~「」『』“”‘’。、，！？：；（）"
_LABEL_KINDS: frozenset[LearnKind] = frozenset({"watermark_text", "sfx_text"})  # one correction is enough

_LOCK = threading.RLock()


def core(token: str) -> str:
    """A word without the punctuation around it."""
    return token.strip(PUNCTUATION)


def rule_id(kind: LearnKind, wrong: str, right: str) -> str:
    """Stable id of a rule (so its `enabled` switch survives a rebuild)."""
    return hashlib.sha1(f"{kind}\0{wrong}\0{right}".encode()).hexdigest()[:12]


def word_changes(pairs: Iterable[tuple[str, str]], *, capitalised: bool = False) -> Counter[tuple[str, str]]:
    """How often each word was replaced by another (same number of words on both sides of a change),
    punctuation around words ignored; with `capitalised`, only words starting with a capital letter
    (names, titles) on both sides."""
    counts: Counter[tuple[str, str]] = Counter()
    for before, after in pairs:
        a, b = before.split(), after.split()
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if tag != "replace" or i2 - i1 != j2 - j1:
                continue
            for x, y in zip(a[i1:i2], b[j1:j2], strict=True):
                wrong, right = core(x), core(y)
                if not wrong or not right or wrong == right:
                    continue
                if capitalised and not (wrong[0].isupper() and right[0].isupper()):
                    continue
                counts[(wrong, right)] += 1
    return counts


def _kept(pairs: Iterable[tuple[str, str]]) -> Counter[str]:
    """How often each word appears in the corrected texts (a word the editor kept somewhere is not wrong)."""
    return Counter(core(token) for _before, after in pairs for token in after.split())


def _replacements(kind: LearnKind, counts: Counter[tuple[str, str]], kept: Counter[str]) -> list[LearnedRule]:
    """One rule per wrong word: its most frequent replacement, unless the word was also kept as often."""
    best: dict[str, tuple[str, int]] = {}
    for (wrong, right), count in counts.most_common():
        if wrong not in best:
            best[wrong] = (right, count)
    return [
        LearnedRule(id=rule_id(kind, wrong, right), kind=kind, wrong=wrong, right=right, count=count)
        for wrong, (right, count) in best.items()
        if kept[wrong] < count
    ]


def _texts(kind: LearnKind, texts: Iterable[str]) -> list[LearnedRule]:
    """One rule per region text, counted; a one-character text only as a sound effect (쾅, ドン) — deleting
    or hiding every region that reads one character is too blunt a lesson."""
    shortest = 1 if kind == "sfx_text" else 2
    return [
        LearnedRule(id=rule_id(kind, text, ""), kind=kind, wrong=text, count=count)
        for text, count in Counter(t for t in texts if len(t) >= shortest).most_common()
    ]


def build_memory(found: Corrections, previous: SeriesMemory | None = None) -> SeriesMemory:
    """The rules and translation memory `found` adds up to; `enabled` switches are kept from `previous`."""
    rules = [
        *_replacements("ocr_fix", word_changes(found.ocr_fixes), _kept(found.ocr_fixes)),
        *_replacements(
            "preferred_term", word_changes(found.english_fixes, capitalised=True), _kept(found.english_fixes)
        ),
        *_texts("drop_text", found.deleted),
        *_texts("watermark_text", (text for text, kind in found.kinds if kind == "watermark")),
        *_texts("sfx_text", (text for text, kind in found.kinds if kind == "sfx")),
    ]
    switched = {rule.id: rule.enabled for rule in previous.rules} if previous is not None else {}
    rules = [rule.model_copy(update={"enabled": switched.get(rule.id, True)}) for rule in rules]
    entries: dict[str, MemoryEntry] = {}
    for source, english, typed, chapter in found.translations:
        seen = entries.get(source)
        entries[source] = MemoryEntry(
            source=source,
            english=english,  # the latest one wins: the editor's current wording
            count=(seen.count + 1) if seen else 1,
            typed=typed or (seen.typed if seen else False),
            chapter=chapter,
        )
    return SeriesMemory(rules=rules, translations=list(entries.values()))


def is_active(rule: LearnedRule, cfg: LearnConfig) -> bool:
    """True when the rule is switched on and has enough evidence."""
    return rule.enabled and rule.count >= (1 if rule.kind in _LABEL_KINDS else cfg.min_count)


def memory_path(series: SeriesPaths) -> Path:
    """The series' memory.json."""
    return series.work_dir / MEMORY_FILE


def _stale(series: SeriesPaths, path: Path) -> bool:
    """True when memory.json is missing or not newer than one of the chapters' edits.json (a tie rebuilds:
    a coarse file clock must never hide an edit)."""
    if not path.is_file():
        return True
    built = path.stat().st_mtime_ns
    edits = (series.chapter(chapter).artifact(EDITS_FILE) for chapter in series.chapters())
    return any(edit.is_file() and edit.stat().st_mtime_ns >= built for edit in edits)


def current_memory(series: SeriesPaths, *, rebuild: bool = False) -> SeriesMemory:
    """The series' memory, rebuilt from the chapters' edits first when one changed since (or `rebuild`)."""
    with _LOCK:
        path = memory_path(series)
        previous = SeriesMemory.load(path) if path.is_file() else None
        if previous is not None and not rebuild and not _stale(series, path):
            return previous
        memory = build_memory(harvest_series(series), previous)
        if memory.rules or memory.translations or previous is not None:
            memory.save(path)
        return memory


def set_rule_enabled(series: SeriesPaths, rule: str, enabled: bool) -> LearnedRule:
    """Switch one learned rule on or off; LookupError when there is no such rule."""
    with _LOCK:
        memory = current_memory(series)
        index = next((i for i, r in enumerate(memory.rules) if r.id == rule), None)
        if index is None:
            raise LookupError(f"no learned rule {rule!r}")
        memory.rules[index] = memory.rules[index].model_copy(update={"enabled": enabled})
        memory.save(memory_path(series))
        return memory.rules[index]

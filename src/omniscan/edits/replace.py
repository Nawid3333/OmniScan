"""Find and replace across a chapter's English lines or source texts, as hand edits.

`FindReplace` is the pure rule: literal text or a regular expression, optionally whole words only; when it
ignores case, the replacement takes each match's case (JINWOO -> JIN-WOO, Jinwoo -> Jin-woo). `plan` lists what
a rule would change in a chapter (watermarks never); `apply_changes` records the changes through edits/store.py,
one undo step per chapter, so they survive re-runs, can be undone and are learned from like any hand correction.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from omniscan.core.paths import ChapterPaths
from omniscan.edits import store
from omniscan.translate.on_demand import english_lines

Target = Literal["english", "source"]


def _same_case(matched: str, new: str) -> str:
    """`new` in the case of `matched`: all capitals, a capital first letter, or as typed."""
    if matched.isupper():
        return new.upper()
    if matched[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


@dataclass(frozen=True, slots=True)
class FindReplace:
    """What to find and what to put instead (`replace` may use \\1 … groups when `regex`)."""

    find: str
    replace: str
    regex: bool = False
    whole_word: bool = False
    case_sensitive: bool = True

    def replacer(self) -> Callable[[str], str]:
        """The rule as a function of a text; ValueError for an empty search or a bad regular expression."""
        if not self.find:
            raise ValueError("nothing to find")
        body = self.find if self.regex else re.escape(self.find)
        if self.whole_word:
            body = rf"(?<!\w)(?:{body})(?!\w)"
        try:
            pattern = re.compile(body, 0 if self.case_sensitive else re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"bad regular expression {self.find!r}: {exc}") from exc

        def one(match: re.Match[str]) -> str:
            try:
                new = match.expand(self.replace) if self.regex else self.replace
            except (re.error, IndexError) as exc:
                raise ValueError(f"bad replacement {self.replace!r}: {exc}") from exc
            return new if self.case_sensitive else _same_case(match.group(0), new)

        return lambda text: pattern.sub(one, text)


@dataclass(frozen=True, slots=True)
class Change:
    """One line a rule changes."""

    chapter: str
    region_id: str
    before: str
    after: str


def plan(paths: ChapterPaths, rule: FindReplace, target: Target) -> list[Change]:
    """What `rule` would change in the chapter's English lines or source texts, in reading order."""
    replace = rule.replacer()
    regions = [r for r in store.current_regions(paths) if r.kind != "watermark"]
    lines = english_lines(paths) if target == "english" else {r.id: r.text for r in regions}
    changes: list[Change] = []
    for region in regions:
        before = lines.get(region.id, "")
        after = replace(before) if before else before
        if after != before:
            changes.append(Change(paths.chapter, region.id, before, after))
    return changes


def apply_changes(
    paths: ChapterPaths, changes: Sequence[Change], target: Target, *, direction: store.Direction
) -> None:
    """Record `changes` (of this chapter) as hand edits: English lines or source texts, one undo step."""
    with store.edit_group(paths):
        for change in changes:
            if target == "english":
                store.set_translation(paths, change.region_id, change.after, direction=direction)
            else:
                store.update_region(paths, change.region_id, direction=direction, text=change.after)

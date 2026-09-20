"""Choosing which chapters to acquire: pasted URL lists and a comma-separated subset spec."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from omniscan.acquire.sources import ChapterSource, named_chapters
from omniscan.core.paths import chapter_number

_TOKEN_RE = re.compile(r"\d+(?:-\d*)?|-\d+")


def read_url_list(
    text: str, *, first_number: int = 1, name_template: str = "Chapter {n}"
) -> list[ChapterSource]:
    """Chapter sources from pasted 'URL' / 'NAME | URL' lines; blank lines and '#' comments are skipped."""
    entries: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, url = line.partition(" | ")
        if separator:
            entries.append((name.strip(), url.strip()))
        else:
            entries.append((name_template.format(n=first_number + len(entries)), line))
    if not entries:
        raise ValueError("no chapter links found")
    return named_chapters(entries)


def select_chapters(
    sources: Sequence[ChapterSource],
    spec: str | None,
    *,
    by: Literal["auto", "number", "position"] = "auto",
) -> list[ChapterSource]:
    """Sources matching a spec of 'N', 'A-B', 'A-', '-B' tokens, in original order; None/'all' → everything."""
    if spec is None or not spec.strip() or spec.strip().lower() == "all":
        return list(sources)
    numbers = [chapter_number(source.name) for source in sources]
    basis = by if by != "auto" else ("number" if all(n is not None for n in numbers) else "position")
    values: list[float | None] = (
        numbers if basis == "number" else [float(position) for position in range(1, len(sources) + 1)]
    )
    selected: set[int] = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        lo, hi = _parse_token(token)
        hits = [
            index
            for index, value in enumerate(values)
            if value is not None and (lo is None or value >= lo) and (hi is None or value <= hi)
        ]
        if not hits:
            raise ValueError(f"selection: nothing matches '{token}'")
        selected.update(hits)
    return [sources[index] for index in sorted(selected)]


def _parse_token(token: str) -> tuple[int | None, int | None]:
    """(lo, hi) of one selection token, None for an open end; ValueError on a malformed or reversed token."""
    if not token:
        raise ValueError("selection: empty token")
    if _TOKEN_RE.fullmatch(token) is None:
        raise ValueError(f"selection: invalid token '{token}'")
    if token.startswith("-"):
        return None, int(token[1:])
    if token.endswith("-"):
        return int(token[:-1]), None
    lo, separator, hi = token.partition("-")
    if not separator:
        return int(lo), int(lo)
    if int(lo) > int(hi):
        raise ValueError(f"selection: reversed range '{token}'")
    return int(lo), int(hi)

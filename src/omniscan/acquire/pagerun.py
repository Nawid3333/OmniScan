"""Selection of the chapter pages out of the full image list extract.pics returns for one page."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from omniscan.acquire.filters import url_looks_like_page

PAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_THUMB_RE = re.compile(r"(\d{2,4})x(\d{2,4})")
_RUN_DIGITS_RE = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class PageSelection:
    """The URLs kept by `select_pages` plus the ones dropped, both in document order."""

    kept: tuple[str, ...]
    dropped: tuple[tuple[str, str], ...]


def select_pages(
    urls: Sequence[str],
    *,
    min_run: int = 3,
    min_share: float = 0.5,
) -> PageSelection:
    """Pick the chapter pages out of every image URL of one page: duplicates, formats, thumbnails, runs."""
    verdicts: list[tuple[str, str | None]] = []
    candidates: list[tuple[int, str]] = []
    seen: set[str] = set()
    for position, url in enumerate(urls):
        reason = _reject_before_run(url, seen)
        seen.add(url)
        verdict: tuple[str, str | None] = (url, reason)
        verdicts.append(verdict)
        if reason is None:
            candidates.append((position, url))
    if candidates:
        _apply_run_rule(verdicts, candidates, min_run=min_run, min_share=min_share)
    kept = tuple(url for url, reason in verdicts if reason is None)
    dropped = tuple((url, reason) for url, reason in verdicts if reason is not None)
    return PageSelection(kept=kept, dropped=dropped)


def _reject_before_run(url: str, seen: set[str]) -> str | None:
    """Rejection reason from the per-URL rules (duplicates, format, tokens, thumbnails); None for candidates."""
    if url in seen:
        return "duplicate"
    name = PurePosixPath(urlsplit(url).path).name
    suffix = PurePosixPath(name).suffix.lower()
    if suffix and suffix not in PAGE_SUFFIXES:
        return f"not a page format ({suffix})"
    ok, reason = url_looks_like_page(url)
    if not ok:
        return reason
    match = _THUMB_RE.search(name)
    if match and int(match.group(1)) <= 400 and int(match.group(2)) <= 400:
        return "thumbnail size in name"
    return None


def _apply_run_rule(
    verdicts: list[tuple[str, str | None]],
    candidates: list[tuple[int, str]],
    *,
    min_run: int,
    min_share: float,
) -> None:
    """Keep only the largest (host, directory, pattern) group when it is a dominant run."""
    groups: dict[tuple[str, str, str], list[int]] = {}
    for position, url in candidates:
        groups.setdefault(_run_key(url), []).append(position)
    best_key = max(groups, key=lambda key: (len(groups[key]), -groups[key][0]))
    best = groups[best_key]
    if len(best) >= min_run and len(best) / len(candidates) >= min_share:
        winners = set(best)
        for position, url in candidates:
            if position not in winners:
                verdicts[position] = (url, "not part of the page run")


def _run_key(url: str) -> tuple[str, str, str]:
    """(host, directory, pattern) under which consecutive chapter pages group together."""
    parts = urlsplit(url)
    directory, _, name = parts.path.rpartition("/")
    return (parts.netloc.lower(), directory, _RUN_DIGITS_RE.sub("#", name))
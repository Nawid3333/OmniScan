"""Completeness checks for downloaded chapters: decodable pages, page counts, widths, duplicates, numbering."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image

from omniscan.core.paths import chapter_number, list_images


@dataclass(frozen=True, slots=True)
class PageInfo:
    """One decoded page of a chapter: file name, pixel size, byte size and content hash."""

    name: str
    width: int
    height: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Finding:
    """One completeness result: severity, short code and a human-readable message."""

    level: Literal["error", "warn"]
    code: str
    message: str


def inspect_chapter(chapter_dir: Path) -> tuple[list[PageInfo], list[Finding]]:
    """Read every image of a chapter folder (natural order); undecodable files become 'unreadable' errors."""
    pages: list[PageInfo] = []
    findings: list[Finding] = []
    for path in list_images(chapter_dir):
        data = path.read_bytes()
        try:
            with Image.open(BytesIO(data)) as image:
                width, height = image.size
        except Exception:
            findings.append(Finding("error", "unreadable", f"{path.name}: cannot be decoded"))
            continue
        pages.append(PageInfo(path.name, width, height, len(data), hashlib.sha256(data).hexdigest()))
    return pages, findings


def check_pages(
    pages: Sequence[PageInfo],
    *,
    median_pages: float | None = None,
    previous_pages: int | None = None,
) -> list[Finding]:
    """Warnings/errors about a page list: count vs. median/previous, width outliers and duplicate hashes."""
    findings: list[Finding] = []
    if not pages:
        return [Finding("error", "no_pages", "no pages found")]
    count = len(pages)
    if (median_pages is not None and count < 0.5 * median_pages) or (
        previous_pages is not None and count < 0.5 * previous_pages
    ):
        if median_pages is not None:
            message = f"{count} pages; the series median is {median_pages:g}"
        else:
            message = f"{count} pages; the previous chapter had {previous_pages}"
        findings.append(Finding("warn", "few_pages", message))
    if median_pages is not None and count > 2 * median_pages:
        findings.append(
            Finding("warn", "many_pages", f"{count} pages; the series median is {median_pages:g}")
        )
    usual = _usual_width([page.width for page in pages])
    if usual is not None:
        differ = sum(1 for width in (page.width for page in pages) if abs(width - usual) > 0.05 * usual)
        if differ:
            findings.append(
                Finding("warn", "mixed_widths", f"{differ} page(s) differ in width from the usual {usual} px")
            )
    extra = len(pages) - len({page.sha256 for page in pages})
    if extra:
        findings.append(
            Finding("warn", "duplicate_pages", f"{extra} page(s) are exact duplicates of another page")
        )
    return findings


def check_numbering(names: Sequence[str]) -> list[Finding]:
    """Gaps in the integer chapter numbers of folder names; decimals and unnumbered names are ignored."""
    numbers = {
        int(number) for name in names if (number := chapter_number(name)) is not None and number.is_integer()
    }
    if len(numbers) < 2:
        return []
    missing = sorted(set(range(min(numbers), max(numbers) + 1)) - numbers)
    runs: list[tuple[int, int]] = []
    for number in missing:
        if runs and runs[-1][1] == number - 1:
            runs[-1] = (runs[-1][0], number)
        else:
            runs.append((number, number))
    return [Finding("warn", "numbering_gap", _gap_message(start, end)) for start, end in runs]


def verdict(findings: Sequence[Finding]) -> Literal["ok", "review", "failed"]:
    """'failed' when any error, 'review' when any warning, else 'ok'."""
    levels = {finding.level for finding in findings}
    if "error" in levels:
        return "failed"
    if "warn" in levels:
        return "review"
    return "ok"


def _usual_width(widths: Sequence[int]) -> int:
    """The most common width; the smaller width wins a tie."""
    counts = Counter(widths)
    top = max(counts.values())
    return min(width for width, count in counts.items() if count == top)


def _gap_message(start: int, end: int) -> str:
    """'chapter 5 is missing' or 'chapters 5–7 are missing' for one run of missing numbers."""
    if start == end:
        return f"chapter {start} is missing"
    return f"chapters {start}–{end} are missing"

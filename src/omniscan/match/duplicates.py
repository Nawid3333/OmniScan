"""Duplicate/near-duplicate chapter detection within one chapter-set directory.

Reuses chapters.py's own page-coverage quality metric (chapter_qualities), comparing every chapter of
`root` against every other chapter of the *same* `root` -- unlike match_chapters's 1:1 alignment across
two independent sets, a chapter here may appear in more than one reported pair (e.g. three accidental
copies of the same chapter each pair with the other two). A high quality normally means the same pages
were imported twice, not "these are similar chapters of the same story" -- ordinary consecutive chapters
of one series share very little page-level art in practice."""

from __future__ import annotations

from pathlib import Path

from omniscan.core.paths import list_chapters
from omniscan.core.schemas import Artifact, Model
from omniscan.match.chapters import Thresholds, chapter_qualities
from omniscan.match.pages import chapter_hashes


class DuplicateThresholds(Model):
    """The knobs of one duplicate-detection run; stored in the report so its numbers can be reproduced."""

    page_similarity: float = 0.75  # dhash similarity at/above which two pages may align
    page_gap: float = 0.25  # page-alignment penalty for leaving a page unmatched
    min_quality: float = 0.9  # page-coverage quality at/above which two chapters are flagged as duplicates


class DuplicatePair(Model):
    """One pair of chapters flagged as likely duplicates of each other."""

    a: str  # chapter folder name (natural-sort earlier of the two)
    b: str  # chapter folder name (natural-sort later of the two)
    quality: float  # same page-coverage quality formula as chapters.ChapterMatch.quality


class DuplicateReport(Artifact):
    """The result of one duplicate-detection run (`omniscan match duplicates`)."""

    root: str
    thresholds: DuplicateThresholds
    duplicates: list[DuplicatePair]  # sorted by descending quality, ties broken by (a, b)


def find_duplicate_chapters(root: Path, thresholds: DuplicateThresholds | None = None) -> DuplicateReport:
    """Every pair of distinct chapters under `root` whose page-coverage quality is >= min_quality.

    Raises ValueError (same condition and message style as chapters.match_chapters) when `root` has no
    chapter folders at all. A chapter folder with no page images can never appear in a reported pair (its
    hashes list is empty, giving 0.0 quality against anything)."""
    thresholds = DuplicateThresholds() if thresholds is None else thresholds
    chapter_dirs = list_chapters(root)
    if not chapter_dirs:
        raise ValueError(f"no chapter folders with page images under {root}")
    names = [p.name for p in chapter_dirs]
    chapters_arg = list(zip(names, [chapter_hashes(p) for p in chapter_dirs], strict=True))
    quality, _page_matches = chapter_qualities(
        chapters_arg,
        chapters_arg,
        Thresholds(
            page_similarity=thresholds.page_similarity,
            page_gap=thresholds.page_gap,
            min_quality=thresholds.min_quality,
        ),
    )
    pairs = [
        DuplicatePair(a=names[i], b=names[j], quality=round(float(quality[i, j]), 4))
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if quality[i, j] >= thresholds.min_quality
    ]
    pairs.sort(key=lambda pair: (-pair.quality, pair.a, pair.b))
    return DuplicateReport(root=str(root), thresholds=thresholds, duplicates=pairs)

"""Chapter-to-chapter alignment of two independently-sourced chapter sets of one series.

Folder names, chapter numbers and chapter counts may all differ between the two sides — matching
runs on page art alone (dHash), aligned at page level inside a chapter pair and at chapter level
across the whole series. Chapters with no confident counterpart come out flagged, never forced."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from omniscan.core.paths import list_chapters
from omniscan.core.schemas import Artifact, Model
from omniscan.match.align import FORBIDDEN, align
from omniscan.match.pages import chapter_hashes, similarity_blocks

PageMatches = list[tuple[int, int, float]]  # (page_a, page_b, dhash similarity) of one chapter pair


class Thresholds(Model):
    """The knobs of one match run; stored in the artifact so its numbers can be reproduced."""

    page_similarity: float = 0.75  # dhash similarity at/above which two pages may align
    page_gap: float = 0.25  # page-alignment penalty for leaving a page unmatched
    chapter_gap: float = 0.6  # chapter-alignment penalty for leaving a chapter unmatched
    min_quality: float = 0.3  # page-coverage quality at/above which a chapter pair may match
    review_quality: float = 0.5  # matched chapters below this are flagged for a second look


class PagePair(Model):
    """One aligned page pair inside a matched chapter pair (0-based page indices, dhash similarity)."""

    page_a: int
    page_b: int
    similarity: float


class RunnerUp(Model):
    """The best alternative partner a matched chapter could have had (a wrong-pairing warning)."""

    side: Literal["a", "b"]
    name: str
    quality: float


class ChapterMatch(Model):
    """One matched chapter pair with its page-level evidence (the confidence signal is `quality`)."""

    a: str  # chapter folder name in dir_a
    b: str  # chapter folder name in dir_b
    quality: float  # aligned-page similarity sum / larger page count: 1.0 = every page matched perfectly
    pages_a: int
    pages_b: int
    pages: list[PagePair]  # the page-level alignment of the two chapters
    runner_up: RunnerUp | None  # the strongest other candidate partner, if one was allowed at all
    review: bool  # True = quality below thresholds.review_quality: check this pair by hand


class ChapterMapping(Artifact):
    """The hand-editable chapter alignment of two chapter-set directories (`omniscan match chapters`)."""

    dir_a: str
    dir_b: str
    thresholds: Thresholds
    matched: list[ChapterMatch]
    unmatched_a: list[str]
    unmatched_b: list[str]

    def b_of(self, chapter_a: str) -> str | None:
        """B folder name matched to an A folder name; None when it is unmatched (or unknown)."""
        return next((m.b for m in self.matched if m.a == chapter_a), None)

    def a_of(self, chapter_b: str) -> str | None:
        """A folder name matched to a B folder name; None when it is unmatched (or unknown)."""
        return next((m.a for m in self.matched if m.b == chapter_b), None)


def match_chapters(dir_a: Path, dir_b: Path, thresholds: Thresholds | None = None) -> ChapterMapping:
    """Compute the chapter alignment of two chapter-set directories (one subfolder per chapter)."""
    thresholds = Thresholds() if thresholds is None else thresholds
    chapters_a = _read(dir_a)
    chapters_b = _read(dir_b)
    if not chapters_a or not chapters_b:
        raise ValueError(f"no chapter folders with page images under {dir_a} / {dir_b}")
    quality, page_matches = _chapter_qualities(chapters_a, chapters_b, thresholds)
    chapter_align = align(
        np.where(quality >= thresholds.min_quality, quality - 1.0, FORBIDDEN),
        gap=thresholds.chapter_gap,
    )
    matched = [
        _chapter_match(chapters_a[i], chapters_b[j], quality[i, j], page_matches[(i, j)], thresholds)
        for i, j in chapter_align.matches
    ]
    _add_runner_ups(matched, chapter_align.matches, quality, chapters_a, chapters_b, thresholds)
    return ChapterMapping(
        dir_a=str(dir_a),
        dir_b=str(dir_b),
        thresholds=thresholds,
        matched=matched,
        unmatched_a=[chapters_a[i][0] for i in chapter_align.only_a],
        unmatched_b=[chapters_b[j][0] for j in chapter_align.only_b],
    )


def _read(root: Path) -> list[tuple[str, list[int]]]:
    """(chapter folder name, page dhashes) of every chapter, in reading order.

    Chapter folders with no page images stay in the list with zero hashes: they can only come out
    unmatched, which is the honest answer for an empty folder."""
    return [(p.name, chapter_hashes(p)) for p in list_chapters(root)]


def _chapter_qualities(
    chapters_a: Sequence[tuple[str, list[int]]],
    chapters_b: Sequence[tuple[str, list[int]]],
    thresholds: Thresholds,
) -> tuple[np.ndarray, dict[tuple[int, int], PageMatches]]:
    """[len_a, len_b] page-coverage quality of every chapter pair, plus each viable pair's page matches.

    A pair's quality is the sum of its aligned page similarities divided by the larger page count —
    1.0 when every page of both sides matched perfectly, ~0 when nothing matched. Before any page-level
    alignment runs, a cheap upper bound (each page's best partner on the other side, averaged) prunes
    the pairs that provably cannot reach `min_quality`; only survivors pay for the DP."""
    len_a, len_b = len(chapters_a), len(chapters_b)
    quality = np.zeros((len_a, len_b))
    page_matches: dict[tuple[int, int], PageMatches] = {}
    lengths_a = np.asarray([len(hashes) for _, hashes in chapters_a])
    lengths_b = np.asarray([len(hashes) for _, hashes in chapters_b])
    for a_indices, block in similarity_blocks(
        [hashes for _, hashes in chapters_a], [hashes for _, hashes in chapters_b]
    ):
        valid = (lengths_a[list(a_indices)] > 0)[:, None] & (lengths_b > 0)[None, :]
        block = np.where(valid[:, :, None, None], block, -1.0)  # similarities are >= 0; -1 prunes
        bound = np.minimum(
            block.max(axis=3).mean(axis=2),  # per A page, its best B page, averaged
            block.max(axis=2).mean(axis=3),  # per B page, its best A page, averaged
        )
        rows = lengths_a[list(a_indices)]
        for a_row in range(block.shape[0]):
            a_index = a_indices.start + a_row
            for b_index in np.flatnonzero(bound[a_row] >= thresholds.min_quality):
                matches = align(
                    np.where(
                        block[a_row, b_index][: rows[a_row], : lengths_b[b_index]]
                        >= thresholds.page_similarity,
                        block[a_row, b_index][: rows[a_row], : lengths_b[b_index]],
                        FORBIDDEN,
                    ),
                    gap=thresholds.page_gap,
                ).matches
                quality[a_index, int(b_index)] = sum(
                    float(block[a_row, b_index][i, j]) for i, j in matches
                ) / max(rows[a_row], lengths_b[b_index])
                page_matches[(a_index, int(b_index))] = [
                    (i, j, float(block[a_row, b_index][i, j])) for i, j in matches
                ]
    return quality, page_matches


def _chapter_match(
    chapter_a: tuple[str, list[int]],
    chapter_b: tuple[str, list[int]],
    pair_quality: float,
    matches: PageMatches,
    thresholds: Thresholds,
) -> ChapterMatch:
    """Assemble one matched chapter pair's artifact entry from its page alignment."""
    return ChapterMatch(
        a=chapter_a[0],
        b=chapter_b[0],
        quality=round(float(pair_quality), 4),
        pages_a=len(chapter_a[1]),
        pages_b=len(chapter_b[1]),
        pages=[PagePair(page_a=i, page_b=j, similarity=round(sim, 4)) for i, j, sim in sorted(matches)],
        runner_up=None,  # filled in by _add_runner_ups
        review=float(pair_quality) < thresholds.review_quality,
    )


def _add_runner_ups(
    matched: list[ChapterMatch],
    pair_indices: Sequence[tuple[int, int]],
    quality: np.ndarray,
    chapters_a: Sequence[tuple[str, list[int]]],
    chapters_b: Sequence[tuple[str, list[int]]],
    thresholds: Thresholds,
) -> None:
    """Attach each matched pair's strongest alternative partner (`side` = the alternative's side).

    An alternative can be a chapter that was matched elsewhere: the point is to warn when the chosen
    pairing was not clearly better than the next-best option, so the owner knows where to look."""
    allowed = np.where(quality >= thresholds.min_quality, quality, 0.0)
    names_a = [name for name, _ in chapters_a]
    names_b = [name for name, _ in chapters_b]
    for (i, j), match in zip(pair_indices, matched, strict=True):
        candidates: list[RunnerUp] = []
        for side, row, own, names in (
            ("b", allowed[i], j, names_b),  # other B chapters A's chapter could have matched
            ("a", allowed[:, j], i, names_a),  # other A chapters B's chapter could have matched
        ):
            other = _best_other(row, own)
            if other is not None:
                candidates.append(RunnerUp(side=side, name=names[other[0]], quality=round(other[1], 4)))
        match.runner_up = max(candidates, key=lambda candidate: candidate.quality) if candidates else None


def _best_other(row: np.ndarray, own: int) -> tuple[int, float] | None:
    """(index, quality) of a row's best allowed partner other than `own`; None when there is none."""
    masked = row.copy()
    masked[own] = 0.0
    best = int(np.argmax(masked))
    return (best, float(masked[best])) if masked[best] > 0.0 else None

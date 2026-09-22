"""Needleman-Wunsch-style global alignment of two ordered sequences over a score matrix.

Used twice by chapter matching: at page level (align the pages of two candidate chapters, tolerant
of an extra ad page on either side) and at chapter level (align the chapters of two series-wide
chapter sets, tolerant of a chapter only present on one side). One algorithm, no special cases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FORBIDDEN = -1e9  # pair score the caller rules out: no alignment can afford it (all-gap paths cost ~-2)


@dataclass(frozen=True, slots=True)
class Alignment:
    """Monotone index correspondence of two sequences A and B.

    `matches` holds (a_index, b_index) pairs in increasing order; `only_a` / `only_b` are the indices
    left without a counterpart; `score` is the alignment's total score (matches + gap penalties)."""

    matches: tuple[tuple[int, int], ...]
    only_a: tuple[int, ...]
    only_b: tuple[int, ...]
    score: float


def align(score: np.ndarray, gap: float) -> Alignment:
    """Globally align two sequences given their [len_a, len_b] score matrix (maximised) and per-gap penalty.

    A pair whose score is `-1e9` can never be matched (the caller gates dissimilar pairs out that way);
    such pairs are treated as gaps. Ties prefer matching, then skipping A."""
    len_a, len_b = score.shape
    if len_a == 0 or len_b == 0:
        return Alignment((), tuple(range(len_a)), tuple(range(len_b)), -gap * (len_a + len_b))
    table = _dp_table(score, gap)
    return _traceback(table, score, gap)


def _dp_table(score: np.ndarray, gap: float) -> np.ndarray:
    """Full (len_a+1, len_b+1) DP matrix; table[i, j] aligns the first i of A with the first j of B.

    Row i is built from row i-1 with vector ops. The left-move chain `D[i][j] = max(c[j], D[i][j-1]-gap)`
    is unrolled to `max_k<=j(c[k] + k*gap) - j*gap` (linear gaps telescope), so a row is O(len_b) numpy
    work instead of a Python loop over len_b."""
    len_a, len_b = score.shape
    table = np.empty((len_a + 1, len_b + 1), dtype=np.float64)
    table[0, :] = -gap * np.arange(len_b + 1)
    table[:, 0] = -gap * np.arange(len_a + 1)
    cols = np.arange(1, len_b + 1, dtype=np.float64)
    for i in range(1, len_a + 1):
        prev = table[i - 1]
        enter = np.maximum(prev[:-1] + score[i - 1], prev[1:] - gap)
        table[i, 1:] = np.maximum.accumulate(enter + gap * cols) - gap * cols
    return table


def _traceback(table: np.ndarray, score: np.ndarray, gap: float) -> Alignment:
    """Walk the DP table from the last cell back to (0, 0), reading off matches and gaps.

    `table` is stored after float round-trips through the telescoped form, so predecessors are matched
    with a tolerance (1e-9 is far above the ~1e-12 round-off and far below any meaningful score gap)."""
    len_a, len_b = score.shape
    matches: list[tuple[int, int]] = []
    only_a: list[int] = []
    only_b: list[int] = []
    i, j = len_a, len_b
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(table[i, j] - (table[i - 1, j - 1] + score[i - 1, j - 1])) < 1e-9:
            matches.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and abs(table[i, j] - (table[i - 1, j] - gap)) < 1e-9:
            only_a.append(i - 1)
            i -= 1
        else:
            only_b.append(j - 1)
            j -= 1
    return Alignment(
        matches=tuple(reversed(matches)),
        only_a=tuple(reversed(only_a)),
        only_b=tuple(reversed(only_b)),
        score=float(table[-1, -1]),
    )

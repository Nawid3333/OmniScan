"""Tests for the global sequence alignment used by chapter matching (page and chapter level)."""

from __future__ import annotations

import numpy as np
import pytest

from omniscan.match.align import Alignment, align

FORBIDDEN = -1e9


def score_matrix(pairs: dict[tuple[int, int], float], len_a: int, len_b: int) -> np.ndarray:
    """Score matrix from allowed (i, j) -> score pairs; every other pair is forbidden (-1e9)."""
    score = np.full((len_a, len_b), FORBIDDEN)
    for (i, j), value in pairs.items():
        score[i, j] = value
    return score


def test_identical_sequences_align_one_to_one() -> None:
    score = score_matrix({(i, i): 1.0 for i in range(4)}, 4, 4)
    result = align(score, gap=0.25)
    assert result.matches == ((0, 0), (1, 1), (2, 2), (3, 3))
    assert result.only_a == () and result.only_b == ()
    assert result.score == pytest.approx(4.0)


def test_offset_sequences_align_with_leading_gap() -> None:
    # A[0..3] correspond to B[1..4]: B has one extra early item
    score = score_matrix({(i, i + 1): 1.0 for i in range(4)}, 4, 5)
    result = align(score, gap=0.25)
    assert result.matches == ((0, 1), (1, 2), (2, 3), (3, 4))
    assert result.only_a == ()
    assert result.only_b == (0,)
    assert result.score == pytest.approx(4.0 - 0.25)


def test_item_only_in_a_becomes_a_gap_not_a_forced_match() -> None:
    # a2 has no counterpart; matching it to anything is forbidden
    score = score_matrix({(0, 0): 1.0, (1, 1): 1.0, (3, 2): 1.0}, 4, 3)
    result = align(score, gap=0.25)
    assert result.matches == ((0, 0), (1, 1), (3, 2))
    assert result.only_a == (2,)
    assert result.only_b == ()


def test_forbidden_pair_is_never_taken_even_at_the_end() -> None:
    # a2 could pair with b2 for a positive score, but it is gated out: it must become a gap
    score = score_matrix({(0, 0): 1.0, (1, 1): 1.0, (2, 2): 0.9}, 3, 3)
    score[2, 2] = FORBIDDEN
    result = align(score, gap=0.25)
    assert result.matches == ((0, 0), (1, 1))
    assert result.only_a == (2,) and result.only_b == (2,)


def test_good_match_outweighs_two_gaps_bad_match_does_not() -> None:
    # a0->b0 (0.9) is worth crossing one gap each side; a forbidden pair is not
    score = score_matrix({(1, 1): 0.9}, 2, 2)
    result = align(score, gap=0.25)
    assert result.matches == ((1, 1),)
    assert result.only_a == (0,) and result.only_b == (0,)
    assert result.score == pytest.approx(0.9 - 2 * 0.25)


def test_insertion_in_b_shifts_the_tail() -> None:
    # B has an extra item in the middle: b2 is only in B
    score = score_matrix({(0, 0): 1.0, (1, 1): 1.0, (2, 3): 1.0}, 3, 4)
    result = align(score, gap=0.25)
    assert result.matches == ((0, 0), (1, 1), (2, 3))
    assert result.only_a == () and result.only_b == (2,)


def test_weak_but_real_match_loses_to_a_better_partner() -> None:
    # a1 fits b1 (0.8) far better than b2 (0.3): the DP must not greedily grab b2 with a1
    score = score_matrix({(0, 0): 0.8, (1, 1): 0.8, (1, 2): 0.3, (2, 2): 0.8}, 3, 3)
    result = align(score, gap=0.25)
    assert result.matches == ((0, 0), (1, 1), (2, 2))
    assert result.score == pytest.approx(2.4)


def test_empty_sides_produce_all_gaps() -> None:
    empty = np.zeros((0, 3))
    result = align(empty, gap=0.5)
    assert result == Alignment((), (), (0, 1, 2), -1.5)
    other = np.zeros((3, 0))
    assert align(other, gap=0.5).only_a == (0, 1, 2)


def test_monotonicity_and_index_coverage() -> None:
    # a scattered but globally consistent score matrix: alignment must stay strictly increasing
    score = score_matrix({(0, 1): 0.9, (1, 3): 0.9, (2, 4): 0.9, (0, 0): 0.5, (1, 2): 0.5}, 3, 5)
    result = align(score, gap=0.25)
    a_indices = [i for i, _ in result.matches]
    b_indices = [j for _, j in result.matches]
    assert a_indices == sorted(a_indices) and len(set(a_indices)) == len(a_indices)
    assert b_indices == sorted(b_indices) and len(set(b_indices)) == len(b_indices)
    assert set(result.only_a) | {i for i, _ in result.matches} == set(range(3))
    assert set(result.only_b) | {j for _, j in result.matches} == set(range(5))

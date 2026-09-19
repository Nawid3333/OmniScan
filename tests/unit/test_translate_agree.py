"""Tests for omniscan.translate.agree (candidate agreement over normalised lines)."""

from __future__ import annotations

import random
import string
from itertools import combinations

import pytest

from omniscan.translate.agree import agreement, candidates_agree, normalize_line, similarity


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("“Seong-jin… Wait!”", "seong jin wait"),
        ("Hello,   World!!", "hello world"),
        ("  ", ""),
        ("Ａｂｃ ＆ ｄ", "abc d"),
        ("It's   fine — really?", "it s fine really"),
        ("…", ""),
        ("a+b=c", "a b c"),  # symbols (Unicode category S) are spaced out like punctuation
        ("$5 <3 ©", "5 3"),
        ("성진이가 왔다", "성진이가 왔다"),
    ],
)
def test_normalize_line_goldens(text: str, expected: str) -> None:
    assert normalize_line(text) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("I can't believe you're here.", "I can't believe you are here.", 0.9818),
        ("Get out!", "get out", 1.0),
        ("Run!", "Watch out behind you!", 0.1739),
        ("Seong-jin, wait!", "Sung-jin, wait!", 0.8889),
        ("", "", 1.0),
        ("", "Hi", 0.0),
        ("...", "!!", 1.0),
    ],
)
def test_similarity_goldens(a: str, b: str, expected: float) -> None:
    assert similarity(a, b) == pytest.approx(expected, rel=1e-4)
    assert similarity(b, a) == pytest.approx(expected, rel=1e-4)  # order-independent


def test_agreement_fewer_than_two_texts_is_one() -> None:
    assert agreement([]) == 1.0
    assert agreement(["Get out!"]) == 1.0


def test_agreement_is_minimum_pair() -> None:
    assert agreement(["Get out!", "get out", "Run!"]) == pytest.approx(0.2, rel=1e-4)
    assert agreement(["Same line", "Same line"]) == 1.0


def test_candidates_agree_threshold_boundary() -> None:
    assert candidates_agree(["Get out!", "get out", "Get out"], 0.9) is True
    assert candidates_agree(["Get out!", "get out", "Get out", "Run!"], 0.9) is False
    texts = ["a b", "a c"]
    assert candidates_agree(texts, agreement(texts)) is True  # boundary is inclusive


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_candidates_agree_rejects_out_of_range_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        candidates_agree(["a", "a"], threshold)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_candidates_agree_accepts_boundary_thresholds(threshold: float) -> None:
    assert candidates_agree(["a", "a"], threshold) is True


def test_similarity_properties_over_random_ascii() -> None:
    """Hypothesis-free properties over seeded random ASCII strings: range, symmetry, reflexivity."""
    rng = random.Random(20260919)
    alphabet = string.ascii_letters + string.digits + string.punctuation + " "
    texts = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60))) for _ in range(200)]
    for text in texts:
        assert similarity(text, text) == 1.0
        assert 0.0 <= similarity(text, "") <= 1.0
    for a, b in combinations(texts, 2):
        score = similarity(a, b)
        assert 0.0 <= score <= 1.0
        assert score == similarity(b, a)

"""Tests for omniscan.eval.metrics."""

from __future__ import annotations

import pytest

from omniscan.eval.metrics import cer, chrf, levenshtein, normalize


def test_normalize_strips_punctuation_and_spaces() -> None:
    assert normalize("Hello, World! 안녕... 하세요?  ABC-12") == "helloworld안녕하세요abc12"


def test_normalize_empty() -> None:
    assert normalize("") == ""


def test_levenshtein() -> None:
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0


def test_cer_golden_values() -> None:
    assert cer("안녕하세요", "안녕하세여") == 0.2
    assert cer("첫째 줄", "첫째줄") == 0.0
    assert cer("안녕", "안녕하세요") == 1.5
    assert cer("abc", "xyz") == 1.0
    assert cer("Hello!", "hello") == 0.0


def test_cer_empty_truth_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        cer("", "x")
    with pytest.raises(ValueError, match="empty"):
        cer("...", "x")


def test_chrf_golden_values() -> None:
    assert chrf("The cat sat on the mat", "The cat sat on the mat") == 1.0
    assert chrf("The cat sat on the mat", "The cat sits on a mat") == pytest.approx(0.434395, abs=1e-5)
    assert chrf("abc", "xyz") == 0.0
    assert chrf("ab", "ab") == 1.0
    assert chrf("Hello, world!", "hello world") == 1.0
    assert chrf(
        "I must have fallen asleep",
        "I must have fallen asleep with the window open",
    ) == pytest.approx(0.575235, abs=1e-5)


def test_chrf_is_not_symmetric() -> None:
    assert chrf("a b c d e f", "a b") != chrf("a b", "a b c d e f")


def test_chrf_empty_inputs() -> None:
    assert chrf("", "") == 0.0
    assert chrf("", "abc") == 0.0

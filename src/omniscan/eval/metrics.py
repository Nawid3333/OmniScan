"""Text metrics for `omniscan eval`: normalisation, edit distance, CER and chrF."""

from __future__ import annotations

import unicodedata
from collections import Counter


def normalize(text: str) -> str:
    """NFKC, lower-case, keep only letters and numbers (punctuation, spaces and marks drop out)."""
    folded = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in folded if unicodedata.category(ch)[0] in ("L", "N"))


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert / delete / substitute, each cost 1)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(truth: str, read: str) -> float:
    """Character error rate: edit distance of the normalised texts over the truth length."""
    expected = normalize(truth)
    if not expected:
        raise ValueError("cer: normalised truth is empty")
    return levenshtein(expected, normalize(read)) / len(expected)


def chrf(hypothesis: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Character F-score (sacreBLEU-style, no whitespace tokenisation) of two texts."""
    hyp = normalize(hypothesis)
    ref = normalize(reference)
    precisions: list[float] = []
    recalls: list[float] = []
    for n in range(1, max_n + 1):
        hyp_n = Counter(hyp[i : i + n] for i in range(len(hyp) - n + 1))
        ref_n = Counter(ref[i : i + n] for i in range(len(ref) - n + 1))
        total_hyp = sum(hyp_n.values())
        total_ref = sum(ref_n.values())
        if total_hyp == 0 or total_ref == 0:
            continue
        match = sum((hyp_n & ref_n).values())
        precisions.append(match / total_hyp)
        recalls.append(match / total_ref)
    if not precisions:
        return 0.0
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision + recall == 0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

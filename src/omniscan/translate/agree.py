"""Candidate agreement: minimum pairwise similarity between translations of one line."""

from __future__ import annotations

import difflib
import unicodedata
from collections.abc import Sequence
from itertools import combinations


def normalize_line(text: str) -> str:
    """Casefolded NFKC text with punctuation/symbols spaced out and whitespace collapsed."""
    flat = unicodedata.normalize("NFKC", text).casefold()
    spaced = "".join(" " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in flat)
    return " ".join(spaced.split())


def similarity(a: str, b: str) -> float:
    """Pairwise line similarity in [0.0, 1.0] (difflib ratio over normalised lines; order-independent)."""
    na, nb = sorted((normalize_line(a), normalize_line(b)))
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio()


def agreement(texts: Sequence[str]) -> float:
    """The minimum pairwise similarity over `texts`; 1.0 for fewer than two texts."""
    return min((similarity(a, b) for a, b in combinations(texts, 2)), default=1.0)


def candidates_agree(texts: Sequence[str], threshold: float) -> bool:
    """True when every pair of candidate translations is at least `threshold` similar."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold {threshold!r} outside [0.0, 1.0]")
    return agreement(texts) >= threshold

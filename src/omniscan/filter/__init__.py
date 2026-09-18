"""Promo filter: detect scanlator promo pages/slices against user-supplied examples via perceptual hash."""

from omniscan.filter.decide import (
    ExampleHash,
    best_match,
    decide_files,
    decide_slices,
    effective_decision,
    load_examples,
    restore,
)
from omniscan.filter.hashing import dhash, hamming, similarity

__all__ = [
    "ExampleHash",
    "best_match",
    "decide_files",
    "decide_slices",
    "dhash",
    "effective_decision",
    "hamming",
    "load_examples",
    "restore",
    "similarity",
]

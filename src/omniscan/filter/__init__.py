"""Promo filter: detect scanlator promo pages/slices against user-supplied examples via perceptual hash."""

from omniscan.filter.apply import (
    EMPTY_OVERRIDES,
    Overrides,
    apply_slice_filter,
    example_files,
    examples_fingerprint,
    file_verdict,
    load_overrides,
    record_override,
    slice_verdict,
)
from omniscan.filter.decide import (
    ExampleHash,
    best_match,
    decide_files,
    decide_slices,
    effective_decision,
    load_examples,
    restore,
)
from omniscan.filter.hashing import dhash, dhash_tensor, hamming, similarity

__all__ = [
    "EMPTY_OVERRIDES",
    "ExampleHash",
    "Overrides",
    "apply_slice_filter",
    "best_match",
    "decide_files",
    "decide_slices",
    "dhash",
    "dhash_tensor",
    "effective_decision",
    "example_files",
    "examples_fingerprint",
    "file_verdict",
    "hamming",
    "load_examples",
    "load_overrides",
    "record_override",
    "restore",
    "similarity",
    "slice_verdict",
]

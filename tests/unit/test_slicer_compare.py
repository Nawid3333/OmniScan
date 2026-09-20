"""Unit tests for compare_strategies and summarize (card S2)."""

import pytest

from omniscan.core.config import SlicerConfig
from omniscan.core.schemas import Slice, SlicesArtifact
from omniscan.slicer import STRATEGIES, compare_strategies, summarize
from tests.fixtures.strip_layouts import random_strip


def _artifact(slices: list[tuple[int, int, bool, bool]]) -> SlicesArtifact:
    """A hand-made artifact whose slices are (y0, y1, forced_cut, blank)."""
    return SlicesArtifact(
        strip_width=10,
        strip_height=slices[-1][1] if slices else 0,
        bands=[],
        slices=[
            Slice(index=i, y0=y0, y1=y1, forced_cut=forced, blank=blank)
            for i, (y0, y1, forced, blank) in enumerate(slices)
        ],
        params={},
    )


def test_compare_returns_artifacts_in_strategies_order() -> None:
    strip, _ = random_strip(0)
    cfg = SlicerConfig()
    result = compare_strategies(strip, cfg)
    assert list(result) == list(STRATEGIES)
    for name, artifact in result.items():
        assert artifact.params["strategy"] == name

    only = compare_strategies(strip, cfg, strategies=("fixed",))
    assert list(only) == ["fixed"]

    pair = compare_strategies(strip, cfg, strategies=("fixed", "smart"))
    assert list(pair) == ["fixed", "smart"]


def test_compare_unknown_strategy_propagates() -> None:
    strip, _ = random_strip(0)
    with pytest.raises(ValueError, match="unknown slicer strategy"):
        compare_strategies(strip, SlicerConfig(), strategies=("fixed", "nope"))


def test_summarize_counts_and_heights() -> None:
    artifact = _artifact([(0, 100, False, True), (100, 300, True, False), (300, 500, True, False)])
    summary = summarize("smart", artifact)
    assert summary.strategy == "smart"
    assert summary.slices == 3
    assert (summary.min_height, summary.median_height, summary.max_height) == (100, 200, 200)
    assert (summary.forced, summary.blank) == (2, 1)
    assert summary.cuts == (100, 300)


def test_summarize_median_even_count() -> None:
    slices = [
        (0, 100, False, False),
        (100, 300, False, False),
        (300, 600, False, False),
        (600, 1000, False, False),
    ]
    assert summarize("fixed", _artifact(slices)).median_height == 250  # heights 100..400: (200 + 300) / 2


def test_summarize_empty_artifact() -> None:
    summary = summarize("page", _artifact([]))
    assert summary.slices == 0
    assert summary.min_height == summary.median_height == summary.max_height == 0
    assert summary.forced == 0
    assert summary.blank == 0
    assert summary.cuts == ()


def test_summarize_is_frozen() -> None:
    summary = summarize("smart", _artifact([(0, 100, False, False)]))
    with pytest.raises(Exception, match="cannot assign to field"):
        summary.slices = 5  # type: ignore[misc]

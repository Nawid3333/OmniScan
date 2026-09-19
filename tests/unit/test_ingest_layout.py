"""Tests for layout.py: dominant_width / stack_layout."""

from __future__ import annotations

import pytest

from omniscan.ingest.layout import dominant_width, stack_layout


def test_dominant_width_picks_most_frequent() -> None:
    assert dominant_width([800, 800, 800, 1000]) == 800


def test_dominant_width_tie_breaks_to_larger() -> None:
    assert dominant_width([800, 800, 1000, 1000]) == 1000


def test_dominant_width_empty_raises() -> None:
    with pytest.raises(ValueError, match="widths is empty"):
        dominant_width([])


def test_stack_layout_stacks_cumulatively() -> None:
    assert stack_layout([(800, 1000), (1600, 2000), (800, 500)], strip_width=800) == [
        (1.0, 0, 1000),
        (0.5, 1000, 2000),
        (1.0, 2000, 2500),
    ]


def test_stack_layout_height_never_below_one() -> None:
    layout = stack_layout([(10000, 1)], strip_width=100)
    assert layout == [(0.01, 0, 1)]


def test_stack_layout_y_ranges_are_contiguous() -> None:
    sizes = [(400, 300), (800, 600), (1200, 900)]
    layout = stack_layout(sizes, strip_width=800)
    assert [y0 for _, y0, _ in layout][1:] == [y1 for _, _, y1 in layout][:-1]
    assert layout[-1][2] == 600 + 600 + 600  # each scaled to strip_width=800


def test_stack_layout_rounds_to_nearest() -> None:
    """A fractional scaled height (99 * 0.625 = 61.875) rounds to 62, not floor-truncated to 61."""
    assert stack_layout([(480, 99)], strip_width=300) == [(0.625, 0, 62)]

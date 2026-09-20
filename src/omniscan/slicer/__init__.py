"""Slicer core: uniform-band detection, cut planning, slice extraction, strategies and compare mode."""

from omniscan.slicer.bands import RowStats, find_uniform_bands, row_stats
from omniscan.slicer.compare import StrategySummary, compare_strategies, summarize
from omniscan.slicer.cuts import Cut, plan_cuts
from omniscan.slicer.page_mode import slice_by_pages
from omniscan.slicer.slice import slice_strip
from omniscan.slicer.strategies import (
    STRATEGIES,
    gutter_cut_rows,
    slice_fixed,
    slice_page,
    slice_simple_gutter,
    slice_with_strategy,
)

__all__ = [
    "STRATEGIES",
    "Cut",
    "RowStats",
    "StrategySummary",
    "compare_strategies",
    "find_uniform_bands",
    "gutter_cut_rows",
    "plan_cuts",
    "row_stats",
    "slice_by_pages",
    "slice_fixed",
    "slice_page",
    "slice_simple_gutter",
    "slice_strip",
    "slice_with_strategy",
    "summarize",
]

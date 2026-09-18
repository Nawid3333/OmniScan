"""Slicer core: uniform-band detection, cut planning and slice extraction."""

from omniscan.slicer.bands import RowStats, find_uniform_bands, row_stats
from omniscan.slicer.cuts import Cut, plan_cuts
from omniscan.slicer.page_mode import slice_by_pages
from omniscan.slicer.slice import slice_strip

__all__ = [
    "Cut",
    "RowStats",
    "find_uniform_bands",
    "plan_cuts",
    "row_stats",
    "slice_by_pages",
    "slice_strip",
]

"""Compare mode: run several slicer strategies over one strip and summarise where each would cut."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from omniscan.core.config import SlicerConfig
from omniscan.core.schemas import SlicesArtifact, SourceFile
from omniscan.slicer.strategies import STRATEGIES, slice_with_strategy


@dataclass(frozen=True, slots=True)
class StrategySummary:
    """What one strategy would do to a strip: slice sizes and the y of every internal boundary."""

    strategy: str
    slices: int
    min_height: int
    median_height: int
    max_height: int
    forced: int
    blank: int
    cuts: tuple[int, ...]


def compare_strategies(
    strip: torch.Tensor,
    cfg: SlicerConfig,
    source_files: list[SourceFile] | None = None,
    strategies: Sequence[str] = STRATEGIES,
) -> dict[str, SlicesArtifact]:
    """Run every named strategy over the strip, in order (an unknown name raises ValueError)."""
    return {name: slice_with_strategy(strip, cfg, source_files, strategy=name) for name in strategies}


def summarize(strategy: str, artifact: SlicesArtifact) -> StrategySummary:
    """Count slices/forced/blank and the height stats of one artifact (all zeros when empty)."""
    heights = [s.y1 - s.y0 for s in artifact.slices]
    return StrategySummary(
        strategy=strategy,
        slices=len(heights),
        min_height=min(heights) if heights else 0,
        median_height=int(statistics.median(heights)) if heights else 0,
        max_height=max(heights) if heights else 0,
        forced=sum(1 for s in artifact.slices if s.forced_cut),
        blank=sum(1 for s in artifact.slices if s.blank),
        cuts=tuple(s.y1 for s in artifact.slices[:-1]),
    )

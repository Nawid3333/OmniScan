"""Persistence of a series' fixed-position watermark regions (watermarks.json in the series work dir)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_FILE_NAME = "watermarks.json"


class WatermarkRegion(BaseModel):
    """A watermark's position as a fraction of EVERY raw page's own width/height (pydantic v2)."""

    model_config = ConfigDict(extra="forbid")

    index: int  # position in the series' list, assigned by the store (0-based, stable until removal)
    x0_frac: float  # 0.0..1.0, x0_frac < x1_frac
    y0_frac: float  # 0.0..1.0, y0_frac < y1_frac
    x1_frac: float
    y1_frac: float
    note: str | None = None


def _validate_fractions(x0_frac: float, y0_frac: float, x1_frac: float, y1_frac: float) -> None:
    """Raise ValueError naming the first violated bound unless 0<=x0<x1<=1 and 0<=y0<y1<=1."""
    for lo_name, lo, hi_name, hi in (
        ("x0_frac", x0_frac, "x1_frac", x1_frac),
        ("y0_frac", y0_frac, "y1_frac", y1_frac),
    ):
        if lo < 0.0:
            raise ValueError(f"{lo_name} {lo} outside [0.0, 1.0]")
        if hi > 1.0:
            raise ValueError(f"{hi_name} {hi} outside [0.0, 1.0]")
        if not lo < hi:
            raise ValueError(f"{lo_name} {lo} must be < {hi_name} {hi}")


class WatermarkStore:
    """CRUD over the series' watermarks.json; atomic saves, indices never reused."""

    def __init__(self, series_work_dir: Path) -> None:
        self._path = series_work_dir / _FILE_NAME

    def list(self) -> list[WatermarkRegion]:
        """All regions ordered by index; empty if the file does not exist (nothing is created)."""
        if not self._path.is_file():
            return []
        regions = [
            WatermarkRegion.model_validate(item) for item in json.loads(self._path.read_text())["regions"]
        ]
        return sorted(regions, key=lambda region: region.index)

    def add(
        self,
        x0_frac: float,
        y0_frac: float,
        x1_frac: float,
        y1_frac: float,
        note: str | None = None,
    ) -> WatermarkRegion:
        """Validate the fractions, assign the next unused index, persist and return the new region."""
        _validate_fractions(x0_frac, y0_frac, x1_frac, y1_frac)
        regions = self.list()
        region = WatermarkRegion(
            index=max((existing.index for existing in regions), default=-1) + 1,
            x0_frac=x0_frac,
            y0_frac=y0_frac,
            x1_frac=x1_frac,
            y1_frac=y1_frac,
            note=note,
        )
        regions.append(region)
        self._save(regions)
        return region

    def remove(self, index: int) -> None:
        """Drop the region with this index (remaining indices keep their values); KeyError if absent."""
        regions = self.list()
        remaining = [region for region in regions if region.index != index]
        if len(remaining) == len(regions):
            raise KeyError(index)
        self._save(remaining)

    def _save(self, regions: list[WatermarkRegion]) -> None:
        """Atomically write the regions (tmp file + rename, same pattern as core.schemas.Artifact.save)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = {"regions": [region.model_dump() for region in regions]}
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)

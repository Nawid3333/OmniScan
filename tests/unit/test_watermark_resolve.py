"""Tests for omniscan.watermark.resolve (strip-space resolution against synthetic ingest artifacts)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from omniscan.core.schemas import BBox, IngestArtifact, SourceFile
from omniscan.watermark.resolve import resolve_watermark_regions
from omniscan.watermark.store import WatermarkRegion


def source_file(
    width: int = 1000,
    y0: int = 0,
    y1: int = 1400,
    scale: float = 1.0,
    index: int = 0,
) -> SourceFile:
    """A synthetic SourceFile covering rows [y0, y1) of the strip."""
    return SourceFile(
        index=index,
        name=f"p{index:03d}.jpg",
        sha256="0" * 64,
        width=width,
        height=round((y1 - y0) / scale),
        y0=y0,
        y1=y1,
        scale=scale,
    )


def ingest(files: Sequence[SourceFile] = ()) -> IngestArtifact:
    """A synthetic IngestArtifact wrapping the given files."""
    return IngestArtifact(
        series="test",
        chapter="1",
        strip_width=800,
        strip_height=files[-1].y1 if files else 0,
        files=list(files),
    )


def region(
    x0_frac: float = 0.9, y0_frac: float = 0.0, x1_frac: float = 1.0, y1_frac: float = 0.05
) -> WatermarkRegion:
    """A WatermarkRegion with the given fractions (default: top-right corner)."""
    return WatermarkRegion(index=0, x0_frac=x0_frac, y0_frac=y0_frac, x1_frac=x1_frac, y1_frac=y1_frac)


def test_single_region_single_file_scale_1() -> None:
    boxes = resolve_watermark_regions([region()], ingest([source_file(scale=1.0, width=1000)]))
    assert boxes == [BBox(x0=900, y0=0, x1=1000, y1=70)]


def test_scale_halves_x_but_not_y() -> None:
    boxes = resolve_watermark_regions([region()], ingest([source_file(scale=0.5, width=1000)]))
    assert boxes == [BBox(x0=450, y0=0, x1=500, y1=70)]


def test_y_fractions_interpolate_within_the_files_own_span() -> None:
    boxes = resolve_watermark_regions([region()], ingest([source_file(y0=1400, y1=2800)]))
    assert boxes == [BBox(x0=900, y0=1400, x1=1000, y1=1470)]


def test_two_regions_two_files_nest_region_outer_file_inner() -> None:
    files = [source_file(index=0), source_file(index=1, y0=1400, y1=2800)]
    regions = [region(y1_frac=0.1), region(y0_frac=0.5, y1_frac=0.6)]
    boxes = resolve_watermark_regions(regions, ingest(files))
    assert boxes == [
        BBox(x0=900, y0=0, x1=1000, y1=140),
        BBox(x0=900, y0=1400, x1=1000, y1=1540),
        BBox(x0=900, y0=700, x1=1000, y1=840),
        BBox(x0=900, y0=2100, x1=1000, y1=2240),
    ]


def test_degenerate_pair_omitted_others_kept() -> None:
    thin = source_file(index=1, y0=1400, y1=1401)
    regions = [region(y0_frac=0.4, y1_frac=0.49)]  # rounds to zero height on a 1-row span
    boxes = resolve_watermark_regions(regions, ingest([source_file(), thin]))
    assert boxes == [BBox(x0=900, y0=560, x1=1000, y1=686)]


def test_empty_regions_or_empty_files_return_empty() -> None:
    assert resolve_watermark_regions([], ingest([source_file()])) == []
    assert resolve_watermark_regions([region()], ingest()) == []


@pytest.mark.parametrize(
    "regions",
    [[region()], []],
)
def test_resolution_does_not_mutate_inputs(regions: list[WatermarkRegion]) -> None:
    page = ingest([source_file()])
    regions_before = [r.model_dump() for r in regions]
    files_before = [f.model_dump() for f in page.files]
    resolve_watermark_regions(regions, page)
    assert [r.model_dump() for r in regions] == regions_before
    assert [f.model_dump() for f in page.files] == files_before

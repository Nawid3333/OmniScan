"""Page-mode passthrough slicing: one Slice per SourceFile, no band finding."""

from __future__ import annotations

from itertools import pairwise

from omniscan.core.schemas import Slice, SlicesArtifact, SourceFile


def slice_by_pages(source_files: list[SourceFile], strip_width: int, strip_height: int) -> SlicesArtifact:
    """One Slice per SourceFile, boundaries exactly at that file's [y0, y1); no bands, no cut planning."""
    if not source_files:
        if strip_height != 0:
            raise ValueError(
                f"page mode got no source files but strip_height={strip_height}: "
                "pages must tile [0, strip_height) exactly"
            )
        return SlicesArtifact(
            strip_width=strip_width,
            strip_height=0,
            bands=[],
            slices=[],
            params={"mode": "page"},
        )

    for a, b in pairwise(source_files):
        if b.y0 < a.y0:
            raise ValueError(
                f"pages out of order: {a.name!r} (index {a.index}, y0={a.y0}) precedes "
                f"{b.name!r} (index {b.index}, y0={b.y0}); page mode requires files sorted by y0 ascending"
            )

    first = source_files[0]
    if first.y0 != 0:
        raise ValueError(
            f"first page {first.name!r} (index {first.index}) starts at y0={first.y0}, expected 0: "
            "pages must start at the top of the strip"
        )

    for a, b in pairwise(source_files):
        if b.y0 != a.y1:
            kind = "overlap" if b.y0 < a.y1 else "gap"
            raise ValueError(
                f"{kind} of {abs(b.y0 - a.y1)}px between page {a.name!r} (index {a.index}, "
                f"y1={a.y1}) and page {b.name!r} (index {b.index}, y0={b.y0}): "
                "page mode requires pages to tile the strip exactly"
            )

    last = source_files[-1]
    if last.y1 != strip_height:
        raise ValueError(
            f"last page {last.name!r} (index {last.index}) ends at y1={last.y1}, "
            f"expected strip_height={strip_height}: pages must reach the bottom of the strip"
        )

    slices = [Slice(index=i, y0=f.y0, y1=f.y1, source_files=[f.index]) for i, f in enumerate(source_files)]
    return SlicesArtifact(
        strip_width=strip_width,
        strip_height=strip_height,
        bands=[],
        slices=slices,
        params={"mode": "page"},
    )

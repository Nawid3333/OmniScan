"""Text blocks from another translation tool's project (BallonsTranslator, manga-image-translator), matched to
a chapter's regions.

A block is a box in its page's own pixels with a source text and a translation. It maps into the strip through
the page's scale, and belongs to the region whose text box it overlaps most (at least half of the smaller of the
two boxes), else to the region whose text box or balloon holds its centre, as a LabelPlus label would; watermarks
never take a block (pure functions, no I/O).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from omniscan.core.schemas import BBox, IngestArtifact, Region, SourceFile
from omniscan.interchange.labelplus import find_page, region_at, strip_pages

MIN_OVERLAP = 0.5  # of the smaller box: a block and a region this much on top of each other are the same text
_NO_SPACE = re.compile(r"[぀-ヿ㐀-䶿一-鿿＀-￯]")  # kana, Han, full-width


@dataclass(frozen=True, slots=True)
class Block:
    """One text block of a foreign project: its page (image file name), box in that page's pixels, texts."""

    page: str
    x0: float
    y0: float
    x1: float
    y1: float
    text: str = ""
    translation: str = ""


@dataclass(slots=True)
class BlockMatch:
    """Where a project's blocks land in a chapter."""

    matched: dict[str, list[Block]] = field(default_factory=dict)  # region id -> its blocks, in file order
    unmatched: list[tuple[Block, BBox]] = field(default_factory=list)  # blocks over no region, with strip box
    unknown_pages: list[str] = field(default_factory=list)  # pages the chapter does not have


def join_lines(lines: Sequence[str]) -> str:
    """A block's text lines as one text: a space between two lines unless either side is Japanese or Chinese
    (which are written without spaces); blank lines dropped."""
    out = ""
    for line in (line.strip() for line in lines):
        if not line:
            continue
        if out and not (_NO_SPACE.match(out[-1]) or _NO_SPACE.match(line[0])):
            out += " "
        out += line
    return out


def strip_box(block: Block, page: SourceFile, strip_width: int) -> BBox | None:
    """The block's box in strip pixels, clipped to its page; None when nothing of it is on the page."""
    x0 = max(0, min(round(block.x0 * page.scale), strip_width))
    x1 = max(0, min(round(block.x1 * page.scale), strip_width))
    y0 = max(page.y0, min(page.y0 + round(block.y0 * page.scale), page.y1))
    y1 = max(page.y0, min(page.y0 + round(block.y1 * page.scale), page.y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def overlap(a: BBox, b: BBox) -> float:
    """The area two boxes share, as a fraction of the smaller one."""
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.y1, b.y1) - max(a.y0, b.y0)
    if width <= 0 or height <= 0:
        return 0.0
    return width * height / min(a.width * a.height, b.width * b.height)


def region_for(regions: Sequence[Region], box: BBox) -> Region | None:
    """The region a block with strip box `box` belongs to (see the module docstring), or None."""
    candidates = [r for r in regions if r.kind != "watermark"]
    best = max(candidates, key=lambda r: overlap(r.bbox, box), default=None)
    if best is not None and overlap(best.bbox, box) >= MIN_OVERLAP:
        return best
    return region_at(candidates, (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)


def match_blocks(blocks: Sequence[Block], ingest: IngestArtifact, regions: Sequence[Region]) -> BlockMatch:
    """Map every block with a text or a translation to the region it belongs to."""
    pages = strip_pages(ingest)
    found = BlockMatch()
    for block in blocks:
        if not (block.text.strip() or block.translation.strip()):
            continue
        page = find_page(pages, block.page)
        if page is None:
            if block.page not in found.unknown_pages:
                found.unknown_pages.append(block.page)
            continue
        box = strip_box(block, page, ingest.strip_width)
        if box is None:
            continue
        region = region_for(regions, box)
        if region is None:
            found.unmatched.append((block, box))
        else:
            found.matched.setdefault(region.id, []).append(block)
    return found

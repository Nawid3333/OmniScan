"""LabelPlus translation files (.txt) — the labelling format many scanlation groups translate and proofread in.

A LabelPlus file lists, per page image, numbered labels: a point (x, y as fractions of the image's width and
height), a group (1 = 框内, text inside a balloon; 2 = 框外, text outside) and the label's text. A chapter is
exported as one label per region at the centre of its text box; a file's texts come back as the English lines
of the regions its labels point into (pure functions, no I/O).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePath

from omniscan.core.schemas import BBox, IngestArtifact, Region, SourceFile

GROUPS = ("框内", "框外")  # LabelPlus' own default groups: inside a balloon, outside
VERSION = "1,0"
_PAGE = re.compile(r"^>{3,}\[(.+)\]<{3,}$")
_LABEL = re.compile(r"^-{3,}\[(\d+)\]-{3,}\[(.*)\]$")


@dataclass(frozen=True, slots=True)
class Label:
    """One label: its number on the page, a point (fractions of the page) and a group (1-based)."""

    number: int
    x: float
    y: float
    group: int
    text: str


@dataclass(slots=True)
class LabelFile:
    """A LabelPlus file: its groups, its comment and the labels of every page (file order)."""

    groups: list[str] = field(default_factory=lambda: list(GROUPS))
    comment: str = ""
    pages: dict[str, list[Label]] = field(default_factory=dict)


def _coords(raw: str, line_no: int) -> tuple[float, float, int]:
    """x, y and group of a label marker's bracket ("0.402,0.211,1"); ValueError naming the line."""
    parts = [part.strip() for part in raw.split(",")]
    try:
        x, y = float(parts[0]), float(parts[1])
        group = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    except (IndexError, ValueError) as exc:
        raise ValueError(f"line {line_no}: label position {raw!r} is not 'x,y,group'") from exc
    return x, y, group


def parse(text: str) -> LabelFile:
    """Read a LabelPlus file (a BOM and CRLF line ends are fine); ValueError when it is not one."""
    lines = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    first_page = next((i for i, line in enumerate(lines) if _PAGE.match(line.strip())), None)
    if first_page is None:
        raise ValueError("not a LabelPlus file: no page marker such as >>>>>>>>[001.jpg]<<<<<<<<")
    header = lines[:first_page]
    separators = [i for i, line in enumerate(header) if line.strip() == "-"]
    doc = LabelFile()
    if len(separators) >= 2:
        doc.groups = [line.strip() for line in header[separators[0] + 1 : separators[1]] if line.strip()]
        doc.comment = "\n".join(header[separators[1] + 1 :]).strip("\n")
    page: list[Label] | None = None
    label: tuple[int, float, float, int] | None = None
    body: list[str] = []

    def close() -> None:
        if page is not None and label is not None:
            number, x, y, group = label
            page.append(Label(number, x, y, group, "\n".join(body).strip("\n")))

    for line_no, line in enumerate(lines[first_page:], start=first_page + 1):
        page_match, label_match = _PAGE.match(line.strip()), _LABEL.match(line.strip())
        if page_match:
            close()
            page, label, body = doc.pages.setdefault(page_match.group(1), []), None, []
        elif label_match:
            close()
            label, body = (int(label_match.group(1)), *_coords(label_match.group(2), line_no)), []
        elif label is not None:
            body.append(line)
    close()
    return doc


def write(doc: LabelFile) -> str:
    """The LabelPlus text of `doc` (LF line ends; positions to three decimals, as LabelPlus writes them)."""
    out = [VERSION, "-", *doc.groups, "-", *(doc.comment.split("\n") if doc.comment else []), "", ""]
    for name, labels in doc.pages.items():
        out.append(f">>>>>>>>[{name}]<<<<<<<<")
        for label in labels:
            out.append(
                f"----------------[{label.number}]----------------[{label.x:.3f},{label.y:.3f},{label.group}]"
            )
            out.extend([label.text, ""])
        out.append("")
    return "\n".join(out)


def strip_pages(ingest: IngestArtifact) -> list[SourceFile]:
    """The chapter's pages that are part of the strip, top to bottom."""
    return [f for f in ingest.files if not f.filtered and f.y1 > f.y0]


def page_at(pages: Sequence[SourceFile], y: float) -> SourceFile | None:
    """The page holding strip row `y`."""
    return next((f for f in pages if f.y0 <= y < f.y1), pages[-1] if pages and y == pages[-1].y1 else None)


def _centre(box: BBox) -> tuple[float, float]:
    """The centre of a box in strip pixels."""
    return (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2


def export_labels(
    ingest: IngestArtifact, regions: Sequence[Region], texts: Mapping[str, str], *, comment: str = ""
) -> LabelFile:
    """One label per region with a text in `texts` (watermarks never), at the centre of its text box on the page
    that holds it; group 1 for balloon text, 2 for everything else. Every page is listed, labelled or not."""
    pages = strip_pages(ingest)
    doc = LabelFile(comment=comment, pages={f.name: [] for f in pages})
    for region in regions:
        text = texts.get(region.id, "").strip()
        if region.kind == "watermark" or not text:
            continue
        cx, cy = _centre(region.bbox)
        page = page_at(pages, cy)
        if page is None:
            continue
        labels = doc.pages[page.name]
        labels.append(
            Label(
                number=len(labels) + 1,
                x=cx / ingest.strip_width,
                y=(cy - page.y0) / (page.y1 - page.y0),
                group=1 if region.kind == "bubble_text" else 2,
                text=text,
            )
        )
    return doc


@dataclass(slots=True)
class Matched:
    """What a LabelPlus file's labels point at in a chapter."""

    texts: dict[str, str] = field(default_factory=dict)  # region id -> text (labels in one region joined)
    unmatched: list[tuple[str, Label]] = field(default_factory=list)  # (page, label) outside every region
    unknown_pages: list[str] = field(default_factory=list)  # pages the chapter does not have
    joined: int = 0  # labels added to a region another label already pointed into


def find_page(pages: Sequence[SourceFile], name: str) -> SourceFile | None:
    """The page named `name`, else the one with the same file stem (a group may have renamed 001.jpg to .png)."""
    exact = next((f for f in pages if f.name == name), None)
    if exact is not None:
        return exact
    stem = PurePath(name).stem.casefold()
    return next((f for f in pages if PurePath(f.name).stem.casefold() == stem), None)


def _area(box: BBox) -> int:
    return box.width * box.height


def _contains(box: BBox, x: float, y: float) -> bool:
    return box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1


def region_at(regions: Sequence[Region], x: float, y: float) -> Region | None:
    """The region a label at strip point (x, y) points into: the smallest text box holding it, else the smallest
    balloon holding it; watermarks never."""
    candidates = [r for r in regions if r.kind != "watermark"]
    in_text = [r for r in candidates if _contains(r.bbox, x, y)]
    if in_text:
        return min(in_text, key=lambda r: _area(r.bbox))
    in_bubble = [r for r in candidates if r.bubble_bbox is not None and _contains(r.bubble_bbox, x, y)]
    return min(in_bubble, key=lambda r: _area(r.bubble_bbox or r.bbox)) if in_bubble else None


def match_labels(doc: LabelFile, ingest: IngestArtifact, regions: Sequence[Region]) -> Matched:
    """Map every non-empty label of `doc` to the region it points into (its text on one line)."""
    pages = strip_pages(ingest)
    found = Matched()
    for name, labels in doc.pages.items():
        page = find_page(pages, name)
        if page is None:
            if any(label.text.strip() for label in labels):
                found.unknown_pages.append(name)
            continue
        for label in labels:
            text = " ".join(label.text.split())
            if not text:
                continue
            x, y = label.x * ingest.strip_width, page.y0 + label.y * (page.y1 - page.y0)
            region = region_at(regions, x, y)
            if region is None:
                found.unmatched.append((name, label))
            elif region.id in found.texts:
                found.texts[region.id] += " " + text
                found.joined += 1
            else:
                found.texts[region.id] = text
    return found

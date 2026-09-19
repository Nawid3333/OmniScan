"""Ground truth for `omniscan eval`: Inkscape text-layer SVGs (Pepper&Carrot) as TruthBox lists.

The translated-check folder holds, per chapter, `truth/<lang>/EnnPpp.svg` text layers with one
`flowRoot` per text box (box = `flowRegion/rect`, lines = `flowPara` texts). `parse_svg` maps those
boxes into the strip space of the ingest artifact: SVG pixels are scaled by `file_width / svg_width`,
cropped at the top (the lettered low-res pages lose the SVG's top margin), then clamped to the page
and finally mapped by the source file's strip scale/offset.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path

from omniscan.core.schemas import BBox, IngestArtifact

# (a, b, c, d, e, f): x' = a*x + c*y + e, y' = b*x + d*y + f (SVG matrix order).
_Matrix = tuple[float, float, float, float, float, float]

_IDENTITY: _Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

_TRANSFORM_RE = re.compile(r"([a-zA-Z]+)\s*\(([^)]*)\)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True, slots=True)
class TruthBox:
    """One ground-truth text box of one page, in strip space."""

    page: int  # page number from the SVG file name (E06P01 -> 1)
    bbox: BBox
    lines: tuple[str, ...]  # the flowPara texts, stripped, non-empty, document order

    @property
    def text(self) -> str:
        """The box's full text: its lines joined by one space."""
        return " ".join(self.lines)


@dataclass(frozen=True, slots=True)
class TruthStats:
    """Coverage of the truth dir for one chapter."""

    pages: int  # non-filtered source files with an integer stem that were considered
    pages_without_truth: int  # of those, the pages with no `E*Pnn.svg` for the language
    dropped: int  # truth boxes removed because they fall outside their page


def _mul(outer: _Matrix, inner: _Matrix) -> _Matrix:
    """Matrix product outer @ inner (inner is applied to points first)."""
    a1, b1, c1, d1, e1, f1 = outer
    a2, b2, c2, d2, e2, f2 = inner
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _parse_transform(value: str | None) -> _Matrix:
    """One SVG `transform` attribute as a matrix; identity when absent or unrecognised."""
    if not value:
        return _IDENTITY
    result = _IDENTITY
    for name, args_text in _TRANSFORM_RE.findall(value):
        args = [float(n) for n in _NUMBER_RE.findall(args_text)]
        kind = name.lower()
        if kind == "matrix" and len(args) == 6:
            matrix: _Matrix = (args[0], args[1], args[2], args[3], args[4], args[5])
        elif kind == "translate" and len(args) in (1, 2):
            matrix = (1.0, 0.0, 0.0, 1.0, args[0], args[1] if len(args) == 2 else 0.0)
        elif kind == "scale" and len(args) in (1, 2):
            sy = args[1] if len(args) == 2 else args[0]
            matrix = (args[0], 0.0, 0.0, sy, 0.0, 0.0)
        elif kind == "rotate" and len(args) in (1, 3):
            angle = math.radians(args[0])
            cos, sin = math.cos(angle), math.sin(angle)
            matrix = (cos, sin, -sin, cos, 0.0, 0.0)
            if len(args) == 3:
                matrix = _mul(
                    _mul((1.0, 0.0, 0.0, 1.0, args[1], args[2]), matrix),
                    (1.0, 0.0, 0.0, 1.0, -args[1], -args[2]),
                )
        else:
            return _IDENTITY
        result = _mul(result, matrix)  # right to left: the last list entry applies first
    return result


def _local(tag: object) -> str:
    """Local name of an element tag ('{ns}flowRoot' and 'flowRoot' both -> 'flowRoot')."""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _children(el: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in el if _local(child.tag) == name]


def _length(value: str | None) -> float | None:
    """An SVG length attribute as a float (plain number, optional px suffix); None if absent."""
    if not value:
        return None
    try:
        return float(value.strip().removesuffix("px").strip())
    except ValueError:
        return None


def _svg_bounds(rect: tuple[float, float, float, float], t: _Matrix) -> tuple[float, float, float, float]:
    """SVG-space (x0, y0, x1, y1) of a rect's four corners under transform t."""
    x, y, w, h = rect
    xs: list[float] = []
    ys: list[float] = []
    for cx, cy in ((x, y), (x + w, y), (x, y + h), (x + w, y + h)):
        xs.append(t[0] * cx + t[2] * cy + t[4])
        ys.append(t[1] * cx + t[3] * cy + t[5])
    return min(xs), min(ys), max(xs), max(ys)


def parse_svg(
    svg: str, *, file_width: int, file_height: int, file_y0: int, scale: float
) -> tuple[list[TruthBox], int]:
    """Parse one text-layer SVG into TruthBoxes (page=0) in strip space plus the dropped-box count."""
    root = ET.fromstring(svg)
    svg_width = _length(root.get("width"))
    svg_height = _length(root.get("height"))
    sx = file_width / svg_width if svg_width else 1.0
    crop_top = max(0, round(svg_height * sx) - file_height) if svg_height else 0

    boxes: list[TruthBox] = []
    dropped = 0

    def walk(el: ET.Element, t: _Matrix) -> None:
        nonlocal dropped
        t = _mul(t, _parse_transform(el.get("transform")))
        if _local(el.tag) == "flowRoot":
            region = next(
                (
                    rect
                    for flow_region in _children(el, "flowRegion")
                    for rect in _children(flow_region, "rect")
                ),
                None,
            )
            lines = tuple(
                text
                for para in _children(el, "flowPara")
                if (text := "".join(para.itertext()).strip())
            )
            if region is None or not lines:
                return  # no box or no text: skipped, not dropped
            x, y = _length(region.get("x")) or 0.0, _length(region.get("y")) or 0.0
            w, h = _length(region.get("width")) or 0.0, _length(region.get("height")) or 0.0
            bx0, by0, bx1, by1 = _svg_bounds((x, y, w, h), t)
            px0 = min(max(bx0 * sx, 0.0), float(file_width))
            px1 = min(max(bx1 * sx, 0.0), float(file_width))
            py0 = min(max(by0 * sx - crop_top, 0.0), float(file_height))
            py1 = min(max(by1 * sx - crop_top, 0.0), float(file_height))
            if px1 <= px0 or py1 <= py0:
                dropped += 1
                return
            boxes.append(
                TruthBox(
                    page=0,
                    bbox=BBox(
                        x0=math.floor(px0 * scale + 1e-6),
                        y0=math.floor(py0 * scale + file_y0 + 1e-6),
                        x1=math.ceil(px1 * scale - 1e-6),
                        y1=math.ceil(py1 * scale + file_y0 - 1e-6),
                    ),
                    lines=lines,
                )
            )
            return
        for child in el:
            walk(child, t)

    walk(root, _IDENTITY)
    return boxes, dropped


def _page_number(name: str) -> int | None:
    """The integer page number of a raw file name ('01.jpg' -> 1); None when the stem is not numeric."""
    try:
        return int(Path(name).stem)
    except ValueError:
        return None


def _truth_paths(lang_dir: Path, page: int) -> Path | None:
    """The chapter's truth SVG of one page (`E*Pnn.svg`, first match), or None when absent."""
    matches = sorted(lang_dir.glob(f"E*P{page:02d}.svg"))
    return matches[0] if matches else None


def load_truth(check_dir: Path, lang: str, ingest: IngestArtifact) -> tuple[list[TruthBox], TruthStats]:
    """All truth boxes of a chapter in strip space, in file order then document order, plus coverage."""
    lang_dir = check_dir / lang
    boxes: list[TruthBox] = []
    pages = 0
    without_truth = 0
    dropped = 0
    for source in ingest.files:
        if source.filtered:
            continue
        page = _page_number(source.name)
        if page is None:
            continue
        pages += 1
        path = _truth_paths(lang_dir, page)
        if path is None:
            without_truth += 1
            continue
        page_boxes, page_dropped = parse_svg(
            path.read_text(encoding="utf-8"),
            file_width=source.width,
            file_height=source.height,
            file_y0=source.y0,
            scale=source.scale,
        )
        boxes.extend(replace(box, page=page) for box in page_boxes)
        dropped += page_dropped
    return boxes, TruthStats(pages=pages, pages_without_truth=without_truth, dropped=dropped)


def load_english_pages(check_dir: Path, ingest: IngestArtifact) -> dict[int, str]:
    """Per page number, the English truth text (all boxes' lines joined by spaces) of that page."""
    lang_dir = check_dir / "en"
    pages: dict[int, str] = {}
    for source in ingest.files:
        if source.filtered:
            continue
        page = _page_number(source.name)
        if page is None:
            continue
        path = _truth_paths(lang_dir, page)
        if path is None:
            continue
        page_boxes, _ = parse_svg(
            path.read_text(encoding="utf-8"),
            file_width=source.width,
            file_height=source.height,
            file_y0=source.y0,
            scale=source.scale,
        )
        pages[page] = " ".join(line for box in page_boxes for line in box.lines)
    return pages
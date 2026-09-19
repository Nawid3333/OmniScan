"""Ground truth for `omniscan eval`: Inkscape text-layer SVGs (Pepper&Carrot) as TruthBox lists.

The translated-check folder holds, per chapter, `truth/<lang>/EnnPpp.svg` text layers with one
`flowRoot` per text box (box = `flowRegion/rect`, lines = `flowPara` texts). Newer episodes use
plain `<text>` elements with line `tspan`s and no box geometry instead; `parse_svg` turns each of
them into one approximate box estimated from the line positions, font sizes and text anchors.
`parse_svg` maps those boxes into the strip space of the ingest artifact: SVG pixels are scaled by
`file_width / svg_width`, cropped at the top (the lettered low-res pages lose the SVG's top margin),
then clamped to the page and finally mapped by the source file's strip scale/offset.
"""

from __future__ import annotations

import math
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path

from omniscan.core.schemas import BBox, IngestArtifact

# (a, b, c, d, e, f): x' = a*x + c*y + e, y' = b*x + d*y + f (SVG matrix order).
_Matrix = tuple[float, float, float, float, float, float]

_IDENTITY: _Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

_TRANSFORM_RE = re.compile(r"([a-zA-Z]+)\s*\(([^)]*)\)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

_SODIPODI_ROLE = "{http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd}role"


@dataclass(frozen=True, slots=True)
class TruthBox:
    """One ground-truth text box of one page, in strip space."""

    page: int  # page number from the SVG file name (E06P01 -> 1)
    bbox: BBox
    lines: tuple[str, ...]  # the flowPara/tspan texts, stripped, non-empty, document order
    approx: bool = False  # <text> element: box estimated from font metrics, not read off a rect

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


def _style_prop(style: str | None, prop: str) -> str | None:
    """One CSS property of an SVG `style` attribute ('font-size:20px;...' -> '20px')."""
    if not style:
        return None
    for decl in style.split(";"):
        key, sep, value = decl.partition(":")
        if sep and key.strip().lower() == prop and value.strip():
            return value.strip()
    return None


def _first_number(value: str | None) -> float | None:
    """The first number of an SVG list attribute ('10 20 30' -> 10.0); None when absent."""
    if not value:
        return None
    match = _NUMBER_RE.search(value)
    return float(match.group()) if match else None


def _line_width(text: str, font_size: float) -> float:
    """Estimated rendered width of one line: wide (CJK) glyphs one em, everything else half."""
    return font_size * sum(1.0 if unicodedata.east_asian_width(ch) in ("W", "F") else 0.5 for ch in text)


def _line_box(
    x: float, y: float, font_size: float, anchor: str, text: str
) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of one line in the text element's coordinates, from anchor and font size."""
    width = _line_width(text, font_size)
    left = x - {"middle": width / 2.0, "end": width}.get(anchor, 0.0)
    return left, y - font_size, left + width, y + 0.25 * font_size


def _text_element_lines(el: ET.Element) -> list[tuple[str, float, float, float, str]]:
    """(text, x, y, font-size, text-anchor) of one <text> element's lines; unpositioned lines drop.

    Lines are the child tspans with `sodipodi:role="line"`, or the element's own text when there
    is none. Each line falls back from its tspan's x/y/style to the text element's.
    """
    el_style = el.get("style")
    el_x, el_y = _first_number(el.get("x")), _first_number(el.get("y"))
    tspans = [tspan for tspan in _children(el, "tspan") if tspan.get(_SODIPODI_ROLE) == "line"]
    if not tspans:
        tspans = [el]  # a text without role-line tspans is one line: its own text
    lines: list[tuple[str, float, float, float, str]] = []
    for node in tspans:
        text = "".join(node.itertext()).strip()
        if not text:
            continue
        x = _first_number(node.get("x"))
        y = _first_number(node.get("y"))
        if x is None:
            x = el_x
        if y is None:
            y = el_y
        if x is None or y is None:
            continue
        style = node.get("style")
        font_size = _length(_style_prop(style, "font-size"))
        if font_size is None:
            font_size = _length(_style_prop(el_style, "font-size")) or 16.0
        anchor = _style_prop(style, "text-anchor") or _style_prop(el_style, "text-anchor") or "start"
        if anchor not in ("middle", "end"):
            anchor = "start"
        lines.append((text, x, y, font_size, anchor))
    return lines


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

    def emit(
        bx0: float, by0: float, bx1: float, by1: float, t: _Matrix, lines: tuple[str, ...], approx: bool
    ) -> None:
        """Map one SVG-space box through page and strip space; drop it when it leaves the page."""
        nonlocal dropped
        sx0, sy0, sx1, sy1 = _svg_bounds((bx0, by0, bx1 - bx0, by1 - by0), t)
        px0 = min(max(sx0 * sx, 0.0), float(file_width))
        px1 = min(max(sx1 * sx, 0.0), float(file_width))
        py0 = min(max(sy0 * sx - crop_top, 0.0), float(file_height))
        py1 = min(max(sy1 * sx - crop_top, 0.0), float(file_height))
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
                approx=approx,
            )
        )

    def walk(el: ET.Element, t: _Matrix) -> None:
        t = _mul(t, _parse_transform(el.get("transform")))
        kind = _local(el.tag)
        if kind == "flowRoot":
            region = next(
                (
                    rect
                    for flow_region in _children(el, "flowRegion")
                    for rect in _children(flow_region, "rect")
                ),
                None,
            )
            lines = tuple(
                text for para in _children(el, "flowPara") if (text := "".join(para.itertext()).strip())
            )
            if region is None or not lines:
                return  # no box or no text: skipped, not dropped
            x, y = _length(region.get("x")) or 0.0, _length(region.get("y")) or 0.0
            w, h = _length(region.get("width")) or 0.0, _length(region.get("height")) or 0.0
            emit(x, y, x + w, y + h, t, lines, approx=False)
            return  # flowRoot content is its own box; nested text elements belong to it
        if kind == "text":
            line_data = _text_element_lines(el)
            if line_data:
                line_boxes = [
                    _line_box(x, y, font_size, anchor, text) for text, x, y, font_size, anchor in line_data
                ]
                emit(
                    min(b[0] for b in line_boxes),
                    min(b[1] for b in line_boxes),
                    max(b[2] for b in line_boxes),
                    max(b[3] for b in line_boxes),
                    t,
                    tuple(line[0] for line in line_data),
                    approx=True,
                )
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

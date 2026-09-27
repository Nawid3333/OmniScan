"""Hand-set lettering (edits.json `layout`) applied to the typesetter's layout items (pure CPU, no I/O).

Colour, outline, alignment and angle only restyle an item. A new font, size, box or line breaks set the
line again, the way the typesetter does: explicit lines are kept as written; otherwise the text is fitted
into the box (or the region's own lettering shape) at the given size, or at the largest size that fits.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from omniscan.core.config import TypesetConfig
from omniscan.core.schemas import BBox, ChapterEdits, LayoutEdit, LayoutItem, Region
from omniscan.edits.apply import match_layout_edits
from omniscan.typeset.fit import FontFactory, Shape, fit_shape, line_height, measurer
from omniscan.typeset.fonts import font_file, layout_font_name, load_font
from omniscan.typeset.plan import lettering_shape


def _centred(shape: BBox, width: int, height: int) -> BBox:
    """A width x height box centred in `shape`."""
    cx, cy = (shape.x0 + shape.x1) / 2, (shape.y0 + shape.y1) / 2
    x0, y0 = round(cx - width / 2), round(cy - height / 2)
    return BBox(x0=x0, y0=y0, x1=x0 + width, y1=y0 + height)


def overridden(
    item: LayoutItem,
    edit: LayoutEdit,
    region: Region,
    text: str,
    cfg: TypesetConfig,
    *,
    font_factory: FontFactory = load_font,
) -> LayoutItem:
    """`item` with the hand-set lettering of `edit`; `text` is the region's English line."""
    update: dict[str, object] = {
        key: value
        for key, value in (
            ("color", edit.color),
            ("stroke_px", edit.stroke_px),
            ("stroke_color", edit.stroke_color),
            ("align", edit.align),
            ("angle", edit.angle),
        )
        if value is not None
    }
    if edit.font is None and edit.size_px is None and edit.box is None and edit.lines is None:
        return item.model_copy(update=update)
    path = font_file(edit.font or item.font)
    shape = Shape("rect", edit.box) if edit.box is not None else lettering_shape(region, cfg)
    if edit.lines:
        size = edit.size_px or item.size_px
        font = measurer(font_factory)(path, size)
        lines = list(edit.lines)
        width = math.ceil(max(font.getlength(line) for line in lines))
        height = len(lines) * line_height(size, cfg.line_spacing)
        overflow = width > shape.box.width or height > shape.box.height
    else:
        words = " ".join(text.split())
        if item.lines and all(line == line.upper() for line in item.lines):
            words = words.upper()  # the typesetter lettered this region in capitals
        largest = cfg.sfx_max_px if item.font_role == "sfx" else cfg.max_px
        fit = fit_shape(
            words,
            shape,
            path,
            min_px=edit.size_px or cfg.min_px,
            max_px=edit.size_px or max(largest, item.size_px),
            line_spacing=cfg.line_spacing,
            hyphenate=cfg.hyphenate,
            font_factory=font_factory,
        )
        size, lines, width, height, overflow = fit.size_px, fit.lines, fit.width, fit.height, fit.overflow
    update.update(
        font=layout_font_name(path),
        size_px=size,
        lines=lines,
        box=_centred(shape.box, width, height),
        overflow=overflow,
    )
    return item.model_copy(update=update)


def apply_layout_edits(
    items: Sequence[LayoutItem],
    edits: ChapterEdits,
    regions: Sequence[Region],
    texts: Mapping[str, str],
    cfg: TypesetConfig,
    *,
    font_factory: FontFactory = load_font,
) -> tuple[list[LayoutItem], int]:
    """The layout items with every hand-set lettering applied (a hidden one removed), and the number of
    edits whose region no longer exists. A region without a layout item (no English line) stays unlettered."""
    by_id = {region.id: region for region in regions}
    claims = match_layout_edits(regions, edits)
    edit_of = {region_id: edits.layout[i] for i, region_id in claims.items()}
    result: list[LayoutItem] = []
    for item in items:
        edit = edit_of.get(item.region_id)
        if edit is None:
            result.append(item)
        elif not edit.hidden:
            region = by_id[item.region_id]
            result.append(
                overridden(item, edit, region, texts.get(item.region_id, ""), cfg, font_factory=font_factory)
            )
    return result, len(edits.layout) - len(claims)

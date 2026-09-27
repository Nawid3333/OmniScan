"""BallonsTranslator projects: read one's text blocks, write a chapter as one (pure functions, no I/O).

A BallonsTranslator project is a folder of page images with `imgtrans_<folder name>.json` in it: per page image, a
list of text blocks, each with a box (`xyxy`, page pixels), its text lines (`lines`, quadrilaterals), the source
text (`text`, a list of lines), the translation and the lettering (`fontformat`). The cleaned pages sit in
`inpainted/<page stem>.png`. Reading takes the boxes and texts; writing gives each region one block with one line
box, its source text, its English line and its lettering's size and colours, so BallonsTranslator opens the
chapter ready to re-letter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path

from omniscan.core.schemas import IngestArtifact, LayoutItem, Region, SourceFile
from omniscan.interchange.blocks import Block, join_lines
from omniscan.interchange.labelplus import page_at, strip_pages

PREFIX = "imgtrans_"
INPAINTED_DIR = "inpainted"
IMAGE_SUFFIXES = (".bmp", ".jpg", ".jpeg", ".png", ".webp", ".jxl")  # the page images BallonsTranslator lists
CENTER = 1  # BallonsTranslator's TextAlignment.Center
DEFAULT_FONT_PX = 24  # BallonsTranslator's own default font size


def project_file(folder: Path) -> Path:
    """Where BallonsTranslator keeps the project of the page folder `folder`."""
    return folder / f"{PREFIX}{folder.name}.json"


def page_name(page: SourceFile) -> str:
    """The page's image name in the project: its own, or its JPEG's when BallonsTranslator cannot open the raw."""
    name = Path(page.name)
    return page.name if name.suffix.lower() in IMAGE_SUFFIXES else f"{name.stem}.jpg"


def _text(value: object) -> str:
    """A block's `text` (a list of lines, or one string in old projects) as one text."""
    if isinstance(value, str):
        return join_lines(value.splitlines())
    if isinstance(value, list):
        return join_lines([line for line in value if isinstance(line, str)])
    return ""


def _box(record: object) -> tuple[float, float, float, float] | None:
    """A block record's `xyxy` box, or None when it has none."""
    xyxy = record.get("xyxy") if isinstance(record, dict) else None
    if not isinstance(xyxy, list) or len(xyxy) != 4:
        return None
    if not all(isinstance(v, Real) and not isinstance(v, bool) for v in xyxy):
        return None
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    return x0, y0, x1, y1


def read_project(data: object) -> list[Block]:
    """The text blocks of a BallonsTranslator project (its parsed JSON), pages in file order; ValueError when
    it is not one."""
    pages = data.get("pages") if isinstance(data, dict) else None
    if not isinstance(pages, dict):
        raise ValueError('not a BallonsTranslator project: no "pages" object')
    blocks: list[Block] = []
    for page, records in pages.items():
        if not isinstance(records, list):
            raise ValueError(f"page {page!r}: its text blocks are not a list")
        for number, record in enumerate(records, start=1):
            box = _box(record)
            if box is None:
                raise ValueError(f"page {page!r} block {number}: no xyxy box")
            translation = record.get("translation")
            blocks.append(
                Block(
                    page=str(page),
                    x0=box[0],
                    y0=box[1],
                    x1=box[2],
                    y1=box[3],
                    text=_text(record.get("text")),
                    translation=translation.strip() if isinstance(translation, str) else "",
                )
            )
    return blocks


def _record(
    region: Region, page: SourceFile, english: str, item: LayoutItem | None
) -> dict[str, object] | None:
    """The block of `region` on `page` (page pixels), or None when its box is not on that page."""
    scale = page.scale
    x0 = max(0, min(round(region.bbox.x0 / scale), page.width))
    x1 = max(0, min(round(region.bbox.x1 / scale), page.width))
    y0 = max(0, min(round((region.bbox.y0 - page.y0) / scale), page.height))
    y1 = max(0, min(round((region.bbox.y1 - page.y0) / scale), page.height))
    if x1 <= x0 or y1 <= y0:
        return None
    colour = item.color if item is not None else region.text_color or (0, 0, 0)
    size = item.size_px / scale if item is not None else float(DEFAULT_FONT_PX)
    stroke = item.stroke_px / item.size_px if item is not None and item.size_px > 0 else 0.0
    fontformat: dict[str, object] = {
        "font_size": round(size, 1),
        "frgb": list(colour),
        "srgb": list(item.stroke_color if item is not None else (255, 255, 255)),
        "stroke_width": round(stroke, 3),
        "alignment": CENTER,
        "vertical": False,
    }
    return {
        "xyxy": [x0, y0, x1, y1],
        "lines": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]],
        "language": region.lang,
        "text": [region.text] if region.text else [],
        "translation": english,
        "src_is_vertical": region.orientation == "v",
        "fontformat": fontformat,
    }


def page_blocks(
    ingest: IngestArtifact, regions: Sequence[Region], english: Mapping[str, str], items: Sequence[LayoutItem]
) -> dict[str, list[dict[str, object]]]:
    """Every page's text blocks (by project image name): one per region but watermarks, on the page that holds
    the centre of its text box."""
    pages = strip_pages(ingest)
    lettering = {item.region_id: item for item in items}
    blocks: dict[str, list[dict[str, object]]] = {page_name(page): [] for page in pages}
    for region in regions:
        if region.kind == "watermark":
            continue
        page = page_at(pages, (region.bbox.y0 + region.bbox.y1) / 2)
        if page is None:
            continue
        record = _record(region, page, english.get(region.id, ""), lettering.get(region.id))
        if record is not None:
            blocks[page_name(page)].append(record)
    return blocks


def write_project(
    folder: Path, ingest: IngestArtifact, blocks: Mapping[str, list[dict[str, object]]]
) -> dict[str, object]:
    """The project JSON of the chapter's pages (copied into `folder`) with `blocks` (see `page_blocks`)."""
    pages = strip_pages(ingest)
    return {
        "directory": str(folder),
        "pages": dict(blocks),
        "current_img": page_name(pages[0]) if pages else None,
        "image_info": {page_name(page): {"width": page.width, "height": page.height} for page in pages},
    }

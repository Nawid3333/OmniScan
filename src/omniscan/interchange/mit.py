"""manga-image-translator text files: the `<image>_translations.txt` its `--save-text` writes (pure, no I/O).

For every translated image the file has an `[<image path>]` line, then per text region a `-- <n> --` line, a
`color:` line, `text:` (the source), `trans:` (the translation) and one `coords: [x1, y1, …, x4, y4]` line per
text line (a quadrilateral in the image's pixels; NumPy 2 writes each number as `np.int64(…)`). Several images
can share one file (`--save-text-file`).
"""

from __future__ import annotations

import re

from omniscan.interchange.blocks import Block

_HEADER = re.compile(r"^\[(.+)\]$")
_NUMBER = re.compile(r"^--\s*(\d+)\s*--$")
_NUMPY_SCALAR = re.compile(r"np\.\w+\(([^()]*)\)")
_FLOAT = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _page_name(path: str) -> str:
    """The image file name of a header's path (Windows or POSIX)."""
    return re.split(r"[\\/]", path.strip())[-1]


def _coords(value: str, line_no: int) -> list[float]:
    """The numbers of a `coords:` line; ValueError naming the line when they are not x, y pairs."""
    numbers = [float(n) for n in _FLOAT.findall(_NUMPY_SCALAR.sub(r"\1", value))]
    if not numbers or len(numbers) % 2:
        raise ValueError(f"line {line_no}: coords {value.strip()!r} are not x, y pairs")
    return numbers


class _Region:
    """A region being read: its number, texts and the corners of its text lines."""

    def __init__(self, page: str, number: int, line_no: int) -> None:
        self.page, self.number, self.line_no = page, number, line_no
        self.text: list[str] = []
        self.trans: list[str] = []
        self.points: list[float] = []

    def block(self) -> Block:
        """The region as a block; ValueError when it has no coords."""
        if not self.points:
            raise ValueError(f"line {self.line_no}: region {self.number} of {self.page!r} has no coords")
        xs, ys = self.points[0::2], self.points[1::2]
        return Block(
            page=self.page,
            x0=min(xs),
            y0=min(ys),
            x1=max(xs),
            y1=max(ys),
            text="\n".join(self.text).strip(),
            translation="\n".join(self.trans).strip(),
        )


def parse(text: str) -> list[Block]:
    """The text regions of a manga-image-translator text file, in file order; ValueError when it is not one."""
    blocks: list[Block] = []
    page: str | None = None
    region: _Region | None = None
    target: list[str] | None = None  # the field a continuation line belongs to
    blank = True  # the previous line was blank (a new image or region only starts after one)
    for line_no, line in enumerate(text.lstrip("﻿").splitlines(), start=1):
        stripped = line.strip()
        header, number = _HEADER.match(stripped), _NUMBER.match(stripped)
        if blank and header:
            if region is not None:
                blocks.append(region.block())
            page, region, target = _page_name(header.group(1)), None, None
        elif blank and number and page is not None:
            if region is not None:
                blocks.append(region.block())
            region, target = _Region(page, int(number.group(1)), line_no), None
        elif region is not None and stripped:
            name, _sep, value = line.partition(":")
            if name == "text":
                region.text = [value.strip()]
                target = region.text
            elif name == "trans":
                region.trans = [value.strip()]
                target = region.trans
            elif name == "coords":
                region.points += _coords(value, line_no)
                target = None
            elif name == "color":
                target = None
            elif target is not None:
                target.append(stripped)
        blank = not stripped
    if page is None:
        raise ValueError("not a manga-image-translator text file: no [image path] line")
    if region is not None:
        blocks.append(region.block())
    return blocks

"""Layered Photoshop files (.psd) of a chapter's pages, for groups who finish a release in Photoshop.

Each page becomes one PSD at strip resolution with three layers, bottom to top: `raw` (the page as scanned),
`clean` (after the automatic cleaning and the hand cleanup) and `text` (the lettering on a transparent layer),
plus the finished page as the file's composite image, so a viewer without layer support shows the release.
Hide `text` to re-letter by hand, or paint on `clean`. Channels are PackBits (RLE) compressed, as Photoshop
writes them. Pages are built with the same CPU code as the studio preview (typeset/page_preview.py): no GPU.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from omniscan.cleanup.store import current_crop, strip_crop
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import IngestArtifact, LayoutItem
from omniscan.typeset.page_preview import blend, page_box
from omniscan.typeset.render import render_item

_MAX_PACKET = 128


@dataclass(frozen=True, slots=True)
class PsdLayer:
    """One full-size layer: RGB pixels (uint8 [h, w, 3]) and, for a transparent layer, alpha (uint8 [h, w])."""

    name: str
    rgb: np.ndarray
    alpha: np.ndarray | None = None


def _literal(row: np.ndarray) -> bytes:
    """`row` as PackBits literal packets only (fast for busy rows)."""
    data = row.tobytes()
    return b"".join(
        bytes([len(chunk) - 1]) + chunk
        for chunk in (data[i : i + _MAX_PACKET] for i in range(0, len(data), _MAX_PACKET))
    )


def packbits(row: np.ndarray) -> bytes:
    """One row (uint8 [n]) PackBits-encoded: runs of three or more equal bytes as repeat packets, the rest as
    literals; a row with many short runs is written as literals only (cheap to encode, barely larger)."""
    n = len(row)
    if n == 0:
        return b""
    change = np.flatnonzero(row[1:] != row[:-1]) + 1
    starts = [0, *change.tolist()]
    if len(starts) > max(1, n // 16):
        return _literal(row)
    ends = [*starts[1:], n]
    out = bytearray()
    pending = 0  # start of the literal bytes not written yet
    for start, end in zip(starts, ends, strict=True):
        if end - start < 3:
            continue
        if pending < start:
            out += _literal(row[pending:start])
        value = int(row[start])
        for i in range(start, end, _MAX_PACKET):
            count = min(_MAX_PACKET, end - i)
            if count == 1:
                out += bytes([0, value])
            else:
                out += bytes([257 - count, value])
        pending = end
    if pending < n:
        out += _literal(row[pending:n])
    return bytes(out)


def _rle_plane(plane: np.ndarray) -> tuple[bytes, bytes]:
    """(row byte counts, data) of one channel plane (uint8 [h, w]) in PackBits."""
    rows = [packbits(row) for row in plane]
    return struct.pack(f">{len(rows)}H", *(len(r) for r in rows)), b"".join(rows)


def _pascal(name: str) -> bytes:
    """A layer name as a Pascal string padded to a multiple of 4 bytes (length byte included)."""
    raw = name.encode("ascii", "replace")[:255]
    data = bytes([len(raw)]) + raw
    return data + b"\0" * (-len(data) % 4)


def write_psd(layers: Sequence[PsdLayer], composite: np.ndarray) -> bytes:
    """A PSD (8-bit RGB) with `layers` (bottom first, each the size of `composite`) and the composite image."""
    height, width = composite.shape[:2]
    records = bytearray()
    channel_data = bytearray()
    for layer in layers:
        if layer.rgb.shape != (height, width, 3):
            raise ValueError(f"layer {layer.name!r} is {layer.rgb.shape}, the image is {(height, width, 3)}")
        planes = [(i, np.ascontiguousarray(layer.rgb[..., i])) for i in range(3)]
        if layer.alpha is not None:
            planes.insert(0, (-1, np.ascontiguousarray(layer.alpha)))
        encoded = [(channel, *_rle_plane(plane)) for channel, plane in planes]
        records += struct.pack(">4i", 0, 0, height, width) + struct.pack(">H", len(encoded))
        for channel, counts, data in encoded:
            records += struct.pack(">hI", channel, 2 + len(counts) + len(data))
            channel_data += struct.pack(">H", 1) + counts + data
        extra = struct.pack(">I", 0) + struct.pack(">I", 0) + _pascal(layer.name)
        records += b"8BIMnorm" + bytes([255, 0, 0, 0]) + struct.pack(">I", len(extra)) + extra
    layer_info = struct.pack(">h", len(layers)) + records + channel_data
    layer_info += b"\0" * (len(layer_info) % 2)
    section = struct.pack(">I", len(layer_info)) + layer_info + struct.pack(">I", 0)  # + no global mask
    planes = [_rle_plane(np.ascontiguousarray(composite[..., i])) for i in range(3)]
    header = b"8BPS" + struct.pack(">H6xHIIHH", 1, 3, height, width, 8, 3)
    return b"".join(
        [
            header,
            struct.pack(">I", 0),  # color mode data
            struct.pack(">I", 0),  # image resources
            struct.pack(">I", len(section)),
            section,
            struct.pack(">H", 1),  # composite: RLE
            *(counts for counts, _ in planes),
            *(data for _, data in planes),
        ]
    )


def _over(layer_rgb: np.ndarray, layer_alpha: np.ndarray, x: int, y: int, rgba: np.ndarray) -> None:
    """Composite straight-alpha `rgba` over a transparent layer (rgb + alpha) with its top-left at (x, y)."""
    h, w = layer_alpha.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + rgba.shape[1], w), min(y + rgba.shape[0], h)
    if x0 >= x1 or y0 >= y1:
        return
    src = rgba[y0 - y : y1 - y, x0 - x : x1 - x].astype(np.float32)
    sa = src[..., 3] / 255.0
    da = layer_alpha[y0:y1, x0:x1].astype(np.float32) / 255.0
    out_a = sa + da * (1.0 - sa)
    dst = layer_rgb[y0:y1, x0:x1].astype(np.float32)
    weight = np.divide(sa, out_a, out=np.zeros_like(sa), where=out_a > 0)[..., None]
    layer_rgb[y0:y1, x0:x1] = np.clip(np.round(src[..., :3] * weight + dst * (1.0 - weight)), 0, 255).astype(
        np.uint8
    )
    layer_alpha[y0:y1, x0:x1] = np.clip(np.round(out_a * 255.0), 0, 255).astype(np.uint8)


def page_psd(paths: ChapterPaths, ingest: IngestArtifact, page: int, items: Sequence[LayoutItem]) -> bytes:
    """The PSD of page `page` (its SourceFile index): raw, clean and text layers, the finished page as composite."""
    box = page_box(ingest, page)
    raw = strip_crop(paths, ingest, box)
    clean = current_crop(paths, ingest, box)
    height, width = clean.shape[:2]
    text_rgb = np.zeros((height, width, 3), np.uint8)
    text_alpha = np.zeros((height, width), np.uint8)
    finished = clean.copy()
    for item in items:
        if item.box.y1 + item.stroke_px + 2 < box.y0 or item.box.y0 - item.stroke_px - 2 > box.y1:
            continue  # nowhere near this page
        glyph = render_item(item)
        if glyph is None:
            continue
        _over(text_rgb, text_alpha, glyph.x, glyph.y - box.y0, glyph.rgba)
        blend(finished, glyph.x, glyph.y - box.y0, glyph.rgba)
    layers = [PsdLayer("raw", raw), PsdLayer("clean", clean), PsdLayer("text", text_rgb, text_alpha)]
    return write_psd(layers, finished)

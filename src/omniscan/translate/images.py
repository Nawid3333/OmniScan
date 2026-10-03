"""Page images for the translation model: the pages a request's regions sit on (profiles with `images = true`).

A vision model that sees the page can tell who is shouting, what "that" points at and whether a caption is a
sign or narration. `PageImages` reads a chapter's slices from its raw pages as base64 JPEG for the Ollama
message's `images`: each slice is scaled so the strip's width fits `image_side` px and cut into tiles at most
`image_side` px tall, so a tall webtoon slice keeps its detail instead of shrinking to a sliver (a vision
encoder squares what it gets). Every region of a request then carries the tile it is on (the one holding its
vertical centre) and its box in that tile's pixels (translate/prompts.py). The pixels are read with Pillow like
the editing tools' crops (cleanup/strip.py): a few JPEGs per request next to a model call that takes seconds.
A chapter whose pages cannot be read is translated without images (logged once).

A region's translation key (translate/incremental.py) holds the identity of its tile — the rows and the
sha256 of the raw pages under them, from ingest.json — so a replaced page is translated again; a region sent
without its image is keyed as if images were off, so it is translated again once the pages can be read.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from omniscan.cleanup.strip import strip_crop
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import BBox, IngestArtifact, Region, Slice, SlicesArtifact
from omniscan.translate.profiles import TranslationProfile, unkeyed_fields

log = logging.getLogger(__name__)

IMAGE_FIELDS = frozenset(
    {"images", "image_side", "images_per_request"}
)  # profile fields a key or the stage hash sees only when images are on
JPEG_QUALITY = 85


@dataclass(frozen=True, slots=True)
class Tile:
    """A piece of a slice sent as one image: strip rows [y0, y1) of slice `slice_index`, scaled by `scale`."""

    slice_index: int
    index: int  # the tile's number within its slice
    y0: int
    y1: int
    scale: float  # image px per strip px


@dataclass(frozen=True, slots=True)
class PageImage:
    """One tile as sent to the model: base64 JPEG."""

    tile: Tile
    data: str

    def box(self, bbox: BBox) -> list[int]:
        """A strip-space box in this image's pixels (clipped to the tile's rows)."""
        tile = self.tile

        def row(y: int) -> int:
            return round((min(max(y, tile.y0), tile.y1) - tile.y0) * tile.scale)

        return [round(bbox.x0 * tile.scale), row(bbox.y0), round(bbox.x1 * tile.scale), row(bbox.y1)]


def stage_profile(profile: TranslationProfile) -> dict[str, object]:
    """A profile as the translate stage hashes it: with the page-image settings only when images are on, so a
    profile without them hashes as before images existed."""
    return profile.model_dump(
        exclude=unkeyed_fields(profile) | (set() if profile.images else set(IMAGE_FIELDS))
    )


def image_key(profile: TranslationProfile, page: str | None) -> dict[str, object]:
    """What a translation key adds for a region's page image (`page`: its tile's identity, None when it was not
    sent): nothing when images are off or the image was not sent, so the key is the one without images."""
    if not profile.images or page is None:
        return {}
    return {"images": {"side": profile.image_side, "page": page}}


def tiles_of(piece: Slice, width: int, side: int) -> list[Tile]:
    """How a slice of a `width`-px strip is sent: scaled so the width fits `side` px, cut into equal tiles at
    most `side` px tall."""
    scale = min(1.0, side / width)
    height = piece.y1 - piece.y0
    count = max(1, math.ceil(height * scale / side))
    step = math.ceil(height / count)
    return [
        Tile(piece.index, i, piece.y0 + i * step, min(piece.y1, piece.y0 + (i + 1) * step), scale)
        for i in range(count)
    ]


class PageImages:
    """A chapter's tiles as JPEG images for the translation model, read on first use and kept per size.

    `sent` collects the ids of the regions whose image was handed out since `begin()` (one translation run)."""

    def __init__(self, paths: ChapterPaths) -> None:
        self._paths = paths
        self._layout: tuple[IngestArtifact, dict[int, Slice]] | None = None
        self._cache: dict[tuple[int, int, int], PageImage] = {}
        self._broken = False
        self.sent: set[str] = set()

    def begin(self) -> None:
        """Start a translation run: forget which regions' images were handed out."""
        self.sent = set()

    def _slices(self) -> tuple[IngestArtifact, dict[int, Slice]] | None:
        """The chapter's ingest.json and slices by index; None (logged once) when they cannot be read."""
        if self._layout is None and not self._broken:
            try:
                ingest = IngestArtifact.load(self._paths.artifact("ingest.json"))
                slices = SlicesArtifact.load(self._paths.artifact("slices.json")).slices
            except (OSError, ValueError) as exc:
                self._give_up(f"no page layout ({exc})")
                return None
            self._layout = (ingest, {s.index: s for s in slices})
        return self._layout

    def _give_up(self, why: str) -> None:
        """Stop sending images for this chapter, saying why once."""
        if not self._broken:
            log.warning(
                "%s/%s: translating without page images: %s", self._paths.series, self._paths.chapter, why
            )
        self._broken = True

    def _place(self, region: Region, side: int) -> tuple[IngestArtifact, Tile] | None:
        """The chapter's ingest.json and the tile `region` is sent on (the one holding its vertical centre);
        None without a page layout."""
        layout = None if self._broken else self._slices()
        piece = layout[1].get(region.slice_index) if layout is not None else None
        if layout is None or piece is None:
            return None
        centre = (region.bbox.y0 + region.bbox.y1) // 2
        tiles = tiles_of(piece, layout[0].strip_width, side)
        return layout[0], next((t for t in tiles if centre < t.y1), tiles[-1])

    def tile(self, region: Region, side: int) -> Tile | None:
        """The tile `region` is sent on; None without a page layout."""
        placed = self._place(region, side)
        return placed[1] if placed is not None else None

    def page_id(self, region: Region, side: int) -> str | None:
        """The identity of the image `region` is sent with: its tile's rows and the sha256 of the raw pages under
        them (ingest.json); None without a page layout."""
        placed = self._place(region, side)
        if placed is None:
            return None
        ingest, tile = placed
        pages = [f.sha256 for f in ingest.files if f.y0 < tile.y1 and f.y1 > tile.y0 and not f.filtered]
        text = json.dumps([tile.y0, tile.y1, round(tile.scale, 6), pages])
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def for_regions(self, regions: Sequence[Region], side: int) -> dict[str, PageImage]:
        """Region id -> the image it is on, for these regions ({} when the chapter's pages cannot be read)."""
        found: dict[str, PageImage] = {}
        for region in regions:
            placed = self._place(region, side)
            if placed is None:
                continue
            ingest, tile = placed
            key = (tile.slice_index, tile.index, side)
            if key not in self._cache:
                try:
                    pixels = strip_crop(
                        self._paths, ingest, BBox(x0=0, y0=tile.y0, x1=ingest.strip_width, y1=tile.y1)
                    )
                except (OSError, ValueError) as exc:
                    self._give_up(str(exc))
                    return {}
                self._cache[key] = PageImage(tile, _encode(pixels, tile.scale))
            found[region.id] = self._cache[key]
        self.sent |= found.keys()
        return found


def attached(images: Mapping[str, PageImage]) -> list[PageImage]:
    """The distinct images of a request, in the order their regions first use them."""
    return list({(image.tile.slice_index, image.tile.index): image for image in images.values()}.values())


def _encode(pixels: np.ndarray, scale: float) -> str:
    """A tile's pixels (uint8 [h, w, 3]) scaled by `scale`, as base64 JPEG."""
    image = Image.fromarray(pixels)
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("ascii")

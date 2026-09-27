"""Page images for the translation model: the slices a request's regions sit on (profiles with `images = true`).

A vision model that sees the page can tell who is shouting, what "that" points at and whether a caption is a
sign or narration. `PageImages` reads a chapter's slices from its raw pages and scales each so its long side is
at most the profile's `image_side`, as base64 JPEG for the Ollama message's `images`; every region of the
request then carries the image it is on and its box in that image's pixels (translate/prompts.py). The pixels
are read with Pillow like the editing tools' crops (cleanup/strip.py): a few JPEGs per request next to a model
call that takes seconds. A chapter whose pages cannot be read is translated without images (logged once).
"""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from omniscan.cleanup.strip import strip_crop
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import BBox, IngestArtifact, Region, Slice, SlicesArtifact
from omniscan.translate.profiles import TranslationProfile

log = logging.getLogger(__name__)

IMAGE_FIELDS = frozenset(
    {"images", "image_side", "images_per_request"}
)  # profile fields a key sees only when on
JPEG_QUALITY = 85


@dataclass(frozen=True, slots=True)
class PageImage:
    """One slice as sent to the model: base64 JPEG, its scale (image px per strip px) and its top strip row."""

    slice_index: int
    data: str
    scale: float
    y0: int

    def box(self, bbox: BBox) -> list[int]:
        """A strip-space box in this image's pixels."""
        return [
            round(bbox.x0 * self.scale),
            round((bbox.y0 - self.y0) * self.scale),
            round(bbox.x1 * self.scale),
            round((bbox.y1 - self.y0) * self.scale),
        ]


def image_key(profile: TranslationProfile) -> dict[str, object]:
    """What a translation key adds for a profile's page images: nothing when they are off (keys stay as before)."""
    if not profile.images:
        return {}
    return {"images": {"side": profile.image_side, "per_request": profile.images_per_request}}


def slices_of(regions: Sequence[Region]) -> list[int]:
    """The slices `regions` sit on, in the order they first appear."""
    return list(dict.fromkeys(region.slice_index for region in regions))


class PageImages:
    """A chapter's slices as JPEG images for the translation model, read on first use and kept per size."""

    def __init__(self, paths: ChapterPaths) -> None:
        self._paths = paths
        self._layout: tuple[IngestArtifact, dict[int, Slice]] | None = None
        self._cache: dict[tuple[int, int], PageImage] = {}
        self._broken = False

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

    def get(self, slice_indices: Sequence[int], side: int) -> list[PageImage]:
        """The images of these slices in the order given, each at most `side` px on its long side; [] when the
        chapter's pages cannot be read."""
        layout = None if self._broken else self._slices()
        if layout is None:
            return []
        ingest, slices = layout
        images: list[PageImage] = []
        for index in slice_indices:
            key = (index, side)
            if key not in self._cache:
                piece = slices.get(index)
                if piece is None:
                    continue
                try:
                    pixels = strip_crop(
                        self._paths, ingest, BBox(x0=0, y0=piece.y0, x1=ingest.strip_width, y1=piece.y1)
                    )
                except (OSError, ValueError) as exc:
                    self._give_up(str(exc))
                    return []
                self._cache[key] = _encode(pixels, index, piece.y0, side)
            images.append(self._cache[key])
        return images


def _encode(pixels: np.ndarray, index: int, y0: int, side: int) -> PageImage:
    """A slice's pixels (uint8 [h, w, 3]) scaled to at most `side` px on its long side, as base64 JPEG."""
    image = Image.fromarray(pixels)
    scale = min(1.0, side / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=JPEG_QUALITY)
    return PageImage(
        slice_index=index, data=base64.b64encode(buffer.getvalue()).decode("ascii"), scale=scale, y0=y0
    )

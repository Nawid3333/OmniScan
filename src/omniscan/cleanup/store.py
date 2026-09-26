"""cleanup.json + cleanup.npz: the chapter's hand cleanup, written only by the editing tools.

A brush stroke arrives in one page's own pixels (the Studio paints on the raw page), becomes a strip-space
patch, and its pixels are computed from the page as it currently looks — raw page, then the automatic
cleaning (patches.npz, patches_lama.npz), then earlier hand patches — so an inpainted or cloned stroke never
pulls back lettering the automatic pass already removed. Export applies the patches last, in order.
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from omniscan.cleanup.pixels import fill_pixels, inpaint_pixels, ring_color
from omniscan.cleanup.strip import strip_crop
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import (
    RGB,
    BBox,
    CleanupArtifact,
    CleanupMethod,
    CleanupPatch,
    IngestArtifact,
    InpaintArtifact,
)
from omniscan.edits.store import EditNotFoundError
from omniscan.inpaint.patches import load_patches

CLEANUP_FILE = "cleanup.json"
CLEANUP_NPZ = "cleanup.npz"
MAX_SIDE_PX = 4096  # a stroke's bounding box may be at most this big on either side (strip px)
CONTEXT_PX = 24  # inpainting sees this much of the page around the stroke

_LOCK = threading.RLock()  # one cleanup edit at a time (read-modify-write of both files)

Arrays = dict[str, tuple[np.ndarray | None, np.ndarray]]  # patch id -> (pixels or None for restore, mask)


def load_ingest(paths: ChapterPaths) -> IngestArtifact:
    """The chapter's ingest.json (a cleanup needs the strip's pages)."""
    path = paths.artifact("ingest.json")
    if not path.is_file():
        raise EditNotFoundError("ingest.json not found — run the ingest stage first")
    return IngestArtifact.load(path)


def load_cleanup(paths: ChapterPaths) -> CleanupArtifact | None:
    """The chapter's cleanup.json, or None when nothing was cleaned by hand."""
    path = paths.artifact(CLEANUP_FILE)
    return CleanupArtifact.load(path) if path.is_file() else None


def load_arrays(paths: ChapterPaths) -> Arrays:
    """cleanup.npz as patch id -> (pixels uint8 [h, w, 3] or None, mask bool [h, w]); {} when missing."""
    path = paths.artifact(CLEANUP_NPZ)
    if not path.is_file():
        return {}
    pixels: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    with np.load(path) as data:
        for key in data.files:
            patch_id, _sep, field = key.rpartition(".")
            (pixels if field == "pixels" else masks)[patch_id] = data[key]
    return {patch_id: (pixels.get(patch_id), mask) for patch_id, mask in masks.items()}


def _save(
    paths: ChapterPaths, artifact: CleanupArtifact, arrays: Mapping[str, tuple[np.ndarray | None, np.ndarray]]
) -> None:
    """Write cleanup.npz (atomically), then cleanup.json."""
    flat: dict[str, np.ndarray] = {}
    for patch_id, (pixels, mask) in arrays.items():
        flat[f"{patch_id}.mask"] = mask
        if pixels is not None:
            flat[f"{patch_id}.pixels"] = pixels
    npz = paths.artifact(CLEANUP_NPZ)
    npz.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=npz.parent, prefix=npz.name, suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **flat)  # type: ignore[arg-type]  # plain arrays, never pickled
        tmp.replace(npz)
    finally:
        tmp.unlink(missing_ok=True)
    artifact.save(paths.artifact(CLEANUP_FILE))


def page_to_strip(ingest: IngestArtifact, page: int, box: BBox, mask: np.ndarray) -> tuple[BBox, np.ndarray]:
    """A stroke painted on page `page` (its own pixels: `box` and a bool mask of that box's size) as a strip
    box and mask, clamped into the strip. ValueError for an unknown page or a stroke that covers nothing."""
    source = next((file for file in ingest.files if file.index == page), None)
    if source is None:
        raise ValueError(f"no page with index {page} in ingest.json")
    if mask.shape != (box.height, box.width):
        raise ValueError(f"mask is {mask.shape[1]}x{mask.shape[0]}, box is {box.width}x{box.height}")
    scale = source.scale if source.scale > 0 else 1.0
    full = BBox(
        x0=math.floor(box.x0 * scale),
        y0=source.y0 + math.floor(box.y0 * scale),
        x1=max(math.ceil(box.x1 * scale), math.floor(box.x0 * scale) + 1),
        y1=source.y0 + max(math.ceil(box.y1 * scale), math.floor(box.y0 * scale) + 1),
    )
    if full.width > MAX_SIDE_PX or full.height > MAX_SIDE_PX:
        raise ValueError(f"stroke is larger than {MAX_SIDE_PX} px on a side")
    scaled = mask
    if (full.width, full.height) != (box.width, box.height):
        image = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (full.width, full.height), Image.Resampling.NEAREST
        )
        scaled = np.asarray(image) > 127
    clamped = BBox(
        x0=max(full.x0, 0),
        y0=max(full.y0, 0),
        x1=min(full.x1, ingest.strip_width),
        y1=min(full.y1, ingest.strip_height),
    )
    if clamped.width <= 0 or clamped.height <= 0:
        raise ValueError("the stroke lies outside the strip")
    cut = scaled[clamped.y0 - full.y0 : clamped.y1 - full.y0, clamped.x0 - full.x0 : clamped.x1 - full.x0]
    if not cut.any():
        raise ValueError("the stroke covers no pixel")
    return clamped, np.ascontiguousarray(cut)


def _paste(crop: np.ndarray, crop_box: BBox, box: BBox, pixels: np.ndarray, mask: np.ndarray) -> None:
    """Replace `crop`'s pixels with `pixels` where `mask` is True, for the part of `box` inside `crop_box`."""
    x0, y0 = max(box.x0, crop_box.x0), max(box.y0, crop_box.y0)
    x1, y1 = min(box.x1, crop_box.x1), min(box.y1, crop_box.y1)
    if x0 >= x1 or y0 >= y1:
        return
    sel = mask[y0 - box.y0 : y1 - box.y0, x0 - box.x0 : x1 - box.x0]
    target = crop[y0 - crop_box.y0 : y1 - crop_box.y0, x0 - crop_box.x0 : x1 - crop_box.x0]
    target[sel] = pixels[y0 - box.y0 : y1 - box.y0, x0 - box.x0 : x1 - box.x0][sel]


def current_crop(
    paths: ChapterPaths, ingest: IngestArtifact, box: BBox, *, before: str | None = None
) -> np.ndarray:
    """The page inside `box` as export would show it before any lettering: raw, the automatic cleaning, then
    the hand patches in order (only those painted before patch `before`, when given)."""
    raw = strip_crop(paths, ingest, box)
    crop = raw.copy()
    for name, items_name in (("patches.npz", "inpaint.json"), ("patches_lama.npz", "inpaint_lama.json")):
        npz, items_path = paths.artifact(name), paths.artifact(items_name)
        if not npz.is_file() or not items_path.is_file():
            continue
        patches = load_patches(npz)
        for item in InpaintArtifact.load(items_path).items:
            if item.region_id in patches:
                pixels, mask = patches[item.region_id]
                _paste(crop, box, item.box, pixels, mask)
    artifact = load_cleanup(paths)
    if artifact is not None and fits_strip(artifact, ingest):
        arrays = load_arrays(paths)
        for patch in artifact.patches:
            if patch.id == before:
                break
            if patch.id not in arrays:
                continue
            pixels, mask = arrays[patch.id]
            if pixels is None:  # restore: the raw page under the mask
                _paste(crop, box, patch.box, strip_crop(paths, ingest, patch.box), mask)
            else:
                _paste(crop, box, patch.box, pixels, mask)
    return crop


def fits_strip(artifact: CleanupArtifact, ingest: IngestArtifact) -> bool:
    """True when the cleanup was painted on a strip of this size (a re-imported chapter invalidates it)."""
    return (artifact.strip_width, artifact.strip_height) == (ingest.strip_width, ingest.strip_height)


def _grow(box: BBox, px: int, ingest: IngestArtifact) -> BBox:
    """`box` grown by `px` on every side, clamped into the strip."""
    return BBox(
        x0=max(box.x0 - px, 0),
        y0=max(box.y0 - px, 0),
        x1=min(box.x1 + px, ingest.strip_width),
        y1=min(box.y1 + px, ingest.strip_height),
    )


def next_patch_id(artifact: CleanupArtifact) -> str:
    """The next free patch id (c0001, c0002, …)."""
    used = {patch.id for patch in artifact.patches}
    number = 1
    while f"c{number:04d}" in used:
        number += 1
    return f"c{number:04d}"


def add_patch(
    paths: ChapterPaths,
    *,
    page: int,
    box: BBox,
    mask: np.ndarray,
    method: CleanupMethod,
    color: RGB | None = None,
    offset: tuple[int, int] | None = None,
) -> CleanupPatch:
    """Clean a brush stroke painted on page `page` (page pixels) and append it to the chapter's cleanup.

    "fill" uses `color` or the median colour around the stroke; "inpaint" rebuilds it from the surrounding
    page; "clone" copies the page `offset` (page pixels, source minus destination) away; "restore" shows the
    raw page again. ValueError for a bad stroke, a clone source outside the strip or a missing offset."""
    with _LOCK:
        ingest = load_ingest(paths)
        sbox, smask = page_to_strip(ingest, page, box, mask)
        artifact = load_cleanup(paths)
        arrays = load_arrays(paths)
        if artifact is None or not fits_strip(artifact, ingest):
            artifact = CleanupArtifact(strip_width=ingest.strip_width, strip_height=ingest.strip_height)
            arrays = {}
        patch_id = next_patch_id(artifact)
        strip_offset: tuple[int, int] | None = None
        pixels: np.ndarray | None = None
        if method == "fill":
            if color is None:
                context_box = _grow(sbox, CONTEXT_PX, ingest)
                context = current_crop(paths, ingest, context_box)
                color = ring_color(context, _placed(smask, sbox, context_box))
            pixels = fill_pixels(smask.shape, color)
        elif method == "inpaint":
            context_box = _grow(sbox, CONTEXT_PX, ingest)
            rebuilt = inpaint_pixels(
                current_crop(paths, ingest, context_box), _placed(smask, sbox, context_box)
            )
            pixels = _cut(rebuilt, context_box, sbox)
        elif method == "clone":
            if offset is None:
                raise ValueError("clone needs an offset (the source point minus the first painted point)")
            source = next(file for file in ingest.files if file.index == page)
            scale = source.scale if source.scale > 0 else 1.0
            strip_offset = (round(offset[0] * scale), round(offset[1] * scale))
            src = BBox(
                x0=sbox.x0 + strip_offset[0],
                y0=sbox.y0 + strip_offset[1],
                x1=sbox.x1 + strip_offset[0],
                y1=sbox.y1 + strip_offset[1],
            )
            if src.x0 < 0 or src.y0 < 0 or src.x1 > ingest.strip_width or src.y1 > ingest.strip_height:
                raise ValueError("the clone source lies outside the strip")
            pixels = current_crop(paths, ingest, src)
        patch = CleanupPatch(
            id=patch_id,
            box=sbox,
            method=method,
            color=color if method == "fill" else None,
            offset=strip_offset,
            mask_px=int(smask.sum()),
        )
        artifact.patches.append(patch)
        arrays[patch_id] = (pixels, smask)
        _save(paths, artifact, arrays)
        return patch


def _placed(mask: np.ndarray, box: BBox, context_box: BBox) -> np.ndarray:
    """`mask` (the size of `box`) placed into an all-False mask the size of `context_box`."""
    placed = np.zeros((context_box.height, context_box.width), dtype=bool)
    placed[
        box.y0 - context_box.y0 : box.y1 - context_box.y0, box.x0 - context_box.x0 : box.x1 - context_box.x0
    ] = mask
    return placed


def _cut(pixels: np.ndarray, context_box: BBox, box: BBox) -> np.ndarray:
    """The part of `pixels` (the size of `context_box`) under `box`."""
    return np.ascontiguousarray(
        pixels[
            box.y0 - context_box.y0 : box.y1 - context_box.y0,
            box.x0 - context_box.x0 : box.x1 - context_box.x0,
        ]
    )


def delete_patch(paths: ChapterPaths, patch_id: str) -> None:
    """Remove one hand patch (its pixels come back as the patches under it make them)."""
    with _LOCK:
        artifact = load_cleanup(paths)
        if artifact is None or all(patch.id != patch_id for patch in artifact.patches):
            raise EditNotFoundError(f"cleanup patch {patch_id!r} not found")
        arrays = load_arrays(paths)
        artifact.patches = [patch for patch in artifact.patches if patch.id != patch_id]
        arrays.pop(patch_id, None)
        _save(paths, artifact, arrays)


def patch_rgba(paths: ChapterPaths, patch_id: str) -> np.ndarray:
    """One hand patch as uint8 [h, w, 4]: its pixels (the raw page for "restore"), alpha 255 under the mask."""
    artifact = load_cleanup(paths)
    arrays = load_arrays(paths)
    patch = None if artifact is None else next((p for p in artifact.patches if p.id == patch_id), None)
    if patch is None or patch_id not in arrays:
        raise EditNotFoundError(f"cleanup patch {patch_id!r} not found")
    pixels, mask = arrays[patch_id]
    if pixels is None:
        pixels = strip_crop(paths, load_ingest(paths), patch.box)
    rgba = np.empty((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = pixels
    rgba[..., 3] = np.where(mask, 255, 0)
    return rgba

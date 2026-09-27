"""Tests for omniscan.cleanup — hand cleanup patches computed from real page files (torch-free)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from omniscan.cleanup import store
from omniscan.cleanup.pixels import ring, ring_color
from omniscan.cleanup.strip import strip_crop
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import BBox, IngestArtifact, InpaintArtifact, InpaintItem, SourceFile
from omniscan.edits.store import EditNotFoundError

GREY = (120, 120, 120)
INK = (10, 10, 10)


def box(x0: int, y0: int, x1: int, y1: int) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


@pytest.fixture
def paths(tmp_path: Path) -> ChapterPaths:
    """Two 200x100 PNG pages: page 0 grey with a black 'glyph' at (50..70, 40..60); page 1 is 100x50 (scale 2)
    with a horizontal gradient."""
    paths = ChapterPaths(
        series="S",
        chapter="Chapter 1",
        raw_dir=tmp_path / "library" / "S" / "Chapter 1",
        work_dir=tmp_path / "work" / "S" / "Chapter 1",
        output_dir=tmp_path / "output" / "S" / "Chapter 1",
        filtered_dir=tmp_path / "output" / "S" / "_filtered" / "Chapter 1",
    )
    paths.raw_dir.mkdir(parents=True)
    page0 = np.full((100, 200, 3), GREY, dtype=np.uint8)
    page0[40:60, 50:70] = INK
    Image.fromarray(page0).save(paths.raw_dir / "001.png")
    gradient = np.repeat(np.repeat(np.arange(100, dtype=np.uint8)[None, :, None] * 2, 50, axis=0), 3, axis=2)
    Image.fromarray(gradient).save(paths.raw_dir / "002.png")
    IngestArtifact(
        series="S",
        chapter="Chapter 1",
        strip_width=200,
        strip_height=200,
        files=[
            SourceFile(index=0, name="001.png", sha256="0" * 64, width=200, height=100, y0=0, y1=100),
            SourceFile(
                index=1, name="002.png", sha256="1" * 64, width=100, height=50, y0=100, y1=200, scale=2.0
            ),
        ],
    ).save(paths.artifact("ingest.json"))
    return paths


def full(width: int, height: int) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def auto_clean_glyph(paths: ChapterPaths) -> None:
    """An automatic flat-fill patch that already painted the glyph grey (as patches.npz holds it)."""
    pixels = np.full((20, 20, 3), GREY, dtype=np.uint8)
    np.savez(paths.artifact("patches.npz"), **{"r0001.pixels": pixels, "r0001.mask": full(20, 20)})
    InpaintArtifact(items=[InpaintItem(region_id="r0001", box=box(50, 40, 70, 60), method="flat")]).save(
        paths.artifact("inpaint.json")
    )


def test_strip_crop_reads_and_scales_the_pages(paths: ChapterPaths) -> None:
    ingest = store.load_ingest(paths)
    crop = strip_crop(paths, ingest, box(45, 90, 75, 110))  # across both pages
    assert tuple(crop[0, 0]) == GREY and tuple(crop[0, 10]) == GREY  # y=90 on page 0
    assert crop.shape == (20, 30, 3)
    lower = crop[10:, :, 0].astype(int)  # page 1, scaled x2: columns 45..75 hold gradient values ~45..75
    assert abs(int(lower[5, 0]) - 45) <= 3 and abs(int(lower[5, -1]) - 74) <= 3


def test_page_to_strip_scales_and_clamps(paths: ChapterPaths) -> None:
    ingest = store.load_ingest(paths)
    mask = np.zeros((10, 20), dtype=bool)
    mask[:, :10] = True
    sbox, smask = store.page_to_strip(ingest, 1, box(90, 10, 110, 20), mask)  # page 1 is scale 2
    assert sbox == box(180, 120, 200, 140)  # 220 wide at x2, clamped to the strip's 200
    assert smask.shape == (20, 20) and smask[:, :20].all()
    with pytest.raises(ValueError, match="no page with index 7"):
        store.page_to_strip(ingest, 7, box(0, 0, 2, 2), full(2, 2))
    with pytest.raises(ValueError, match="covers no pixel"):
        store.page_to_strip(ingest, 0, box(0, 0, 2, 2), np.zeros((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="mask is"):
        store.page_to_strip(ingest, 0, box(0, 0, 4, 4), full(2, 2))


def test_fill_takes_the_colour_around_the_stroke_or_the_given_one(paths: ChapterPaths) -> None:
    patch = store.add_patch(paths, page=0, box=box(52, 42, 68, 58), mask=full(16, 16), method="fill")
    assert (patch.id, patch.method, patch.color, patch.mask_px) == (
        "c0001",
        "fill",
        INK,
        256,
    )  # inside the glyph
    patch = store.add_patch(
        paths, page=0, box=box(100, 10, 110, 20), mask=full(10, 10), method="fill", color=(1, 2, 3)
    )
    pixels, mask = store.load_arrays(paths)[patch.id]
    assert pixels is not None and tuple(pixels[0, 0]) == (1, 2, 3) and mask.all()


def test_inpaint_samples_the_page_as_already_cleaned(paths: ChapterPaths) -> None:
    """The automatic pass painted the glyph grey; a stroke over its edge must not pull the ink back."""
    auto_clean_glyph(paths)
    # the stroke covers only the glyph's right edge: on the raw page, ink borders it on the left
    patch = store.add_patch(paths, page=0, box=box(66, 36, 76, 64), mask=full(10, 28), method="inpaint")
    pixels, _mask = store.load_arrays(paths)[patch.id]
    assert pixels is not None
    assert np.abs(pixels.astype(int) - np.array(GREY)).max() <= 4
    # the fill's automatic colour also comes from the cleaned page (here: the inpainted stroke above)
    color = store.add_patch(paths, page=0, box=box(55, 45, 65, 55), mask=full(10, 10), method="fill").color
    assert color is not None and max(abs(c - g) for c, g in zip(color, GREY, strict=True)) <= 4


def test_clone_copies_from_the_offset_and_refuses_a_source_outside(paths: ChapterPaths) -> None:
    patch = store.add_patch(
        paths, page=0, box=box(100, 40, 120, 60), mask=full(20, 20), method="clone", offset=(-50, 0)
    )
    pixels, _mask = store.load_arrays(paths)[patch.id]
    assert patch.offset == (-50, 0) and pixels is not None and tuple(pixels[5, 5]) == INK
    scaled = store.add_patch(
        paths, page=1, box=box(40, 10, 50, 20), mask=full(10, 10), method="clone", offset=(-5, 0)
    )
    assert scaled.offset == (-10, 0)  # page 1 pixels are 2 strip pixels
    with pytest.raises(ValueError, match="outside the strip"):
        store.add_patch(
            paths, page=0, box=box(0, 0, 10, 10), mask=full(10, 10), method="clone", offset=(-5, 0)
        )
    with pytest.raises(ValueError, match="needs an offset"):
        store.add_patch(paths, page=0, box=box(0, 0, 10, 10), mask=full(10, 10), method="clone")


def test_restore_stores_no_pixels_and_previews_the_raw_page(paths: ChapterPaths) -> None:
    auto_clean_glyph(paths)
    patch = store.add_patch(paths, page=0, box=box(50, 40, 60, 50), mask=full(10, 10), method="restore")
    pixels, mask = store.load_arrays(paths)[patch.id]
    assert pixels is None and mask.all()
    rgba = store.patch_rgba(paths, patch.id)
    assert tuple(rgba[0, 0]) == (*INK, 255)
    # later strokes see the restored ink
    ingest = store.load_ingest(paths)
    assert tuple(store.current_crop(paths, ingest, box(50, 40, 70, 60))[5, 5]) == INK
    assert tuple(store.current_crop(paths, ingest, box(50, 40, 70, 60))[15, 15]) == GREY  # still auto-cleaned


def test_delete_and_ids(paths: ChapterPaths) -> None:
    for _ in range(3):
        store.add_patch(paths, page=0, box=box(0, 0, 4, 4), mask=full(4, 4), method="fill", color=(0, 0, 0))
    store.delete_patch(paths, "c0002")
    artifact = store.load_cleanup(paths)
    assert artifact is not None and [p.id for p in artifact.patches] == ["c0001", "c0003"]
    assert set(store.load_arrays(paths)) == {"c0001", "c0003"}
    assert (
        store.add_patch(paths, page=0, box=box(0, 0, 4, 4), mask=full(4, 4), method="restore").id == "c0002"
    )
    with pytest.raises(EditNotFoundError):
        store.delete_patch(paths, "c0009")
    with pytest.raises(EditNotFoundError):
        store.patch_rgba(paths, "c0009")


def test_a_cleanup_of_another_strip_size_is_replaced(paths: ChapterPaths) -> None:
    store.add_patch(paths, page=0, box=box(0, 0, 4, 4), mask=full(4, 4), method="restore")
    artifact = store.load_cleanup(paths)
    assert artifact is not None
    artifact.model_copy(update={"strip_height": 999}).save(paths.artifact(store.CLEANUP_FILE))
    patch = store.add_patch(paths, page=0, box=box(0, 0, 4, 4), mask=full(4, 4), method="restore")
    fresh = store.load_cleanup(paths)
    assert patch.id == "c0001" and fresh is not None and fresh.strip_height == 200 and len(fresh.patches) == 1


def test_missing_ingest_is_not_found(tmp_path: Path) -> None:
    empty = ChapterPaths("S", "C", tmp_path / "r", tmp_path / "w", tmp_path / "o", tmp_path / "f")
    with pytest.raises(EditNotFoundError, match=r"ingest\.json"):
        store.add_patch(empty, page=0, box=box(0, 0, 2, 2), mask=full(2, 2), method="restore")


def test_ring_is_just_outside_the_mask() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[3:6, 3:6] = True
    band = ring(mask, 1)
    assert not (band & mask).any() and band[2, 4] and band[6, 4] and not band[0, 0]
    crop = np.zeros((9, 9, 3), dtype=np.uint8)
    crop[ring(mask)] = (9, 8, 7)  # ring_color samples the default-width band
    assert ring_color(crop, mask) == (9, 8, 7)
    assert ring_color(crop, np.ones((9, 9), dtype=bool)) == (255, 255, 255)

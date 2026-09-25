"""Tests for the region inpainting pipeline (card C6a, part 3) — hand-built strips and synthetic Korean pages."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from omniscan.core.config import InpaintConfig
from omniscan.core.schemas import BBox, OcrLine, Region
from omniscan.inpaint.pipeline import inpaint_regions
from tests.fixtures.korean_pages import make_korean_page, to_regions_artifact

SEEDS = range(6)


def to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL RGB image as a uint8 [3, H, W] CPU tensor."""
    return torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).contiguous()


def region(rid: str, kind: str, box: BBox, *, lines: list[OcrLine] | None = None) -> Region:
    """A region whose text box is `box`, with one line on it unless `lines` is given."""
    return Region(
        id=rid,
        slice_index=0,
        kind=kind,  # type: ignore[arg-type]
        bbox=box,
        lines=lines if lines is not None else [OcrLine(bbox=box, text="텍스트", score=0.99, engine="test")],
        text="텍스트",
        confidence=0.99,
    )


def test_white_bubble_is_flat_filled() -> None:
    strip = torch.full((3, 300, 400), 255, dtype=torch.uint8)
    strip[:, 80:120, 100:160] = 0  # the black text rectangle
    box = BBox(x0=100, y0=80, x1=160, y1=120)
    artifact, patches, metrics = inpaint_regions(
        strip, [region("r0001", "bubble_text", box)], InpaintConfig()
    )
    (item,) = artifact.items
    assert item.region_id == "r0001"
    assert item.method == "flat"
    assert item.fill == (255, 255, 255)
    assert item.needs_lama is False
    assert item.mask_px == 46 * 66  # the dilated rectangle area
    assert item.box == BBox(x0=92, y0=72, x1=168, y1=128)  # the rectangle grown by pad_px
    pixels, mask = patches["r0001"]
    assert pixels.shape == (3, 56, 76)
    assert mask.shape == (56, 76)
    assert torch.equal(pixels, torch.full((3, 56, 76), 255, dtype=torch.uint8))
    assert metrics == {
        "regions": 1.0,
        "flat": 1.0,
        "needs_lama": 0.0,
        "skipped": 0.0,
        "mask_px": 3036.0,
        "glyph_masks": 0.0,  # the rectangle flat fill succeeded first
    }


def test_skips_lineless_and_kept_watermarks_and_flags_lama() -> None:
    generator = torch.Generator().manual_seed(3)
    strip = torch.full((3, 300, 400), 255, dtype=torch.uint8)
    strip[:, 200:260, 100:300] = torch.randint(0, 256, (3, 60, 200), generator=generator, dtype=torch.uint8)
    regions = [
        region("r0001", "watermark", BBox(x0=10, y0=10, x1=50, y1=30)),
        region("r0002", "bubble_text", BBox(x0=10, y0=60, x1=50, y1=90), lines=[]),
        region("r0003", "sfx", BBox(x0=20, y0=20, x1=60, y1=40)),
        region("r0004", "bubble_text", BBox(x0=150, y0=210, x1=250, y1=250)),
    ]
    artifact, patches, metrics = inpaint_regions(strip, regions, InpaintConfig(remove_watermarks=False))
    assert [item.region_id for item in artifact.items] == ["r0003", "r0004"]
    assert set(patches) == {"r0003", "r0004"}
    for item in artifact.items:
        assert item.method == "none"
        assert item.fill is None
        assert item.needs_lama is True
    sfx_pixels, _ = patches["r0003"]
    assert torch.equal(sfx_pixels, strip[:, 12:48, 12:68])  # the unchanged crop
    assert metrics == {
        "regions": 2.0,
        "flat": 0.0,
        "needs_lama": 2.0,
        "skipped": 2.0,
        # r0003 sits on plain white (no ink to find): its dilated line box; r0004's random pixels split
        # into two "ink" clusters whose grown mask covers almost the whole dilated box (4876 px)
        "mask_px": 1196.0 + 4856.0,
        "glyph_masks": 1.0,
    }


def test_watermark_text_is_erased_like_lettering() -> None:
    # ad text stamped on a flat margin: its read line is flat-filled like any other lettering
    strip = torch.full((3, 200, 300), 240, dtype=torch.uint8)
    strip[:, 90:110, 100:200] = 30
    box = BBox(x0=100, y0=90, x1=200, y1=110)
    artifact, patches, _ = inpaint_regions(strip, [region("r0001", "watermark", box)], InpaintConfig())
    (item,) = artifact.items
    assert item.method == "flat" and item.fill == (240, 240, 240)
    pixels, _ = patches["r0001"]
    assert int(pixels.min()) == 240


def test_a_stored_watermark_zone_is_erased_whole() -> None:
    # a fixed-position zone nothing was read in, over busy art: LaMa gets the whole box grown by
    # mask_dilate_px — never a glyph mask, whatever ink the zone holds
    generator = torch.Generator().manual_seed(5)
    strip = torch.randint(0, 256, (3, 200, 300), generator=generator, dtype=torch.uint8)
    strip[:, 50:70, 60:200] = 0
    zone = BBox(x0=40, y0=40, x1=220, y1=80)
    artifact, _, metrics = inpaint_regions(
        strip, [region("r0001", "watermark", zone, lines=[])], InpaintConfig()
    )
    (item,) = artifact.items
    assert item.needs_lama is True and item.method == "none"
    assert item.mask_px == (180 + 6) * (40 + 6)
    assert metrics["glyph_masks"] == 0.0


def test_a_stored_watermark_zone_on_a_flat_margin_is_flat_filled() -> None:
    strip = torch.full((3, 200, 300), 255, dtype=torch.uint8)
    strip[:, 170:190, 200:290] = 90  # a grey logo in the corner
    zone = BBox(x0=190, y0=160, x1=295, y1=195)
    artifact, patches, _ = inpaint_regions(
        strip, [region("r0001", "watermark", zone, lines=[])], InpaintConfig()
    )
    (item,) = artifact.items
    assert item.method == "flat" and item.fill == (255, 255, 255)
    assert int(patches["r0001"][0].min()) == 255


def test_no_regions_gives_empty_artifact() -> None:
    strip = torch.full((3, 50, 50), 255, dtype=torch.uint8)
    artifact, patches, metrics = inpaint_regions(strip, [], InpaintConfig())
    assert artifact.items == []
    assert patches == {}
    assert metrics == {
        "regions": 0.0,
        "flat": 0.0,
        "needs_lama": 0.0,
        "skipped": 0.0,
        "mask_px": 0.0,
        "glyph_masks": 0.0,
    }


def assert_patch_cleans_like_clean(
    strip: torch.Tensor,
    clean: torch.Tensor,
    truth: torch.Tensor,
    pixels: torch.Tensor,
    mask: torch.Tensor,
    box: BBox,
) -> None:
    """Applying the patch (replace mask pixels) leaves < 1 % of the truth text off `clean`; the rest is untouched."""
    crop = strip[:, box.y0 : box.y1, box.x0 : box.x1]
    applied = torch.where(mask, pixels, crop)
    box_truth = truth[box.y0 : box.y1, box.x0 : box.x1]
    diff = (applied.float() - clean[:, box.y0 : box.y1, box.x0 : box.x1].float()).abs().amax(dim=0)
    bad = int((diff > 40)[box_truth].sum())
    total = int(box_truth.sum())
    assert total > 0
    assert bad / total < 0.01
    outside = ~mask
    assert torch.equal(pixels[:, outside], crop[:, outside])


@pytest.mark.parametrize("seed", SEEDS)
def test_korean_pages_flat_background(seed: int) -> None:
    page = make_korean_page(seed, background="flat", n_bubbles=4, n_free=1, n_sfx=1)
    strip = to_tensor(page.image)
    clean = to_tensor(page.clean)
    truth = torch.from_numpy(page.text_mask)
    artifact, patches, _ = inpaint_regions(strip, to_regions_artifact(page).regions, InpaintConfig())
    assert len(artifact.items) == 6
    for item in artifact.items:
        kind = page.regions[int(item.region_id[1:]) - 1].kind
        if kind == "bubble_text":
            assert item.method == "flat"
        if kind == "free_text":
            assert item.method == "flat"  # the ring is the flat background
        if kind == "sfx":
            assert item.method == "flat"  # only the glyphs, whose surrounding band is the flat background
        if item.method == "flat":
            assert item.fill is not None and item.needs_lama is False
            pixels, mask = patches[item.region_id]
            assert_patch_cleans_like_clean(strip, clean, truth, pixels, mask, item.box)


@pytest.mark.parametrize("seed", SEEDS)
def test_korean_pages_noise_background_free_text_needs_lama(seed: int) -> None:
    page = make_korean_page(seed, background="noise", n_bubbles=4, n_free=1, n_sfx=1)
    strip = to_tensor(page.image)
    artifact, _, _ = inpaint_regions(strip, to_regions_artifact(page).regions, InpaintConfig())
    for item in artifact.items:
        if page.regions[int(item.region_id[1:]) - 1].kind == "free_text":
            assert item.method == "none"
            assert item.needs_lama is True

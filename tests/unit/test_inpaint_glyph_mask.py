"""Tests for glyph-precise text masks (inpaint/glyph_mask.py): real rendered text, synthetic pages, CPU."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

from omniscan.core.config import InpaintConfig
from omniscan.core.schemas import BBox, OcrLine, Region
from omniscan.gpu.morph import dilate, erode, otsu_threshold
from omniscan.inpaint.flat import line_mask
from omniscan.inpaint.glyph_mask import fill_holes, find_glyphs, local_ring, outline_width, stroke_width
from omniscan.inpaint.pipeline import inpaint_regions
from tests.fixtures.korean_pages import FONTS_DIR, make_korean_page, to_regions_artifact

FONT = FONTS_DIR / "NanumGothic-Bold.ttf"


def draw_text(
    text: str,
    *,
    size: int = 60,
    fill: tuple[int, int, int] = (0, 0, 0),
    background: tuple[int, int, int] = (255, 255, 255),
    stroke: int = 0,
    stroke_fill: tuple[int, int, int] = (255, 255, 255),
    pad: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    """Render `text` on a plain background: uint8 [3, h, w] crop, bool truth mask of changed pixels and
    the tight ink box (crop-relative), with `pad` px of background around it."""
    font = ImageFont.truetype(str(FONT), size)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    x0, y0, x1, y1 = (round(v) for v in probe.textbbox((0, 0), text, font=font, stroke_width=stroke))
    w, h = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
    image = Image.new("RGB", (w, h), background)
    ImageDraw.Draw(image).text(
        (pad - x0, pad - y0), text, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill
    )
    crop = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()
    truth = (crop != torch.tensor(background, dtype=torch.uint8)[:, None, None]).any(dim=0)
    return crop, truth, (pad, pad, w - pad, h - pad)


# ---------------------------------------------------------------- morphology helpers


def test_dilate_and_erode_are_duals_on_a_square() -> None:
    mask = torch.zeros((20, 20), dtype=torch.bool)
    mask[5:15, 5:15] = True
    assert int(dilate(mask, 2).sum()) == 14 * 14
    assert int(erode(mask, 2).sum()) == 6 * 6
    assert torch.equal(erode(dilate(mask, 3), 3), mask)


def test_erode_treats_the_image_border_as_unset() -> None:
    full = torch.ones((6, 6), dtype=torch.bool)
    assert int(erode(full, 1).sum()) == 16  # the outer ring is lost


def test_otsu_threshold_splits_two_levels() -> None:
    values = torch.cat([torch.full((100,), 20.0), torch.full((50,), 200.0)])
    cut = otsu_threshold(values)
    assert 20.0 < cut <= 200.0
    assert int((values < cut).sum()) == 100


def test_fill_holes_fills_enclosed_regions_only() -> None:
    ring = torch.zeros((30, 30), dtype=torch.bool)
    ring[5:25, 5:25] = True
    ring[8:22, 8:22] = False  # a closed frame: its inside is a hole
    open_c = ring.clone()
    open_c[12:18, 22:25] = False  # a gap in the right side: no longer enclosed
    assert torch.equal(fill_holes(ring)[8:22, 8:22], torch.ones((14, 14), dtype=torch.bool))
    assert not bool(fill_holes(open_c)[8:22, 8:22].any())
    assert torch.equal(fill_holes(ring) & ~ring, ~ring & fill_holes(ring))


def test_stroke_width_of_a_bar() -> None:
    bar = torch.zeros((40, 200), dtype=torch.bool)
    bar[10:16, 10:190] = True  # 6 px wide, 180 px long
    assert stroke_width(bar) == pytest.approx(6.0, abs=0.5)
    assert stroke_width(torch.zeros((5, 5), dtype=torch.bool)) == 0.0


def test_local_ring_is_the_band_around_the_mask() -> None:
    mask = torch.zeros((20, 20), dtype=torch.bool)
    mask[8:12, 8:12] = True
    ring = local_ring(mask, 2)
    assert not bool((ring & mask).any())
    assert int(ring.sum()) == 8 * 8 - 4 * 4


# ---------------------------------------------------------------- find_glyphs


def test_black_text_on_white_is_found_and_covered() -> None:
    crop, truth, box = draw_text("괜찮아요?", size=80)
    glyphs = find_glyphs(crop, [box])
    assert glyphs is not None
    assert glyphs.ink_is_dark is True
    assert not bool((truth & ~glyphs.mask).any())  # every drawn pixel is removed
    box_px = (box[2] - box[0]) * (box[3] - box[1])
    assert int(glyphs.mask.sum()) < 0.75 * box_px  # much less than the whole line box


def test_white_text_on_dark_is_found() -> None:
    crop, truth, box = draw_text("SLAM", fill=(250, 250, 250), background=(20, 30, 60))
    glyphs = find_glyphs(crop, [box])
    assert glyphs is not None
    assert glyphs.ink_is_dark is False
    assert not bool((truth & ~glyphs.mask).any())


def test_heavy_lettering_that_fills_most_of_its_box_is_still_ink() -> None:
    # the ink is the majority inside the tight box; the border (background) decides the side
    crop, truth, box = draw_text("■■", size=90, pad=10)
    glyphs = find_glyphs(crop, [box], max_ink=0.99)
    assert glyphs is not None and glyphs.ink_is_dark is True
    assert not bool((truth & ~glyphs.mask).any())


def test_outlined_lettering_covers_fill_and_outline() -> None:
    crop, truth, box = draw_text(
        "쾅!", size=110, fill=(40, 90, 230), background=(225, 170, 180), stroke=6, stroke_fill=(255, 255, 255)
    )
    glyphs = find_glyphs(crop, [box], grow=1.6)
    assert glyphs is not None
    assert not bool((truth & ~glyphs.mask).any())


def test_coloured_text_of_the_background_brightness_is_found() -> None:
    # green and lavender have almost the same luminance: only a colour split separates them
    crop, truth, box = draw_text(
        "슈웅", size=90, fill=(90, 220, 40), background=(205, 205, 235), stroke=5, stroke_fill=(255, 255, 255)
    )
    glyphs = find_glyphs(crop, [box], grow=1.6)
    assert glyphs is not None
    assert not bool((truth & ~glyphs.mask).any())


def test_coloured_text_of_exactly_the_background_luminance_is_found() -> None:
    # red (luma ~111) on teal (luma ~106), no outline: brightness alone shows no text at all
    crop, truth, box = draw_text("BAM", size=90, fill=(230, 60, 60), background=(40, 140, 100))
    glyphs = find_glyphs(crop, [box])
    assert glyphs is not None
    assert not bool((truth & ~glyphs.mask).any())


def test_a_frame_around_the_text_does_not_swallow_the_box() -> None:
    # a caption frame inside the text box: filling what the ink encloses would mask the whole box
    crop, _, _ = draw_text("Seoul, 2026", size=40, pad=30)
    crop[:, 6:-6, 6:12] = 0
    crop[:, 6:-6, -12:-6] = 0
    crop[:, 6:12, 6:-6] = 0
    crop[:, -12:-6, 6:-6] = 0
    frame_box = (6, 6, crop.shape[2] - 6, crop.shape[1] - 6)
    glyphs = find_glyphs(crop, [frame_box])
    assert glyphs is not None
    assert int(glyphs.mask.sum()) < 0.6 * (frame_box[2] - frame_box[0]) * (frame_box[3] - frame_box[1])


def test_outline_width_measures_the_outline() -> None:
    crop, _, box = draw_text("O", size=120, fill=(0, 0, 0), background=(200, 200, 200), stroke=4)
    glyphs = find_glyphs(crop, [box], grow=3.0, max_grow_px=20)
    assert glyphs is not None
    assert outline_width(crop, glyphs.body, 20) >= 3


@pytest.mark.parametrize(
    "crop",
    [
        torch.full((3, 60, 200), 240, dtype=torch.uint8),  # nothing drawn
        torch.cat(  # two halves only 20 levels apart: no text contrast
            [
                torch.full((3, 60, 100), 120, dtype=torch.uint8),
                torch.full((3, 60, 100), 140, dtype=torch.uint8),
            ],
            dim=2,
        ),
    ],
)
def test_no_glyphs_without_contrast(crop: torch.Tensor) -> None:
    assert find_glyphs(crop, [(10, 10, 190, 50)]) is None


def test_no_glyphs_in_a_tiny_box() -> None:
    crop, _, _ = draw_text("A", size=40)
    assert find_glyphs(crop, [(0, 0, 4, 4)]) is None


def test_boxes_limit_where_ink_is_looked_for() -> None:
    crop, _, box = draw_text("AB", size=80)
    left = (box[0], box[1], (box[0] + box[2]) // 2, box[3])
    glyphs = find_glyphs(crop, [left])
    assert glyphs is not None
    inside = line_mask(crop.shape[1], crop.shape[2], [left], dilate_px=0, device=crop.device)
    assert not bool((glyphs.ink & ~inside).any())


# ---------------------------------------------------------------- synthetic pages (every kind, every background)


def to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).contiguous()


@pytest.mark.parametrize("background", ["flat", "gradient", "noise"])
@pytest.mark.parametrize("seed", range(4))
def test_pipeline_masks_cover_every_text_pixel(background: str, seed: int) -> None:
    page = make_korean_page(seed, background=background, n_bubbles=3, n_free=1, n_sfx=1)  # type: ignore[arg-type]
    strip = to_tensor(page.image)
    truth = torch.from_numpy(page.text_mask)
    artifact, patches, metrics = inpaint_regions(strip, to_regions_artifact(page).regions, InpaintConfig())
    assert metrics["glyph_masks"] >= 1.0 or background == "flat"
    for item in artifact.items:
        _, mask = patches[item.region_id]
        box = item.box
        region_truth = truth[box.y0 : box.y1, box.x0 : box.x1]
        assert not bool((region_truth & ~mask).any()), item.region_id


@pytest.mark.parametrize("seed", range(4))
def test_lama_masks_on_art_are_smaller_than_line_boxes(seed: int) -> None:
    page = make_korean_page(seed, background="gradient", n_bubbles=2, n_free=1, n_sfx=1)
    strip = to_tensor(page.image)
    regions = to_regions_artifact(page).regions
    tight, _, _ = inpaint_regions(strip, regions, InpaintConfig())
    boxes, _, _ = inpaint_regions(strip, regions, InpaintConfig(glyph_mask=False))
    tight_px = sum(item.mask_px for item in tight.items if item.needs_lama)
    box_px = sum(item.mask_px for item in boxes.items if item.needs_lama)
    assert 0 < tight_px < 0.8 * box_px


def _text_region(rid: str, kind: str, box: BBox) -> Region:
    return Region(
        id=rid,
        slice_index=0,
        kind=kind,  # type: ignore[arg-type]
        bbox=box,
        lines=[OcrLine(bbox=box, text="쾅", score=0.99, engine="test")],
        text="쾅",
        confidence=0.99,
    )


def test_text_touching_a_bubble_outline_is_flat_filled_around_the_glyphs() -> None:
    crop, truth, box = draw_text("말도 안 돼", size=48, pad=14)
    strip = crop.clone()
    strip[:, :, :3] = 0  # a bubble outline running through the rectangle's padding
    region = _text_region("r1", "bubble_text", BBox(x0=box[0] - 8, y0=box[1], x1=box[2], y1=box[3]))
    artifact, patches, _ = inpaint_regions(strip, [region], InpaintConfig())
    (item,) = artifact.items
    assert item.method == "flat" and item.fill == (255, 255, 255)
    _, mask = patches["r1"]
    outline = torch.zeros_like(truth)
    outline[:, :3] = True
    patch = item.box
    assert not bool((mask & outline[patch.y0 : patch.y1, patch.x0 : patch.x1]).any())  # left alone
    assert not bool((truth[patch.y0 : patch.y1, patch.x0 : patch.x1] & ~mask).any())


def test_glyph_flat_fill_only_needs_the_band_around_the_glyphs_to_be_flat() -> None:
    # dark art at the patch's outer edge: the line box's ring is not flat, the band around the glyphs is
    cfg = InpaintConfig(pad_px=16)
    crop, truth, box = draw_text("괜찮아?", size=44, pad=16)
    crop[:, :3, :] = 0
    crop[:, -3:, :] = 0
    crop[:, :, :3] = 0
    crop[:, :, -3:] = 0
    region = _text_region("r1", "free_text", BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]))
    boxes_only, _, _ = inpaint_regions(crop, [region], InpaintConfig(pad_px=16, glyph_mask=False))
    assert [item.method for item in boxes_only.items] == ["none"]
    artifact, patches, _ = inpaint_regions(crop, [region], cfg)
    (item,) = artifact.items
    assert item.method == "flat" and item.fill == (255, 255, 255)
    _, mask = patches["r1"]
    assert not bool((truth & ~mask).any())


@pytest.mark.parametrize("mode", ["subtitle", "keep"])
def test_sfx_are_left_on_the_page_unless_replaced(mode: str) -> None:
    crop, _, box = draw_text("쾅", size=90)
    region = _text_region("r1", "sfx", BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]))
    artifact, _, metrics = inpaint_regions(crop, [region], InpaintConfig(), sfx_mode=mode)
    assert artifact.items == [] and metrics["skipped"] == 1.0
    replaced, _, _ = inpaint_regions(crop, [region], InpaintConfig(), sfx_mode="replace")
    assert [item.method for item in replaced.items] == ["flat"]

"""Tests for the LaMa region pipeline (card C6b, part 3) — hand-built patches, the inpainter is faked."""

from __future__ import annotations

import numpy as np
import torch

from omniscan.core.config import InpaintConfig
from omniscan.core.schemas import BBox, InpaintArtifact, InpaintItem
from omniscan.inpaint.lama_pipeline import _tile_boxes, _tile_starts, dilate_mask, lama_regions, window_origin


class FakeInpainter:
    """Records every (image, mask) call and paints all masked pixels (1, 2, 3)."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def inpaint(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self.calls.append((image, mask))
        out = image.clone()
        out[:, mask] = torch.tensor((1, 2, 3), dtype=torch.uint8, device=image.device)[:, None]
        return out


def np_patch(mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """A flat-fill patch in load_patches format: pixels uint8 [h, w, 3], mask bool [h, w]."""
    pixels = torch.randint(0, 256, (3, *mask.shape), dtype=torch.uint8).permute(1, 2, 0).numpy()
    return pixels, mask.numpy()


def artifact_with(rid: str, box: BBox, *, needs_lama: bool) -> InpaintArtifact:
    """One region: the LaMa pass processes it only when `needs_lama`."""
    return InpaintArtifact(
        items=[
            InpaintItem(
                region_id=rid,
                box=box,
                method="none" if needs_lama else "flat",
                fill=None if needs_lama else (9, 9, 9),
                needs_lama=needs_lama,
                mask_px=123,
            )
        ]
    )


# ---------------------------------------------------------------- dilate_mask / window_origin (tests 6, 7)


def test_dilate_mask_grows_clips_and_keeps_shape() -> None:
    mask = torch.zeros((9, 9), dtype=torch.bool)
    mask[4, 4] = True
    grown = dilate_mask(mask, 2)
    assert grown.shape == mask.shape and grown.dtype == torch.bool and grown.device == mask.device
    assert int(grown.sum()) == 25
    assert grown[2:7, 2:7].all() and not grown[1, 1] and not grown[7, 7]
    edge = torch.zeros((5, 5), dtype=torch.bool)
    edge[0, 0] = True
    clipped = dilate_mask(edge, 2)
    assert int(clipped.sum()) == 9  # the 5x5 block clipped to the 3x3 corner
    assert clipped[:3, :3].all()
    assert torch.equal(dilate_mask(edge, 0), edge)  # px=0: an equal copy


def test_window_origin_centres_and_clamps() -> None:
    assert window_origin(BBox(x0=100, y0=100, x1=200, y1=150), 800, 1400, 512) == (0, 0, 512, 512)
    assert window_origin(BBox(x0=700, y0=1300, x1=780, y1=1350), 800, 1400, 512) == (288, 888, 512, 512)
    assert window_origin(BBox(x0=50, y0=50, x1=250, y1=250), 300, 300, 512) == (0, 0, 300, 300)
    wx, wy, w, h = window_origin(BBox(x0=356, y0=600, x1=456, y1=700), 800, 1400, 512)
    assert (wx, wy, w, h) == (406 - 256, 650 - 256, 512, 512)  # centred: cx - 256 exactly


# ---------------------------------------------------------------- lama_regions (tests 8-11)


def noisy_strip(seed: int, height: int, width: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (3, height, width), generator=generator, dtype=torch.uint8)


def test_lama_region_inpaints_a_window_and_cuts_the_patch_back_out() -> None:
    strip = noisy_strip(7, 1400, 800)
    box = BBox(
        x0=100, y0=200, x1=220, y1=280
    )  # cx=160, cy=240: both clamp to 0, so window = strip[:512, :512]
    flat_mask = torch.zeros((80, 120), dtype=torch.bool)
    flat_mask[10:40, 20:90] = True
    patches = {"r0001": np_patch(flat_mask)}
    inpaint = artifact_with("r0001", box, needs_lama=True)
    inpaint.items.append(
        InpaintItem(region_id="r0002", box=BBox(x0=0, y0=0, x1=10, y1=10), method="flat", fill=(9, 9, 9))
    )
    fake = FakeInpainter()

    out_artifact, out_patches, metrics = lama_regions(strip, inpaint, patches, fake, InpaintConfig())

    assert len(fake.calls) == 1  # the needs_lama=False item is ignored
    image, mask = fake.calls[0]
    assert image.shape == (3, 512, 512) and image.dtype == torch.uint8
    assert torch.equal(image, strip[:, :512, :512])
    expected = torch.zeros((512, 512), dtype=torch.bool)
    expected[200:280, 100:220] = dilate_mask(flat_mask, 4)
    assert torch.equal(mask, expected) and mask.dtype == torch.bool

    (item,) = out_artifact.items
    assert item.region_id == "r0001"
    assert item.method == "lama"
    assert item.fill is None and item.needs_lama is False
    assert item.box == box
    assert item.mask_px == int(dilate_mask(flat_mask, 4).sum())
    pixels, patch_mask = out_patches["r0001"]
    assert pixels.shape == (3, 80, 120) and pixels.dtype == torch.uint8
    assert patch_mask.shape == (80, 120) and torch.equal(patch_mask, dilate_mask(flat_mask, 4))
    crop = strip[:, 200:280, 100:220]
    n = int(patch_mask.sum())
    assert torch.equal(pixels[:, patch_mask].T, torch.tensor((1, 2, 3), dtype=torch.uint8).repeat(n, 1))
    assert torch.equal(pixels[:, ~patch_mask], crop[:, ~patch_mask])  # the strip pixels came through
    assert metrics == {"regions": 1.0, "skipped_too_large": 0.0, "mask_px": float(n)}


def test_tile_starts_fits_whole_and_splits_with_overlap() -> None:
    assert _tile_starts(300, 448, 32) == [0]  # fits in one tile: no split
    assert _tile_starts(448, 448, 32) == [0]  # exactly the limit: still one tile
    assert _tile_starts(460, 448, 32) == [0, 12]  # 460 - 448 = 12: the last tile ends flush at 460
    assert _tile_starts(1000, 448, 32) == [0, 416, 552]  # step = 448 - 32 = 416; last tile flush at 1000-448


def test_tile_boxes_splits_only_the_axis_that_needs_it() -> None:
    box = BBox(x0=100, y0=200, x1=560, y1=280)  # 460 wide, 80 tall
    tiles = _tile_boxes(box, 448, 32)
    assert [(t.x0, t.x1, t.y0, t.y1) for t in tiles] == [(100, 548, 200, 280), (112, 560, 200, 280)]
    assert _tile_boxes(box, 500, 32) == [box]  # fits inside the limit: returned whole


def test_region_wider_than_the_usable_window_is_tiled_not_skipped() -> None:
    strip = noisy_strip(3, 1400, 800)
    box = BBox(x0=100, y0=200, x1=613, y1=280)  # 513 px wide > 512 - 2*32 (the real Solo Leveling case)
    flat_mask = torch.zeros((80, 513), dtype=torch.bool)
    flat_mask[10:40, 20:490] = True
    patches = {"r0001": np_patch(flat_mask)}
    fake = FakeInpainter()

    out_artifact, out_patches, metrics = lama_regions(
        strip, artifact_with("r0001", box, needs_lama=True), patches, fake, InpaintConfig()
    )

    assert len(fake.calls) == 2  # two tiles, each a full lama_window crop
    for image, _mask in fake.calls:
        assert image.shape == (3, 512, 512) and image.dtype == torch.uint8

    (item,) = out_artifact.items
    assert item.region_id == "r0001" and item.method == "lama" and item.needs_lama is False
    assert item.box == box
    expected_mask = dilate_mask(flat_mask, 4)
    assert item.mask_px == int(expected_mask.sum())
    pixels, patch_mask = out_patches["r0001"]
    assert pixels.shape == (3, 80, 513) and pixels.dtype == torch.uint8
    assert torch.equal(patch_mask, expected_mask)
    n = int(expected_mask.sum())
    assert torch.equal(pixels[:, expected_mask].T, torch.tensor((1, 2, 3), dtype=torch.uint8).repeat(n, 1))
    crop = strip[:, 200:280, 100:613]
    assert torch.equal(pixels[:, ~expected_mask], crop[:, ~expected_mask])  # untouched pixels came through
    assert metrics == {"regions": 1.0, "skipped_too_large": 0.0, "mask_px": float(n)}


def test_region_too_tall_and_too_wide_is_tiled_on_both_axes() -> None:
    strip = noisy_strip(9, 1400, 1400)
    box = BBox(x0=50, y0=50, x1=550, y1=550)  # 500x500, both axes over the 448 limit
    flat_mask = torch.zeros((500, 500), dtype=torch.bool)
    flat_mask[100:400, 100:400] = True
    patches = {"r0001": np_patch(flat_mask)}
    fake = FakeInpainter()

    out_artifact, out_patches, metrics = lama_regions(
        strip, artifact_with("r0001", box, needs_lama=True), patches, fake, InpaintConfig()
    )

    assert len(fake.calls) == 4  # a 2x2 tile grid
    (item,) = out_artifact.items
    assert item.box == box
    pixels, _patch_mask = out_patches["r0001"]
    assert pixels.shape == (3, 500, 500)
    expected_mask = dilate_mask(flat_mask, 4)
    n = int(expected_mask.sum())
    assert torch.equal(pixels[:, expected_mask].T, torch.tensor((1, 2, 3), dtype=torch.uint8).repeat(n, 1))
    assert metrics["skipped_too_large"] == 0.0


def test_strip_smaller_than_the_window_is_replicate_padded() -> None:
    strip = noisy_strip(11, 300, 300)
    box = BBox(x0=50, y0=60, x1=150, y1=120)
    flat_mask = torch.zeros((60, 100), dtype=torch.bool)
    flat_mask[5:30, 10:70] = True
    patches = {"r0001": np_patch(flat_mask)}
    fake = FakeInpainter()

    out_artifact, out_patches, metrics = lama_regions(
        strip, artifact_with("r0001", box, needs_lama=True), patches, fake, InpaintConfig()
    )

    image, mask = fake.calls[0]
    assert image.shape == (3, 512, 512) and image.dtype == torch.uint8
    assert torch.equal(image[:, :300, :300], strip)
    assert torch.equal(image[:, :300, 300:], strip[:, :, -1:].expand(3, 300, 212))
    assert torch.equal(image[:, 300:, :300], strip[:, -1:, :].expand(3, 212, 300))
    assert torch.equal(image[:, 300:, 300:], strip[:, -1:, -1:].expand(3, 212, 212))
    assert not mask[:, 300:].any() and not mask[300:, :300].any()
    expected = torch.zeros((512, 512), dtype=torch.bool)
    expected[60:120, 50:150] = dilate_mask(flat_mask, 4)
    assert torch.equal(mask, expected)
    (item,) = out_artifact.items
    assert item.method == "lama"
    pixels, patch_mask = out_patches["r0001"]
    assert pixels.shape == (3, 60, 100)  # still exactly the box size
    assert torch.equal(patch_mask, dilate_mask(flat_mask, 4))
    assert metrics["regions"] == 1.0


def test_two_regions_run_in_order_and_none_gives_an_empty_artifact() -> None:
    strip = noisy_strip(5, 1400, 800)
    box1 = BBox(x0=100, y0=200, x1=220, y1=280)  # window (0, 0)
    box2 = BBox(x0=500, y0=900, x1=620, y1=980)  # cx=560, cy=940: clamped window (288, 684)
    patches = {
        "r0001": np_patch(torch.zeros((80, 120), dtype=torch.bool)),
        "r0002": np_patch(torch.zeros((80, 120), dtype=torch.bool)),
    }
    patches["r0001"][1][30:60, 30:90] = True
    patches["r0002"][1][10:70, 40:100] = True
    inpaint = InpaintArtifact(
        items=[
            InpaintItem(region_id="r0001", box=box1, method="none", needs_lama=True, mask_px=1),
            InpaintItem(region_id="r0002", box=box2, method="none", needs_lama=True, mask_px=2),
        ]
    )
    fake = FakeInpainter()
    out_artifact, out_patches, metrics = lama_regions(strip, inpaint, patches, fake, InpaintConfig())

    assert [item.region_id for item in out_artifact.items] == ["r0001", "r0002"]
    assert set(out_patches) == {"r0001", "r0002"}
    assert len(fake.calls) == 2
    assert torch.equal(fake.calls[0][0], strip[:, :512, :512])
    assert torch.equal(fake.calls[1][0], strip[:, 684:1196, 288:800])
    for (_image, mask), (rid, by, bx) in zip(
        fake.calls, (("r0001", 200, 100), ("r0002", 900 - 684, 500 - 288)), strict=True
    ):
        expected = torch.zeros((512, 512), dtype=torch.bool)
        expected[by : by + 80, bx : bx + 120] = torch.from_numpy(patches[rid][1])
        assert torch.equal(mask, dilate_mask(expected, 4))
    mask_px = float(
        sum(int(dilate_mask(torch.from_numpy(patches[rid][1]), 4).sum()) for rid in ("r0001", "r0002"))
    )
    assert metrics == {"regions": 2.0, "skipped_too_large": 0.0, "mask_px": mask_px}

    fake2 = FakeInpainter()
    empty = lama_regions(strip, InpaintArtifact(items=[]), {}, fake2, InpaintConfig())
    assert empty[0].items == [] and empty[1] == {}
    assert empty[2] == {"regions": 0.0, "skipped_too_large": 0.0, "mask_px": 0.0}
    assert fake2.calls == []


# ---------------------------------------------------------------- no lettering as context


def test_every_tile_masks_the_whole_region_not_just_its_own_part() -> None:
    strip = noisy_strip(3, 1400, 800)
    box = BBox(x0=100, y0=200, x1=613, y1=280)  # tiled: 513 px wide
    flat_mask = torch.zeros((80, 513), dtype=torch.bool)
    flat_mask[10:40, 20:490] = True
    fake = FakeInpainter()
    lama_regions(
        strip,
        artifact_with("r0001", box, needs_lama=True),
        {"r0001": np_patch(flat_mask)},
        fake,
        InpaintConfig(),
    )
    grown = dilate_mask(flat_mask, 4)
    tiles = _tile_boxes(box, 512 - 2 * 32, 32)
    assert len(fake.calls) == len(tiles) == 2
    for tile, (_image, mask) in zip(tiles, fake.calls, strict=True):
        wx, _wy, _w, _h = window_origin(tile, 800, 1400, 512)
        inside = grown[:, max(0, wx - box.x0) : wx + 512 - box.x0]
        assert int(mask.sum()) == int(inside.sum())  # every masked pixel of the region inside the window


def test_a_neighbouring_region_waiting_for_lama_is_masked_too() -> None:
    strip = noisy_strip(5, 1400, 800)
    first, second = BBox(x0=100, y0=100, x1=200, y1=150), BBox(x0=250, y0=100, x1=350, y1=150)
    mask = torch.ones((50, 100), dtype=torch.bool)
    inpaint = InpaintArtifact(
        items=[
            InpaintItem(region_id=rid, box=box, method="none", needs_lama=True)
            for rid, box in (("r0001", first), ("r0002", second))
        ]
    )
    fake = FakeInpainter()
    lama_regions(strip, inpaint, {"r0001": np_patch(mask), "r0002": np_patch(mask)}, fake, InpaintConfig())
    first_mask = fake.calls[0][1]
    assert bool(first_mask[100:150, 250:350].all())  # r0002's lettering is no context for r0001
    second_image = fake.calls[1][0]
    wx, wy, _w, _h = window_origin(second, 800, 1400, 512)  # r0001 (x 100..200) is inside this window
    done = second_image[:, 100 - wy : 150 - wy, 100 - wx : 200 - wx]
    assert bool((done == torch.tensor((1, 2, 3), dtype=torch.uint8)[:, None, None]).all())  # its LaMa result


def test_flat_filled_neighbours_are_clean_context() -> None:
    strip = noisy_strip(8, 1400, 800)
    lama_box, flat_box = BBox(x0=100, y0=100, x1=200, y1=150), BBox(x0=250, y0=100, x1=350, y1=150)
    mask = torch.ones((50, 100), dtype=torch.bool)
    flat_pixels = np.full((50, 100, 3), 200, dtype=np.uint8)
    inpaint = InpaintArtifact(
        items=[
            InpaintItem(region_id="r0001", box=lama_box, method="none", needs_lama=True),
            InpaintItem(region_id="r0002", box=flat_box, method="flat", fill=(200, 200, 200)),
        ]
    )
    fake = FakeInpainter()
    lama_regions(
        strip, inpaint, {"r0001": np_patch(mask), "r0002": (flat_pixels, mask.numpy())}, fake, InpaintConfig()
    )
    ((image, _mask),) = fake.calls
    assert bool((image[:, 100:150, 250:350] == 200).all())

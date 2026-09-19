"""Tests for the inpaint_lama stage wiring (card C6b, part 4); the GPU test runs the full path with LaMa."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from omniscan.core.config import Config, GpuConfig, InpaintConfig, PathsConfig
from omniscan.core.schemas import (
    Band,
    BBox,
    IngestArtifact,
    InpaintArtifact,
    InpaintItem,
    Manifest,
    Slice,
    SlicesArtifact,
)
from omniscan.core.stage import ChapterContext, make_context, run_stage
from omniscan.ingest.stage import IngestStage
from omniscan.ingest.strip import load_strip
from omniscan.inpaint.lama_stage import LamaStage
from omniscan.inpaint.patches import load_patches, save_patches
from omniscan.inpaint.stage import InpaintStage
from tests.fixtures import images
from tests.fixtures.korean_pages import make_korean_page, to_regions_artifact

SERIES = "S"
CHAPTER = "Chapter 1"
PAGE = (400, 300)
STRIP_HEIGHT = 600


class FakeScheduler:
    """The GpuScheduler surface run_stage needs, returning a scripted model mapping."""

    def __init__(self, models: Mapping[str, Any]) -> None:
        self.models = dict(models)

    def acquire(self, group: str) -> Mapping[str, Any]:
        assert group == "inpaint"
        return self.models

    def reset_peak(self) -> None:
        pass

    def peak_gib(self) -> float:
        return 0.0


class FakeInpainter:
    """Records every (image, mask) call and paints all masked pixels (1, 2, 3)."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def inpaint(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self.calls.append((image, mask))
        out = image.clone()
        out[:, mask] = torch.tensor((1, 2, 3), dtype=torch.uint8, device=image.device)[:, None]
        return out


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """CPU-only config with all paths under tmp_path."""
    return Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
    )


def write_pages(cfg: Config) -> None:
    raw = cfg.paths.library_root / SERIES / CHAPTER
    raw.mkdir(parents=True, exist_ok=True)
    images.plain_jpeg(raw / "001.jpg", size=PAGE, color=(250, 250, 250))
    images.plain_jpeg(raw / "002.jpg", size=PAGE, color=(250, 250, 250))


def write_slices(cfg: Config) -> None:
    work = cfg.paths.work_root / SERIES / CHAPTER
    work.mkdir(parents=True, exist_ok=True)
    SlicesArtifact(
        strip_width=PAGE[0],
        strip_height=STRIP_HEIGHT,
        bands=[Band(y0=0, y1=STRIP_HEIGHT, color=(250, 250, 250))],
        slices=[Slice(index=0, y0=0, y1=STRIP_HEIGHT)],
    ).save(work / "slices.json")


def write_inpaint(cfg: Config, strip: torch.Tensor) -> None:
    """Hand-built inpaint.json + patches.npz as the flat stage would write them."""
    work = cfg.paths.work_root / SERIES / CHAPTER
    box = BBox(x0=100, y0=200, x1=220, y1=280)
    flat_mask = torch.zeros((80, 120), dtype=torch.bool)
    flat_mask[10:40, 20:90] = True
    pixels = strip[:, box.y0 : box.y1, box.x0 : box.x1].clone()
    save_patches(
        work / "patches.npz",
        {
            "r0001": (pixels, flat_mask),
            "r0002": (strip[:, 0:20, 0:20].clone(), torch.zeros((20, 20), dtype=torch.bool)),
        },
    )
    InpaintArtifact(
        items=[
            InpaintItem(region_id="r0001", box=box, method="none", fill=None, needs_lama=True, mask_px=1),
            InpaintItem(
                region_id="r0002",
                box=BBox(x0=0, y0=0, x1=20, y1=20),
                method="flat",
                fill=(250, 250, 250),
                needs_lama=False,
                mask_px=5,
            ),
        ]
    ).save(work / "inpaint.json")


def prepared_context(cfg: Config, scheduler: FakeScheduler) -> ChapterContext:
    """A chapter with ingest + slices + inpaint.json/patches.npz and the stage's GPU group faked."""
    write_pages(cfg)
    ctx = make_context(cfg, SERIES, CHAPTER, gpu=scheduler)
    assert run_stage(IngestStage(), ctx).status == "done"
    write_slices(cfg)
    write_inpaint(cfg, load_strip(ctx, IngestArtifact.load(ctx.paths.artifact("ingest.json"))))
    return ctx


# ---------------------------------------------------------------- stage (test 12)


def test_lama_stage_round_trip(cfg: Config) -> None:
    fake = FakeInpainter()
    ctx = prepared_context(cfg, FakeScheduler({"lama": fake}))

    outcome = run_stage(LamaStage(), ctx)
    assert outcome.status == "done"
    assert outcome.metrics["regions"] == 1.0
    artifact = InpaintArtifact.load(ctx.paths.artifact("inpaint_lama.json"))
    (item,) = artifact.items
    assert item.region_id == "r0001" and item.method == "lama" and item.needs_lama is False
    patches = load_patches(ctx.paths.artifact("patches_lama.npz"))
    assert set(patches) == {"r0001"}
    pixels, mask = patches["r0001"]
    assert pixels.shape == (item.box.height, item.box.width, 3)
    assert mask.shape == (item.box.height, item.box.width)
    assert Manifest.load(ctx.paths.manifest).stages["inpaint_lama"].status == "done"

    def fresh(model: FakeInpainter, config: Config = cfg) -> ChapterContext:
        return make_context(config, SERIES, CHAPTER, gpu=FakeScheduler({"lama": model}))

    assert run_stage(LamaStage(), fresh(fake)).status == "skipped"  # unchanged inputs and config

    cfg2 = cfg.model_copy(update={"inpaint": InpaintConfig(lama_window=256)})
    assert run_stage(LamaStage(), fresh(fake, cfg2)).status == "done"  # the changed window re-runs
    assert run_stage(LamaStage(), fresh(fake, cfg2)).status == "skipped"


def test_lama_stage_missing_patches_fails_with_documented_message(cfg: Config) -> None:
    ctx = prepared_context(cfg, FakeScheduler({"lama": FakeInpainter()}))
    (ctx.paths.work_dir / "patches.npz").unlink()

    outcome = run_stage(LamaStage(), ctx)
    assert outcome.status == "failed"
    assert outcome.error is not None
    assert "patches.npz missing — run the inpaint stage first" in outcome.error


# ---------------------------------------------------------------- full path with the real model (test 15)


def apply_patches(
    strip: torch.Tensor, artifact: InpaintArtifact, patches: Mapping[str, tuple[np.ndarray, np.ndarray]]
) -> None:
    """Replace each patch's masked pixels in the strip in place (flat patches first, LaMa last)."""
    for item in artifact.items:
        if item.region_id not in patches:
            continue
        pixels, mask = patches[item.region_id]
        crop = strip[:, item.box.y0 : item.box.y1, item.box.x0 : item.box.x1]
        mask_t = torch.from_numpy(np.ascontiguousarray(mask))
        pixels_t = torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1)
        crop[:, mask_t] = pixels_t[:, mask_t]


@pytest.mark.gpu
def test_lama_stage_full_path_on_a_korean_page(tmp_path: Path) -> None:
    import httpx

    from omniscan.gpu.device import resolve_device
    from omniscan.inpaint.lama import LamaInpainter
    from omniscan.inpaint.lama_stage import LamaStage

    device = resolve_device()
    models_dir = tmp_path / "models"
    try:
        inpainter = LamaInpainter.load(InpaintConfig(), models_dir, device)
    except httpx.HTTPError as exc:  # the 205 MB download failed (e.g. offline): skip, not fail
        pytest.skip(f"LaMa weights not downloadable: {exc}")

    page = make_korean_page(seed=2, n_bubbles=0, n_free=1, n_sfx=1, background="gradient")
    cfg = Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=models_dir,
        ),
    )
    raw = cfg.paths.library_root / SERIES / CHAPTER
    raw.mkdir(parents=True)
    page.image.save(raw / "001.jpg", format="JPEG", quality=95)

    ctx = make_context(cfg, SERIES, CHAPTER, gpu=FakeScheduler({"lama": inpainter}))
    assert run_stage(IngestStage(), ctx).status == "done"
    SlicesArtifact(
        strip_width=page.image.width,
        strip_height=page.image.height,
        bands=[Band(y0=0, y1=page.image.height, color=(250, 250, 250))],
        slices=[Slice(index=0, y0=0, y1=page.image.height)],
    ).save(ctx.paths.artifact("slices.json"))
    to_regions_artifact(page).save(ctx.paths.artifact("ocr.json"))

    assert run_stage(InpaintStage(), ctx).status == "done"
    flat = InpaintArtifact.load(ctx.paths.artifact("inpaint.json"))
    needs = {page.regions[int(item.region_id[1:]) - 1].kind: item for item in flat.items}
    assert set(needs) == {"free_text", "sfx"}
    assert all(item.needs_lama for item in flat.items)

    outcome = run_stage(LamaStage(), ctx)
    assert outcome.status == "done"
    lama_artifact = InpaintArtifact.load(ctx.paths.artifact("inpaint_lama.json"))
    assert [item.region_id for item in lama_artifact.items] == [item.region_id for item in flat.items]
    assert all(item.method == "lama" and item.needs_lama is False for item in lama_artifact.items)

    strip = load_strip(ctx, IngestArtifact.load(ctx.paths.artifact("ingest.json")))
    apply_patches(strip, flat, load_patches(ctx.paths.artifact("patches.npz")))
    apply_patches(strip, lama_artifact, load_patches(ctx.paths.artifact("patches_lama.npz")))

    clean = torch.from_numpy(np.array(page.clean, dtype=np.uint8)).permute(2, 0, 1)
    truth = torch.from_numpy(page.text_mask)
    diff = (strip.float() - clean.float()).abs().amax(dim=0)
    bad = int((diff > 60)[truth].sum())
    total = int(truth.sum())
    print(
        f"\nLaMa full path: {bad}/{total} truth text pixels differ from clean by > 60 ({100.0 * bad / total:.2f}%)"
    )
    assert bad / total <= 0.05
    torch.cuda.empty_cache()

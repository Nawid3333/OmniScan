"""Tests for the LaMa model wrapper (card C6b, part 2); the GPU test runs the real TorchScript model."""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import torch
from PIL import Image

from omniscan.core.config import InpaintConfig
from omniscan.inpaint.lama import LamaInpainter
from omniscan.inpaint.lama_pipeline import dilate_mask, window_origin
from tests.fixtures.korean_pages import make_korean_page

PAGE = (800, 1400)


class FakeScript:
    """Stand-in for the TorchScript module: records calls, returns the input (mask pixels stay black)."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], bool, bool]] = []

    def eval(self) -> FakeScript:
        return self

    def __call__(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        self.calls.append((tuple(x.shape), tuple(m.shape), bool(x.eq(0).all()), bool(m.eq(0).all())))
        return torch.zeros_like(x)


def to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL RGB image as a uint8 [3, H, W] CPU tensor."""
    return torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).contiguous()


# ---------------------------------------------------------------- CPU (fake model)


def test_inpaint_replaces_only_masked_pixels() -> None:
    model = FakeScript()
    inpainter = LamaInpainter(model, torch.device("cpu"), window=64)
    assert inpainter.window == 64
    image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))
    mask = torch.zeros((64, 64), dtype=torch.bool)
    mask[10:20, 30:44] = True
    out = inpainter.inpaint(image, mask)
    assert out.dtype == torch.uint8 and out.device == image.device
    assert torch.equal(out[:, ~mask], image[:, ~mask])  # bit-identical outside the mask
    assert torch.equal(out[:, mask], torch.zeros(3, int(mask.sum()), dtype=torch.uint8))
    assert model.calls == [((1, 3, 64, 64), (1, 1, 64, 64), False, False)]


def test_inpaint_rejects_wrong_shapes() -> None:
    inpainter = LamaInpainter(FakeScript(), torch.device("cpu"), window=64)
    image = torch.zeros((3, 64, 64), dtype=torch.uint8)
    mask = torch.zeros((64, 64), dtype=torch.bool)
    with pytest.raises(ValueError, match="uint8 \\[3, 64, 64\\]"):
        inpainter.inpaint(torch.zeros((3, 63, 64), dtype=torch.uint8), mask[:63])
    with pytest.raises(ValueError, match="bool \\[64, 64\\]"):
        inpainter.inpaint(image, torch.ones((64, 65), dtype=torch.bool))


def test_load_warms_up_on_the_fixed_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeScript()
    monkeypatch.setattr("omniscan.inpaint.lama.torch.jit.load", lambda path, map_location: fake)
    monkeypatch.setattr(
        "omniscan.inpaint.lama.ensure_lama_weights",
        lambda models_dir, cfg: tmp_path / cfg.lama_file,  # type: ignore[misc, return-value]
    )
    inpainter = LamaInpainter.load(InpaintConfig(lama_window=64), tmp_path, torch.device("cpu"))
    assert inpainter.window == 64 and inpainter.model is fake
    assert len(fake.calls) == 2  # two warm-up passes on zero tensors of the fixed shape
    assert fake.calls[0] == ((1, 3, 64, 64), (1, 1, 64, 64), True, True)


# ---------------------------------------------------------------- real model (GPU, test 14)


@pytest.fixture(scope="session")
def lama_models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A session-wide models dir so the 205 MB weights are downloaded once for all GPU tests."""
    return tmp_path_factory.mktemp("lama_models")


@pytest.mark.gpu
def test_real_lama_removes_text_on_a_gradient_page(lama_models_dir: Path) -> None:
    from omniscan.gpu.device import resolve_device

    device = resolve_device()
    try:
        inpainter = LamaInpainter.load(InpaintConfig(), lama_models_dir, device)
    except httpx.HTTPError as exc:  # the 205 MB download failed (e.g. offline): skip, not fail
        pytest.skip(f"LaMa weights not downloadable: {exc}")
    assert {p.dtype for p in inpainter.model.parameters()} == {torch.float32}  # fp32 only: fp16 fails

    page = make_korean_page(seed=2, n_bubbles=0, n_free=1, n_sfx=1, background="gradient")
    strip = to_tensor(page.image)
    truth = next(region for region in page.regions if region.kind == "free_text")
    wx, wy, w, h = window_origin(truth.bbox, PAGE[0], PAGE[1], 512)
    window = strip[:, wy : wy + h, wx : wx + w]
    region_area = torch.zeros((h, w), dtype=torch.bool)  # only the free-text region's own ink
    region_area[
        max(0, truth.bbox.y0 - wy) : min(h, truth.bbox.y1 - wy),
        max(0, truth.bbox.x0 - wx) : min(w, truth.bbox.x1 - wx),
    ] = True
    ink = torch.from_numpy(page.text_mask[wy : wy + h, wx : wx + w]) & region_area
    mask = dilate_mask(ink, 7)
    assert int(mask.sum()) > 0

    result = inpainter.inpaint(window, mask)
    # outside the mask bit-identical to the input (the inpainter moves the window to the device itself)
    assert torch.equal(result[:, ~mask], window[:, ~mask].to(result.device))
    clean = to_tensor(page.clean)[:, wy : wy + h, wx : wx + w].to(result.device)
    diff = (result.float() - clean.float()).abs().mean(dim=0)
    value = float(diff[mask].mean())
    print(f"\nLaMa mean abs diff inside the mask: {value:.2f} (of 255)")
    assert value < 15.0

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    inpainter.inpaint(window, mask)
    torch.cuda.synchronize(device)
    print(f"LaMa steady state: {(time.perf_counter() - t0) * 1000:.0f} ms per 512x512 call")
    torch.cuda.empty_cache()

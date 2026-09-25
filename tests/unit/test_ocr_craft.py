"""The CRAFT sound-effect sweeper (ocr/craft.py): network layout, score maps -> word boxes, weights on disk."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

import omniscan.ocr.craft as craft
from omniscan.models.catalog import ModelEntry
from omniscan.ocr.craft import CRAFT_MODEL_ID, Craft, CraftDetector, word_boxes


def blob(shape: tuple[int, int], cy: float, cx: float, sigma: float, peak: float) -> np.ndarray:
    """A Gaussian score blob (float32) centred at (cy, cx)."""
    ys, xs = np.mgrid[: shape[0], : shape[1]]
    return (peak * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma**2))).astype(np.float32)


def test_the_network_matches_the_published_checkpoint_layout() -> None:
    state = Craft().state_dict()
    assert len(state) == 154  # craft_mlt_25k.pth's keys, "module." prefix stripped
    for key in (
        "basenet.slice1.0.weight",
        "basenet.slice2.14.weight",
        "basenet.slice3.28.running_var",
        "basenet.slice4.38.bias",
        "basenet.slice5.1.weight",
        "upconv4.conv.4.num_batches_tracked",
        "conv_cls.8.bias",
    ):
        assert key in state
    assert state["basenet.slice5.1.weight"].shape == (1024, 512, 3, 3)
    assert state["conv_cls.8.weight"].shape == (2, 16, 1, 1)


def test_the_network_scores_at_half_resolution() -> None:
    with torch.inference_mode():
        out = Craft().eval()(torch.zeros(1, 3, 64, 96))
    assert out.shape == (1, 2, 32, 48)


def test_linked_characters_make_one_word_box() -> None:
    region = np.maximum(blob((60, 120), 30, 30, 5, 0.9), blob((60, 120), 30, 60, 5, 0.9))
    affinity = blob((60, 120), 30, 45, 8, 0.8)  # a link reaching both characters
    ((box, score),) = word_boxes(region, affinity)
    assert score == pytest.approx(0.9, abs=0.01)
    assert box[0] < 22 and box[2] > 68 and box[1] < 24 and box[3] > 36  # both characters, grown


def test_unlinked_characters_are_separate_words() -> None:
    region = np.maximum(blob((60, 160), 30, 30, 5, 0.9), blob((60, 160), 30, 120, 5, 0.9))
    boxes = word_boxes(region, np.zeros_like(region))
    assert len(boxes) == 2


def test_faint_or_tiny_components_are_not_words() -> None:
    faint = blob((60, 60), 30, 30, 6, 0.6)  # above low_text, never above text_threshold
    assert word_boxes(faint, np.zeros_like(faint)) == []
    speck = np.zeros((60, 60), dtype=np.float32)
    speck[10:12, 10:12] = 0.95  # 4 pixels
    assert word_boxes(speck, np.zeros_like(speck)) == []


class FixedModel(torch.nn.Module):
    """Stands in for the network: the same half-resolution maps for any input."""

    def __init__(self, maps: torch.Tensor) -> None:
        super().__init__()
        self.maps = maps
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x)
        return self.maps[None]


def test_detect_returns_boxes_in_tile_pixels() -> None:
    # a 100x150 tile is padded to 128x160; its maps are 64x80, of which 50x75 cover the tile
    region = torch.from_numpy(blob((64, 80), 20, 30, 4, 0.95))
    model = FixedModel(torch.stack([region, torch.zeros_like(region)]))
    detector = CraftDetector(model, torch.device("cpu"))  # type: ignore[arg-type]
    tile = torch.full((3, 100, 150), 255, dtype=torch.uint8)
    ((box, score),) = detector.detect([tile])[0]
    assert model.inputs[0].shape == (1, 3, 128, 160)
    assert score == pytest.approx(0.95, abs=0.01)
    centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    assert centre == pytest.approx((60.0, 40.0), abs=2.0)  # (30, 20) at half resolution
    assert all(0.0 <= v <= 150.0 for v in (box[0], box[2])) and all(
        0.0 <= v <= 100.0 for v in (box[1], box[3])
    )


def craft_entry(archive: bytes, *, name: str = CRAFT_MODEL_ID) -> ModelEntry:
    return ModelEntry(
        id=name,
        name="CRAFT",
        kind="vision",
        format="file",
        size_mb=1,
        license="MIT",
        description="test",
        mirror_url="https://example.invalid/craft.zip",
        sha256=hashlib.sha256(archive).hexdigest(),
        bytes=len(archive),
        upstream_url="https://example.invalid/upstream.zip",
        install_path="craft/craft_mlt_25k.zip",
    )


def zipped_state(state: dict[str, torch.Tensor], member: str = "craft_mlt_25k.pth") -> bytes:
    weights, archive = io.BytesIO(), io.BytesIO()
    torch.save({f"module.{k}": v for k, v in state.items()}, weights)
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(member, weights.getvalue())
    return archive.getvalue()


def test_load_reads_the_installed_zip(tmp_path: Path) -> None:
    state = Craft().state_dict()
    archive = zipped_state(state)
    (tmp_path / "craft").mkdir()
    (tmp_path / "craft" / "craft_mlt_25k.zip").write_bytes(archive)
    detector = CraftDetector.load(torch.device("cpu"), tmp_path, catalog=[craft_entry(archive)])
    loaded = detector.model.state_dict()
    assert torch.equal(loaded["conv_cls.8.weight"], state["conv_cls.8.weight"])
    assert not detector.model.training


def test_load_downloads_the_zip_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = zipped_state(Craft().state_dict())
    entry = craft_entry(archive)
    fetched: list[str] = []

    def fake_download(model: ModelEntry, models_dir: Path) -> str:
        fetched.append(model.id)
        (models_dir / "craft").mkdir(parents=True)
        (models_dir / "craft" / "craft_mlt_25k.zip").write_bytes(archive)
        return "upstream"

    monkeypatch.setattr(craft, "download_model", fake_download)
    CraftDetector.load(torch.device("cpu"), tmp_path, catalog=[entry])
    assert fetched == [CRAFT_MODEL_ID]


def test_load_refuses_a_missing_entry_or_weights(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no 'sfx-detector-craft' entry"):
        CraftDetector.load(torch.device("cpu"), tmp_path, catalog=[])
    archive = zipped_state({"x": torch.zeros(1)}, member="other.pth")
    (tmp_path / "craft").mkdir()
    (tmp_path / "craft" / "craft_mlt_25k.zip").write_bytes(archive)
    with pytest.raises(ValueError, match=r"holds no craft_mlt_25k\.pth"):
        CraftDetector.load(torch.device("cpu"), tmp_path, catalog=[craft_entry(archive)])

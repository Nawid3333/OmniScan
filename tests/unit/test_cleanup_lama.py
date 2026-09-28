"""Tests for LaMa on a brush selection: the "lama" clean method, the service that loads the model for it and the
web route (torch-free: the model is a stand-in; tests/unit/test_cleanup_lama_model.py runs the real window logic)."""

from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import omniscan.cleanup.lama_now as lama_now
from omniscan.cleanup import store
from omniscan.core.config import Config, PathsConfig
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import BBox, CleanupPatch, IngestArtifact, SourceFile
from omniscan.web.app import create_app

GREY, INK, DRAWN = (120, 120, 120), (10, 10, 10), (1, 2, 3)
BASE = "/api/series/S/chapters/Chapter%201"


def box(x0: int, y0: int, x1: int, y1: int) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


@dataclass
class Painter:
    """A stand-in model: paints the masked pixels DRAWN and records what it was shown."""

    context_px: int = 10
    seen: list[tuple[tuple[int, ...], int]] = field(default_factory=list)

    def __call__(self, context: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.seen.append((context.shape, int(mask.sum())))
        out = context.copy()
        out[mask] = DRAWN
        return out


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """One 200x100 grey page with a black mark at (50..70, 40..60)."""
    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )
    paths = SeriesPaths.from_config(cfg, "S").chapter("Chapter 1")
    paths.raw_dir.mkdir(parents=True)
    page = np.full((100, 200, 3), GREY, dtype=np.uint8)
    page[40:60, 50:70] = INK
    Image.fromarray(page).save(paths.raw_dir / "001.png")
    IngestArtifact(
        series="S",
        chapter="Chapter 1",
        strip_width=200,
        strip_height=100,
        files=[SourceFile(index=0, name="001.png", sha256="0" * 64, width=200, height=100, y0=0, y1=100)],
    ).save(paths.artifact("ingest.json"))
    return cfg


def chapter(cfg: Config):
    return SeriesPaths.from_config(cfg, "S").chapter("Chapter 1")


def test_the_lama_method_stores_the_models_rebuild_of_the_stroke(cfg: Config) -> None:
    paths = chapter(cfg)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    painter = Painter()
    patch = store.add_patch(paths, page=0, box=box(50, 40, 70, 60), mask=mask, method="lama", lama=painter)
    assert (patch.method, patch.box, patch.mask_px) == ("lama", box(50, 40, 70, 60), 100)
    assert painter.seen == [((40, 40, 3), 100)]  # the stroke plus the model's context on every side
    rgba = store.patch_rgba(paths, patch.id)
    assert tuple(rgba[10, 10, :3]) == DRAWN and rgba[10, 10, 3] == 255
    assert rgba[0, 0, 3] == 0  # outside the stroke: not applied
    with pytest.raises(ValueError, match="needs the LaMa model"):
        store.add_patch(paths, page=0, box=box(50, 40, 70, 60), mask=mask, method="lama")


def test_the_model_loads_only_for_a_stroke_that_checks_out(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[Config] = []

    @contextmanager
    def fake_model(given: Config) -> Iterator[Painter]:
        loads.append(given)
        yield Painter()

    monkeypatch.setattr(lama_now, "lama_model", fake_model)
    paths = chapter(cfg)
    with pytest.raises(ValueError, match="covers no pixel"):
        lama_now.clean_with_lama(paths, cfg, page=0, box=box(0, 0, 4, 4), mask=np.zeros((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="no page with index 3"):
        lama_now.clean_with_lama(paths, cfg, page=3, box=box(0, 0, 4, 4), mask=np.ones((4, 4), dtype=bool))
    assert loads == []  # a 10-25 s model load is never spent on a stroke that is refused anyway
    patch = lama_now.clean_with_lama(
        paths, cfg, page=0, box=box(50, 40, 70, 60), mask=np.ones((20, 20), dtype=bool)
    )
    assert patch.method == "lama" and loads == [cfg]
    assert [p.id for p in store.load_cleanup(paths).patches] == [patch.id]  # type: ignore[union-attr]


def png_mask(width: int, height: int) -> str:
    buffer = io.BytesIO()
    Image.new("RGBA", (width, height), (255, 255, 255, 255)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def test_the_web_route_cleans_with_lama_and_says_when_it_cannot(cfg: Config) -> None:
    calls: list[tuple[int, BBox, int]] = []
    failure: list[Exception] = []

    def cleaner(paths, given: Config, *, page: int, box: BBox, mask: np.ndarray) -> CleanupPatch:
        if failure:
            raise failure[0]
        calls.append((page, box, int(mask.sum())))
        return store.add_patch(paths, page=page, box=box, mask=mask, method="lama", lama=Painter())

    client = TestClient(create_app(cfg, lama_cleaner=cleaner))
    body = {
        "page": 0,
        "box": {"x0": 50, "y0": 40, "x1": 70, "y1": 60},
        "mask": png_mask(20, 20),
        "method": "lama",
    }
    response = client.post(f"{BASE}/cleanup", json=body)
    assert response.status_code == 201 and response.json()["method"] == "lama"
    assert calls == [(0, box(50, 40, 70, 60), 400)]
    for exc in (RuntimeError("LaMa download failed: HTTP 404"), ImportError("No module named 'torch'")):
        failure[:] = [exc]
        refused = client.post(f"{BASE}/cleanup", json=body)
        assert refused.status_code == 503 and "LaMa is not available" in refused.json()["detail"]
    failure[:] = [ValueError("the stroke covers no pixel of the page")]
    assert client.post(f"{BASE}/cleanup", json=body).status_code == 422
    assert client.post(f"{BASE}/cleanup", json={**body, "mask": png_mask(5, 5)}).status_code == 422

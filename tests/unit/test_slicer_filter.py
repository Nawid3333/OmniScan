"""Tests for the slice stage's promo-filter wiring (card F2a).

The strip is injected via a `load_strip` monkeypatch so the slices are fully controlled: the
`slicer.strategy = "page"` config gives one slice per SourceFile, and the middle "banner" page is a
left-to-right falling luminance ramp — the same shape as the promo example, whose dHash is pinned to
2**64-1 by the golden tests, so banner and example match with similarity 1.0 on every run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from omniscan.core.config import Config, FilterConfig, GpuConfig, PathsConfig, SlicerConfig
from omniscan.core.schemas import IngestArtifact, SlicesArtifact, SourceFile
from omniscan.core.stage import ChapterContext, make_context, run_stage
from omniscan.filter.apply import record_override
from omniscan.slicer.stage import SliceStage
from tests.fixtures import images
from tests.fixtures.strips import art, stack

SERIES = "S"
CHAPTER = "Chapter 1"
WIDTH = 400
ART_H = 200
BANNER_H = 300


def _banner(height: int, width: int) -> torch.Tensor:
    """Falling left-to-right luminance ramp (dHash all-ones on both the PIL and tensor paths)."""
    ramp = torch.tensor([(width - 1 - x) * 255 // (width - 1) for x in range(width)], dtype=torch.uint8)
    return ramp.view(1, 1, width).expand(3, height, width).contiguous()


STRIP = stack(art(ART_H, WIDTH, 11), _banner(BANNER_H, WIDTH), art(ART_H, WIDTH, 13))


def stage_cfg(tmp_path: Path, enabled: bool = True) -> Config:
    return Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
        filter=FilterConfig(enabled=enabled),
        slicer=SlicerConfig(strategy="page"),
    )


def write_chapter(cfg: Config) -> ChapterContext:
    """Raw dir with three pages, a matching promo example, and a hand-built ingest.json."""
    raw = cfg.paths.library_root / SERIES / CHAPTER
    raw.mkdir(parents=True)
    for i in range(3):
        images.plain_jpeg(raw / f"{i + 1:03d}.jpg")
    ctx = make_context(cfg, SERIES, CHAPTER)
    files = [
        SourceFile(index=0, name="001.jpg", sha256="a" * 64, width=WIDTH, height=ART_H, y0=0, y1=ART_H),
        SourceFile(
            index=1,
            name="002.jpg",
            sha256="b" * 64,
            width=WIDTH,
            height=BANNER_H,
            y0=ART_H,
            y1=ART_H + BANNER_H,
        ),
        SourceFile(
            index=2,
            name="003.jpg",
            sha256="c" * 64,
            width=WIDTH,
            height=ART_H,
            y0=ART_H + BANNER_H,
            y1=ART_H + BANNER_H + ART_H,
        ),
    ]
    IngestArtifact(
        series=SERIES, chapter=CHAPTER, strip_width=WIDTH, strip_height=ART_H * 2 + BANNER_H, files=files
    ).save(ctx.paths.artifact("ingest.json"))
    return ctx


def add_example(cfg: Config) -> None:
    cfg.paths.promo_examples.joinpath("global").mkdir(parents=True, exist_ok=True)
    images.gradient_jpeg(cfg.paths.promo_examples / "global" / "end_card.jpg", invert=True)


@pytest.fixture
def patched_strip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("omniscan.slicer.stage.load_strip", lambda ctx, ingest: STRIP)


def filtered_flags(path: Path) -> list[bool]:
    return [s.filtered for s in SlicesArtifact.load(path).slices]


def forbid_dhash_tensor(monkeypatch: pytest.MonkeyPatch, why: str) -> None:
    def spy(image: torch.Tensor, hash_size: int = 8) -> int:
        raise AssertionError(why)

    monkeypatch.setattr("omniscan.filter.apply.dhash_tensor", spy)


def test_slice_stage_flags_the_promo_slice(tmp_path: Path, patched_strip: None) -> None:
    cfg = stage_cfg(tmp_path)
    add_example(cfg)
    ctx = write_chapter(cfg)

    outcome = run_stage(SliceStage(), ctx)
    assert outcome.status == "done"
    assert outcome.metrics["filtered"] == 1.0
    assert ctx.paths.artifact("slices.json").is_file()
    assert filtered_flags(ctx.paths.artifact("slices.json")) == [False, True, False]
    saved = ctx.paths.filtered_dir / f"{CHAPTER}_slice_0001.jpg"
    assert saved.is_file()
    with Image.open(saved) as img:
        assert img.size == (WIDTH, BANNER_H)

    assert run_stage(SliceStage(), make_context(cfg, SERIES, CHAPTER)).status == "skipped"


def test_slice_stage_reruns_when_an_example_is_added(tmp_path: Path, patched_strip: None) -> None:
    cfg = stage_cfg(tmp_path)
    ctx = write_chapter(cfg)

    assert run_stage(SliceStage(), ctx).status == "done"
    assert filtered_flags(ctx.paths.artifact("slices.json")) == [False, False, False]
    assert run_stage(SliceStage(), make_context(cfg, SERIES, CHAPTER)).status == "skipped"

    add_example(cfg)
    assert run_stage(SliceStage(), make_context(cfg, SERIES, CHAPTER)).status == "done"
    assert filtered_flags(ctx.paths.artifact("slices.json")) == [False, True, False]
    assert run_stage(SliceStage(), make_context(cfg, SERIES, CHAPTER)).status == "skipped"


def test_slice_stage_restored_and_forced_overrides(tmp_path: Path, patched_strip: None) -> None:
    cfg = stage_cfg(tmp_path)
    add_example(cfg)
    ctx = write_chapter(cfg)
    assert run_stage(SliceStage(), ctx).status == "done"

    record_override(ctx.paths, "slice", 1, "restored")
    record_override(ctx.paths, "slice", 2, "filtered")  # a non-matching page, forced by hand
    assert run_stage(SliceStage(), make_context(cfg, SERIES, CHAPTER)).status == "done"
    assert filtered_flags(ctx.paths.artifact("slices.json")) == [False, False, True]


def test_slice_stage_without_examples_never_hashes_slices(
    tmp_path: Path, patched_strip: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = stage_cfg(tmp_path)
    ctx = write_chapter(cfg)
    forbid_dhash_tensor(monkeypatch, "dhash_tensor must not run without examples or overrides")

    outcome = run_stage(SliceStage(), ctx)
    assert outcome.status == "done"
    assert outcome.metrics["filtered"] == 0.0


def test_slice_stage_filter_disabled_never_hashes_slices(
    tmp_path: Path, patched_strip: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = stage_cfg(tmp_path, enabled=False)
    add_example(cfg)
    ctx = write_chapter(cfg)
    forbid_dhash_tensor(monkeypatch, "dhash_tensor must not run while the filter is disabled")

    outcome = run_stage(SliceStage(), ctx)
    assert outcome.status == "done"
    assert filtered_flags(ctx.paths.artifact("slices.json")) == [False, False, False]

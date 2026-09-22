"""Tests for ingest_chapter's file-level promo filter and the IngestStage wiring (card F2a)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from omniscan.core.config import Config, FilterConfig, GpuConfig, PathsConfig
from omniscan.core.schemas import IngestArtifact
from omniscan.core.stage import make_context, run_stage
from omniscan.filter.apply import record_override
from omniscan.filter.decide import load_examples
from omniscan.filter.hashing import dhash, hamming, similarity
from omniscan.gpu.codec.turbo import TurboCodec
from omniscan.ingest import ingest_chapter
from omniscan.ingest.stage import IngestStage
from omniscan.ingest.strip import build_strip, jpeg_paths
from tests.fixtures import images

SERIES = "S"
CHAPTER = "Chapter 1"
COLORS = [(200, 60, 60), (60, 200, 120), (80, 100, 220)]


def write_chapter(raw: Path, promo: Path) -> Path:
    """5 pages: 001/003/005 solid page colours, 002/004 near-copies of one promo example."""
    raw.mkdir(parents=True, exist_ok=True)
    promo.joinpath("global").mkdir(parents=True, exist_ok=True)
    example = images.gradient_jpeg(promo / "global" / "end_card.jpg", invert=True)
    for i, color in enumerate(COLORS):
        images.plain_jpeg(raw / f"{2 * i + 1:03d}.jpg", color=color)
    with Image.open(example) as img:
        resized = img.convert("RGB").resize((384, 290))
    rng = np.random.default_rng(0)
    for i in (2, 4):  # page 2 and page 4 (0-based raw indices 1 and 3)
        pixels = np.asarray(resized, dtype=np.int16) + rng.integers(-3, 4, (290, 384, 3), dtype=np.int16)
        Image.fromarray(pixels.clip(0, 255).astype(np.uint8)).save(raw / f"{i:03d}.jpg", format="JPEG")
    return example


def near_copy_similarity(promo: Path, raw: Path) -> float:
    with Image.open(promo / "global" / "end_card.jpg") as example, Image.open(raw / "002.jpg") as page:
        return similarity(dhash(example), dhash(page))


def test_near_copies_really_match_the_example(tmp_path: Path) -> None:
    write_chapter(tmp_path / "raw", tmp_path / "promo")
    assert near_copy_similarity(tmp_path / "promo", tmp_path / "raw") >= 0.90


def test_ingest_filters_promo_files(tmp_path: Path) -> None:
    raw, promo = tmp_path / "raw", tmp_path / "promo"
    write_chapter(raw, promo)
    examples = load_examples(promo, SERIES)

    result = ingest_chapter(
        raw,
        SERIES,
        CHAPTER,
        tmp_path / "cache",
        examples=examples,
        filtered_dir=tmp_path / "out" / "_filtered",
    )
    art = result.artifact
    assert art.filtered_files == ["002.jpg", "004.jpg"]
    assert [f.index for f in art.files] == [0, 2, 4]  # kept files keep their original raw index
    assert [f.name for f in art.files] == ["001.jpg", "003.jpg", "005.jpg"]
    assert art.strip_height == 900  # the sum of the three kept pages
    assert art.strip_width == 400
    assert len(result.converted_paths) == 3
    for name in ("002.jpg", "004.jpg"):
        copied = tmp_path / "out" / "_filtered" / name
        assert copied.is_file()
        assert copied.read_bytes() == (raw / name).read_bytes()  # byte-identical copy

    strip = build_strip(art, jpeg_paths(art, raw, tmp_path / "cache"), TurboCodec(device="cpu"))
    assert strip.shape == (3, 900, 400)
    for file, color in zip(art.files, COLORS, strict=True):
        mean = strip[:, file.y0 : file.y1, :].float().mean(dim=(1, 2)).tolist()
        assert all(abs(m - c) <= 3 for m, c in zip(mean, color, strict=True))


def test_ingest_cache_names_use_raw_indices(tmp_path: Path) -> None:
    raw, promo = tmp_path / "raw", tmp_path / "promo"
    write_chapter(raw, promo)
    (raw / "001.jpg").unlink()
    images.png_with_alpha(raw / "000.png", size=(400, 300))  # needs conversion: cache name shows the index
    result = ingest_chapter(
        raw,
        SERIES,
        CHAPTER,
        tmp_path / "cache",
        examples=load_examples(promo, SERIES),
    )
    converted = [p for p in result.converted_paths if p.parent == tmp_path / "cache"]
    assert [p.name for p in converted] == ["0000_000.jpg"]  # raw index 0, not the position among kept files
    assert result.artifact.files[0].index == 0


def test_ingest_all_filtered_raises(tmp_path: Path) -> None:
    raw, promo = tmp_path / "raw", tmp_path / "promo"
    write_chapter(raw, promo)
    for path in raw.iterdir():
        if path.name not in ("002.jpg", "004.jpg"):
            path.unlink()
    with pytest.raises(ValueError, match="every image"):
        ingest_chapter(raw, SERIES, CHAPTER, tmp_path / "cache", examples=load_examples(promo, SERIES))


def test_ingest_restore_override_keeps_a_matching_file(tmp_path: Path) -> None:
    raw, promo = tmp_path / "raw", tmp_path / "promo"
    write_chapter(raw, promo)
    from omniscan.filter.apply import Overrides

    restored = Overrides(
        files_restored=frozenset({1}),
        files_forced=frozenset(),
        slices_restored=frozenset(),
        slices_forced=frozenset(),
    )
    result = ingest_chapter(
        raw,
        SERIES,
        CHAPTER,
        tmp_path / "cache",
        examples=load_examples(promo, SERIES),
        overrides=restored,
        filtered_dir=tmp_path / "out" / "_filtered",
    )
    assert [f.index for f in result.artifact.files] == [0, 1, 2, 4]
    assert result.artifact.filtered_files == ["004.jpg"]


def test_ingest_force_override_drops_without_examples(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_chapter(raw, tmp_path / "promo")  # examples exist but are not passed
    from omniscan.filter.apply import Overrides

    forced = Overrides(frozenset(), frozenset({0}), frozenset(), frozenset())
    result = ingest_chapter(raw, SERIES, CHAPTER, tmp_path / "cache", overrides=forced)
    assert [f.index for f in result.artifact.files] == [1, 2, 3, 4]
    assert result.artifact.filtered_files == ["001.jpg"]
    assert not (tmp_path / "out").exists()  # no filtered_dir given: nothing copied


def test_ingest_without_examples_matches_pre_change_artifact(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_chapter(raw, tmp_path / "promo")
    result = ingest_chapter(raw, SERIES, CHAPTER, tmp_path / "cache")
    art = result.artifact
    assert art.filtered_files == []
    assert [f.index for f in art.files] == [0, 1, 2, 3, 4]
    # the two 384-wide near-copies are kept too and resize to round(290*400/384)=302 rows each
    assert art.strip_height == 3 * 300 + 2 * 302
    dest = tmp_path / "ingest.json"
    art.save(dest)
    assert IngestArtifact.load(dest) == art

    strip = build_strip(art, jpeg_paths(art, raw, tmp_path / "cache"), TurboCodec(device="cpu"))
    assert strip.shape == (3, art.strip_height, 400)


def test_ingest_dhash_similarity_of_noise_pages_stays_low(tmp_path: Path) -> None:
    """Guard for the fixture: a noise page must not accidentally match the promo example."""
    raw, promo = tmp_path / "raw", tmp_path / "promo"
    write_chapter(raw, promo)
    with Image.open(promo / "global" / "end_card.jpg") as example:
        example_hash = dhash(example)
    rng = np.random.default_rng(5)
    for _ in range(10):
        noise = Image.fromarray(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8))
        assert hamming(example_hash, dhash(noise)) > 6
        assert similarity(example_hash, dhash(noise)) < 0.90


# ---------------------------------------------------------------- IngestStage wiring


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
    )


def test_ingest_stage_filters_and_reports_metrics(tmp_path: Path) -> None:
    cfg = stage_cfg(tmp_path)
    raw = cfg.paths.library_root / SERIES / CHAPTER
    write_chapter(raw, cfg.paths.promo_examples)

    ctx = make_context(cfg, SERIES, CHAPTER)
    outcome = run_stage(IngestStage(), ctx)
    assert outcome.status == "done"
    assert outcome.metrics["filtered_files"] == 2.0
    art = IngestArtifact.load(ctx.paths.artifact("ingest.json"))
    assert art.filtered_files == ["002.jpg", "004.jpg"]
    for name in art.filtered_files:
        assert (ctx.paths.filtered_dir / name).read_bytes() == (raw / name).read_bytes()

    assert run_stage(IngestStage(), make_context(cfg, SERIES, CHAPTER)).status == "skipped"


def test_ingest_stage_reruns_when_an_example_is_added(tmp_path: Path) -> None:
    cfg = stage_cfg(tmp_path)
    raw = cfg.paths.library_root / SERIES / CHAPTER
    raw.mkdir(parents=True)
    example = images.gradient_jpeg(tmp_path / "end_card.jpg", invert=True)
    with Image.open(example) as img:
        resized = img.convert("RGB").resize((384, 290))
    rng = np.random.default_rng(0)
    pixels = np.asarray(resized, dtype=np.int16) + rng.integers(-3, 4, (290, 384, 3), dtype=np.int16)
    Image.fromarray(pixels.clip(0, 255).astype(np.uint8)).save(raw / "002.jpg", format="JPEG")
    images.plain_jpeg(raw / "001.jpg", color=(200, 60, 60))

    ctx = make_context(cfg, SERIES, CHAPTER)
    assert run_stage(IngestStage(), ctx).status == "done"
    assert IngestArtifact.load(ctx.paths.artifact("ingest.json")).filtered_files == []
    assert run_stage(IngestStage(), make_context(cfg, SERIES, CHAPTER)).status == "skipped"

    cfg.paths.promo_examples.joinpath("global").mkdir(parents=True)
    shutil.copy2(example, cfg.paths.promo_examples / "global" / "end_card.jpg")
    assert run_stage(IngestStage(), make_context(cfg, SERIES, CHAPTER)).status == "done"
    assert IngestArtifact.load(ctx.paths.artifact("ingest.json")).filtered_files == ["002.jpg"]
    assert run_stage(IngestStage(), make_context(cfg, SERIES, CHAPTER)).status == "skipped"


def test_ingest_stage_reruns_when_filter_json_changes(tmp_path: Path) -> None:
    cfg = stage_cfg(tmp_path)
    raw = cfg.paths.library_root / SERIES / CHAPTER
    write_chapter(raw, cfg.paths.promo_examples)

    ctx = make_context(cfg, SERIES, CHAPTER)
    assert run_stage(IngestStage(), ctx).status == "done"
    assert run_stage(IngestStage(), make_context(cfg, SERIES, CHAPTER)).status == "skipped"

    record_override(ctx.paths, "file", 1, "restored")  # 002.jpg is raw index 1
    assert run_stage(IngestStage(), make_context(cfg, SERIES, CHAPTER)).status == "done"
    art = IngestArtifact.load(ctx.paths.artifact("ingest.json"))
    assert [f.index for f in art.files] == [0, 1, 2, 4]
    assert art.filtered_files == ["004.jpg"]
    assert run_stage(IngestStage(), make_context(cfg, SERIES, CHAPTER)).status == "skipped"


def test_ingest_stage_filter_disabled_ignores_examples(tmp_path: Path) -> None:
    cfg = stage_cfg(tmp_path, enabled=False)
    raw = cfg.paths.library_root / SERIES / CHAPTER
    write_chapter(raw, cfg.paths.promo_examples)

    ctx = make_context(cfg, SERIES, CHAPTER)
    outcome = run_stage(IngestStage(), ctx)
    assert outcome.status == "done"
    assert outcome.metrics["filtered_files"] == 0.0
    art = IngestArtifact.load(ctx.paths.artifact("ingest.json"))
    assert art.filtered_files == []
    assert [f.index for f in art.files] == [0, 1, 2, 3, 4]

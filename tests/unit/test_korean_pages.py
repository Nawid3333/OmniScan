"""Tests for the synthetic Korean page generator with exact ground truth (card X1)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.core.schemas import BBox, IngestArtifact, RegionsArtifact
from omniscan.core.stage import make_context, run_stage
from omniscan.ingest.stage import IngestStage
from tests.fixtures import korean_pages
from tests.fixtures.korean_pages import (
    KOREAN_LINES,
    SFX_LINES,
    make_korean_page,
    to_regions_artifact,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_korean_chapter.py"
BACKGROUNDS: list[korean_pages.Background] = ["gradient", "noise", "flat"]


def union(lines: tuple[korean_pages.LineTruth, ...]) -> BBox:
    """Union of line truth boxes."""
    return BBox(
        x0=min(line.bbox.x0 for line in lines),
        y0=min(line.bbox.y0 for line in lines),
        x1=max(line.bbox.x1 for line in lines),
        y1=max(line.bbox.y1 for line in lines),
    )


def boxes_overlap(a: BBox, b: BBox) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


# ---------------------------------------------------------------- 1 determinism


def test_determinism() -> None:
    a = make_korean_page(11, background="noise")
    b = make_korean_page(11, background="noise")
    assert np.array_equal(np.asarray(a.image), np.asarray(b.image))
    assert np.array_equal(np.asarray(a.clean), np.asarray(b.clean))
    assert a.regions == b.regions
    assert np.array_equal(a.text_mask, b.text_mask)

    other = make_korean_page(12, background="noise")
    assert not np.array_equal(np.asarray(a.image), np.asarray(other.image))


# ---------------------------------------------------------------- 2 mask / clean consistency


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_mask_and_clean_consistency(seed: int) -> None:
    page = make_korean_page(seed, n_sfx=1)
    mask = np.any(np.asarray(page.image) != np.asarray(page.clean), axis=-1)
    assert np.array_equal(page.text_mask, mask)

    allowed = np.zeros_like(page.text_mask)
    for region in page.regions:
        inside = page.text_mask[region.bbox.y0 : region.bbox.y1, region.bbox.x0 : region.bbox.x1]
        assert int(inside.sum()) >= 50, f"too little ink in {region.text!r}"
        allowed[
            max(0, region.bbox.y0 - 6) : region.bbox.y1 + 6, max(0, region.bbox.x0 - 6) : region.bbox.x1 + 6
        ] = True
    assert not page.text_mask[~allowed].any()


# ---------------------------------------------------------------- 3 geometry invariants


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("background", BACKGROUNDS)
def test_geometry_invariants(seed: int, background: korean_pages.Background) -> None:
    page = make_korean_page(seed, background=background, n_bubbles=4, n_free=1, n_sfx=1)
    for region in page.regions:
        assert region.bbox.x0 >= 0 and region.bbox.x1 <= 800
        assert region.bbox.y0 >= 40 and region.bbox.y1 <= 1360
        assert region.lines
        for line in region.lines:
            assert region.bbox.x0 <= line.bbox.x0 and line.bbox.x1 <= region.bbox.x1
            assert region.bbox.y0 <= line.bbox.y0 and line.bbox.y1 <= region.bbox.y1
        assert region.bbox == union(region.lines)
        if region.bubble_bbox is not None:
            bubble = region.bubble_bbox
            assert bubble.x0 >= 0 and bubble.x1 <= 800 and bubble.y0 >= 40 and bubble.y1 <= 1360
            assert region.bbox.x0 - bubble.x0 >= 8
            assert region.bbox.y0 - bubble.y0 >= 8
            assert bubble.x1 - region.bbox.x1 >= 8
            assert bubble.y1 - region.bbox.y1 >= 8
        if region.bubble_polygon is not None:
            assert region.bubble_bbox is not None
            assert len(region.bubble_polygon) >= 12
            for x, y in region.bubble_polygon:
                assert region.bubble_bbox.x0 - 1 <= x <= region.bubble_bbox.x1 + 1
                assert region.bubble_bbox.y0 - 1 <= y <= region.bubble_bbox.y1 + 1
    for a, b in pairwise(page.regions):
        a_box = a.bubble_bbox or a.bbox
        b_box = b.bubble_bbox or b.bbox
        assert not boxes_overlap(a_box, b_box)


# ---------------------------------------------------------------- 4 reading order


@pytest.mark.parametrize("seed", range(12))
def test_reading_order(seed: int) -> None:
    page = make_korean_page(seed, n_bubbles=4, n_free=1, n_sfx=1)
    for a, b in pairwise(page.regions):
        a_box, b_box = a.bubble_bbox or a.bbox, b.bubble_bbox or b.bbox
        iy = min(a_box.y1, b_box.y1) - max(a_box.y0, b_box.y0)
        smaller = min(a_box.y1 - a_box.y0, b_box.y1 - b_box.y0)
        if iy < 0.5 * smaller:
            assert (a_box.y0 + a_box.y1) / 2 <= (b_box.y0 + b_box.y1) / 2
        else:
            assert a_box.x0 < b_box.x0


# ---------------------------------------------------------------- 5 counts and kinds


def test_counts_and_kinds() -> None:
    page = make_korean_page(0, n_bubbles=4, n_free=1, n_sfx=1)
    counts = Counter(region.kind for region in page.regions)
    assert counts == {"bubble_text": 4, "free_text": 1, "sfx": 1}


def test_no_regions_means_no_text() -> None:
    page = make_korean_page(3, n_bubbles=0, n_free=0, n_sfx=0)
    assert page.regions == ()
    assert np.array_equal(np.asarray(page.image), np.asarray(page.clean))
    assert not page.text_mask.any()


def test_too_many_regions_raise_value_error() -> None:
    with pytest.raises(ValueError, match=r"30 regions on a 800x1400"):
        make_korean_page(0, n_bubbles=30, n_free=0, n_sfx=0, width=800, height=1400)


# ---------------------------------------------------------------- 6 text content


@pytest.mark.parametrize("seed", [5, 6])
def test_text_content(seed: int) -> None:
    page = make_korean_page(seed, n_bubbles=3, n_free=2, n_sfx=2)
    for region in page.regions:
        if region.kind == "sfx":
            assert "".join(region.text.split()) in ["".join(s.split()) for s in SFX_LINES]
            continue
        flat = "".join(region.text.split())
        sentence = next(s for s in KOREAN_LINES if "".join(s.split()) == flat)
        assert " ".join(line.text for line in region.lines) == sentence


def test_corpus_is_exact() -> None:
    assert len(KOREAN_LINES) == 12
    assert KOREAN_LINES[0] == "괜찮아요? 던전이 열렸어!"
    assert KOREAN_LINES[-1] == "약속할게. 반드시 돌아올 거야."


# ---------------------------------------------------------------- 7 fit


@pytest.mark.parametrize("seed", range(6))
def test_bubble_fit_and_colors(seed: int) -> None:
    page = make_korean_page(seed, n_bubbles=4, n_free=1, n_sfx=1)
    for region in page.regions:
        if region.kind != "bubble_text":
            continue
        assert region.bubble_bbox is not None
        bubble_w = region.bubble_bbox.x1 - region.bubble_bbox.x0
        for line in region.lines:
            assert line.bbox.x1 - line.bbox.x0 <= 0.9 * bubble_w
        if region.fill == (24, 24, 32):
            assert region.text_color == (255, 255, 255)
        else:
            assert region.fill == (255, 255, 255)
            assert region.text_color == (0, 0, 0)


# ---------------------------------------------------------------- 8 fonts


@pytest.mark.parametrize("seed", [2, 9])
def test_hangul_is_really_rendered(seed: int) -> None:
    page = make_korean_page(seed, n_sfx=1)
    for region in page.regions:
        for line in region.lines:
            chars = [c for c in line.text if not c.isspace()]
            ink = int(page.text_mask[line.bbox.y0 : line.bbox.y1, line.bbox.x0 : line.bbox.x1].sum())
            assert ink >= 20 * len(chars)


def test_notdef_glyph_differs_from_hangul() -> None:
    path = str(korean_pages.font_path("NanumGothic-Regular.ttf"))

    def render(text: str) -> bytes:
        font = ImageFont.truetype(path, 40)
        img = Image.new("L", (80, 80), 255)
        ImageDraw.Draw(img).text((40, 40), text, font=font, anchor="mm", fill=0)
        return img.tobytes()

    assert render("가") != render("\U0010fffd")


def test_font_path_missing() -> None:
    with pytest.raises(FileNotFoundError, match=r"font not found: nope\.ttf"):
        korean_pages.font_path("nope.ttf")


@pytest.mark.parametrize("font", ["Gaegu-Regular.ttf", "NanumGothic-Bold.ttf"])
def test_other_fonts_work(font: str) -> None:
    page = make_korean_page(3, font=font, n_bubbles=2, n_free=1)
    assert len(page.regions) == 3
    assert all(region.text for region in page.regions)


# ---------------------------------------------------------------- 9 artifact round-trip


def test_to_regions_artifact() -> None:
    page = make_korean_page(4, n_bubbles=3, n_free=1, n_sfx=1)
    artifact = to_regions_artifact(page)
    RegionsArtifact.model_validate_json(artifact.model_dump_json())  # round-trip
    assert [r.id for r in artifact.regions] == [f"r{i + 1:04d}" for i in range(len(page.regions))]
    assert [r.reading_order for r in artifact.regions] == list(range(len(page.regions)))
    for region, truth in zip(artifact.regions, page.regions, strict=True):
        assert region.slice_index == 0
        assert region.kind == truth.kind
        assert region.lang == "ko"
        assert region.orientation == "h"
        assert region.text == truth.text
        assert region.confidence == 1.0
        assert region.text_color == truth.text_color
        assert region.bbox == truth.bbox
        assert region.bubble_bbox == truth.bubble_bbox
        assert region.polygon == (list(truth.bubble_polygon) if truth.bubble_polygon else None)
        assert [line.text for line in region.lines] == [line.text for line in truth.lines]
        assert [line.bbox for line in region.lines] == [line.bbox for line in truth.lines]
        assert all(line.score == 1.0 and line.engine == "truth" for line in region.lines)


# ---------------------------------------------------------------- 10 script


def test_script_writes_chapter_and_truth(tmp_path: Path) -> None:
    library = tmp_path / "library"
    truth_out = tmp_path / "truth.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--library-root",
            str(library),
            "--pages",
            "3",
            "--seed",
            "7",
            "--truth-out",
            str(truth_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    chapter = library / "KoreanDemo" / "Chapter 1"
    pages = sorted(p.name for p in chapter.iterdir())
    assert pages == ["001.jpg", "002.jpg", "003.jpg"]

    for i, name in enumerate(pages, start=1):
        decoded = Image.open(chapter / name)
        assert decoded.size == (800, 1400)
        expected = np.asarray(
            make_korean_page(7 + i, n_bubbles=4, n_free=1, n_sfx=1 if i % 2 else 0).image, dtype=np.float64
        )
        actual = np.asarray(decoded.convert("RGB"), dtype=np.float64)
        assert float(np.abs(actual - expected).mean()) < 5, f"page {name} too far from the generator"

    truths = json.loads(truth_out.read_text(encoding="utf-8"))
    assert isinstance(truths, list) and len(truths) == 3
    for truth in truths:
        RegionsArtifact.model_validate_json(json.dumps(truth))

    library2 = tmp_path / "library2"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--library-root", str(library2), "--pages", "1", "--seed", "7"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert sorted(p.name for p in (library2 / "KoreanDemo" / "Chapter 1").iterdir()) == ["001.jpg"]


# ---------------------------------------------------------------- 11 pipeline smoke


def test_ingest_stage_on_korean_chapter(tmp_path: Path) -> None:
    cfg = Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
    )
    raw = cfg.paths.library_root / "KoreanDemo" / "Chapter 1"
    raw.mkdir(parents=True)
    for i in range(1, 4):
        page = make_korean_page(7 + i, n_bubbles=4, n_free=1, n_sfx=1 if i % 2 == 1 else 0)
        page.image.save(raw / f"{i:03d}.jpg", format="JPEG", quality=92)

    ctx = make_context(cfg, "KoreanDemo", "Chapter 1")
    assert run_stage(IngestStage(), ctx).status == "done"
    ingest = IngestArtifact.load(ctx.paths.artifact("ingest.json"))
    assert ingest.strip_width == 800
    assert ingest.strip_height == 3 * 1400
    assert not any(f.filtered for f in ingest.files)

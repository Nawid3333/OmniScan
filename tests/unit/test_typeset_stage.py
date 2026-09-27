"""Typeset stage wiring: artifacts, resumability, invalidation, failures and metrics (C7b)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.core.config import Config, GpuConfig, PathsConfig, TypesetConfig
from omniscan.core.schemas import (
    BBox,
    FinalArtifact,
    FinalLine,
    InpaintArtifact,
    InpaintItem,
    LayoutArtifact,
    Region,
    RegionsArtifact,
)
from omniscan.core.stage import make_context, run_stage
from omniscan.typeset.stage import TypesetStage

SERIES = "S"
CHAPTER = "Chapter 1"

BUBBLE = BBox(x0=0, y0=0, x1=400, y1=200)

REGIONS = [
    Region(
        id="r0001",
        slice_index=0,
        kind="bubble_text",
        bbox=BBox(x0=150, y0=80, x1=250, y1=120),
        bubble_bbox=BUBBLE,
        text="괜찮아? 던전이 열렸어!",
    ),
    Region(
        id="r0002",
        slice_index=0,
        kind="free_text",
        bbox=BBox(x0=2, y0=2, x1=52, y1=32),
        text="허공에",
    ),
    Region(id="r0003", slice_index=1, kind="sfx", bbox=BBox(x0=0, y0=300, x1=40, y1=320), text="쾅쾅쾅"),
    Region(
        id="r0004",
        slice_index=1,
        kind="watermark",
        bbox=BBox(x0=0, y0=400, x1=200, y1=420),
        text="omniscan",
    ),
]

FINAL_LINES = {
    "r0001": "Are you okay? The dungeon just opened!",
    "r0002": "   ",  # whitespace-only: counted as skipped
    "r0003": "KRAKATHOOM KRAKATHOOM KRAKATHOOM",  # too long for its tiny box: overflow
}


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


def write_inputs(cfg: Config, chapter: str = CHAPTER, final_lines: dict[str, str] | None = None) -> None:
    """Hand-write the three upstream artifacts into the chapter work dir."""
    paths = cfg.paths.work_root / SERIES / chapter
    paths.mkdir(parents=True, exist_ok=True)
    RegionsArtifact(regions=REGIONS).save(paths / "ocr.json")
    lines = FINAL_LINES if final_lines is None else final_lines
    FinalArtifact(
        judge_model="judge-fake",
        lines=[
            FinalLine(region_id=region_id, text=text, decision="pick") for region_id, text in lines.items()
        ],
    ).save(paths / "final.json")
    InpaintArtifact(
        items=[
            InpaintItem(
                region_id="r0001",
                box=BBox(x0=0, y0=0, x1=400, y1=200),
                method="flat",
                fill=(255, 255, 255),
            ),
            InpaintItem(region_id="r0002", box=BBox(x0=0, y0=0, x1=60, y1=40), method="none"),
            InpaintItem(region_id="r0003", box=BBox(x0=0, y0=300, x1=60, y1=330), method="flat"),
            InpaintItem(
                region_id="r0004",
                box=BBox(x0=0, y0=400, x1=200, y1=420),
                method="flat",
                fill=(10, 10, 10),
            ),
        ]
    ).save(paths / "inpaint.json")


def test_typeset_stage_writes_loadable_layout(cfg: Config) -> None:
    write_inputs(cfg)
    ctx = make_context(cfg, SERIES, CHAPTER)

    outcome = run_stage(TypesetStage(), ctx)

    assert outcome.status == "done"
    artifact = LayoutArtifact.load(ctx.paths.artifact("layout.json"))
    assert [item.region_id for item in artifact.items] == ["r0001", "r0003"]
    assert [item.font_role for item in artifact.items] == ["dialogue", "sfx"]
    # webtoon preset dialogue; the sfx weight is unknown, so the "bold" effect face
    assert [item.font for item in artifact.items] == ["Mali-SemiBold.ttf", "Knewave-Regular.ttf"]
    dialogue = artifact.items[0]
    assert dialogue.color == (0, 0, 0)  # white fill -> black text
    assert dialogue.stroke_px == 0
    assert 14 <= dialogue.size_px <= 48 and dialogue.overflow is False
    sfx = artifact.items[1]
    assert sfx.color == (255, 255, 255)
    assert sfx.stroke_px == cfg.typeset.stroke_sfx_px
    assert sfx.overflow is True  # the long SFX line does not fit its tiny box
    record = ctx.manifest.stages["typeset"]
    assert record.status == "done" and record.outputs == ["layout.json"]
    assert outcome.metrics["items"] == 2.0
    assert outcome.metrics["overflow"] == 1.0
    assert outcome.metrics["skipped"] == 1.0


def test_typeset_stage_subtitles_an_sfx_that_was_not_erased(cfg: Config) -> None:
    write_inputs(cfg)
    work = cfg.paths.work_root / SERIES / CHAPTER
    inpaint = InpaintArtifact.load(work / "inpaint.json")
    InpaintArtifact(items=[item for item in inpaint.items if item.region_id != "r0003"]).save(
        work / "inpaint.json"
    )
    assert run_stage(TypesetStage(), make_context(cfg, SERIES, CHAPTER)).status == "done"
    sfx = LayoutArtifact.load(work / "layout.json").items[1]
    assert sfx.region_id == "r0003" and sfx.font_role == "free"  # a small translation ...
    sfx_region = next(region for region in REGIONS if region.id == "r0003")
    assert sfx.box.y0 >= sfx_region.bbox.y1  # ... below the original, which stays on the page


def test_typeset_stage_reads_what_lama_erased(cfg: Config) -> None:
    write_inputs(cfg)
    work = cfg.paths.work_root / SERIES / CHAPTER
    inpaint = InpaintArtifact.load(work / "inpaint.json")
    InpaintArtifact(items=[item for item in inpaint.items if item.region_id != "r0003"]).save(
        work / "inpaint.json"
    )
    InpaintArtifact(
        items=[InpaintItem(region_id="r0003", box=BBox(x0=0, y0=300, x1=60, y1=330), method="lama")]
    ).save(work / "inpaint_lama.json")
    ctx = make_context(cfg, SERIES, CHAPTER)
    assert work / "inpaint_lama.json" in TypesetStage().inputs(ctx)
    assert run_stage(TypesetStage(), ctx).status == "done"
    assert LayoutArtifact.load(work / "layout.json").items[1].font_role == "sfx"


def test_typeset_stage_is_resumable_and_invalidated(cfg: Config) -> None:
    write_inputs(cfg)
    assert run_stage(TypesetStage(), make_context(cfg, SERIES, CHAPTER)).status == "done"
    assert run_stage(TypesetStage(), make_context(cfg, SERIES, CHAPTER)).status == "skipped"

    smaller = cfg.model_copy(update={"typeset": TypesetConfig(max_px=40)})
    assert run_stage(TypesetStage(), make_context(smaller, SERIES, CHAPTER)).status == "done"

    write_inputs(cfg, final_lines={**FINAL_LINES, "r0001": "Is everybody okay? The gate opened!"})
    assert run_stage(TypesetStage(), make_context(cfg, SERIES, CHAPTER)).status == "done"


def test_typeset_stage_missing_inputs_fail_with_the_documented_message(cfg: Config) -> None:
    work = cfg.paths.work_root / SERIES / CHAPTER
    for artifact, stage in (("ocr.json", "ocr"), ("final.json", "judge"), ("inpaint.json", "inpaint")):
        write_inputs(cfg)
        (work / artifact).unlink()
        outcome = run_stage(TypesetStage(), make_context(cfg, SERIES, CHAPTER))
        assert outcome.status == "failed"
        assert outcome.error is not None
        assert f"{artifact} missing — run the {stage} stage first" in outcome.error


def test_typeset_stage_empty_chapter_yields_zeroes(cfg: Config) -> None:
    paths = cfg.paths.work_root / SERIES / CHAPTER
    paths.mkdir(parents=True, exist_ok=True)
    RegionsArtifact(regions=REGIONS).save(paths / "ocr.json")
    FinalArtifact(judge_model="judge-fake", lines=[]).save(paths / "final.json")
    InpaintArtifact(items=[]).save(paths / "inpaint.json")

    outcome = run_stage(TypesetStage(), make_context(cfg, SERIES, CHAPTER))

    assert outcome.status == "done"
    assert outcome.metrics["items"] == 0.0
    assert outcome.metrics["overflow"] == 0.0
    assert outcome.metrics["skipped"] == 3.0  # every translatable region lacks a final line
    assert LayoutArtifact.load(paths / "layout.json").items == []

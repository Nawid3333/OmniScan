"""Tests for the export stage wiring, CLI and the synthetic Korean chapter (card C7c)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, ExportConfig, GpuConfig, PathsConfig
from omniscan.core.schemas import (
    BBox,
    ExportArtifact,
    IngestArtifact,
    InpaintArtifact,
    InpaintItem,
    LayoutArtifact,
    LayoutItem,
    Manifest,
    Slice,
    SlicesArtifact,
    SourceFile,
)
from omniscan.core.stage import make_context, run_stage
from omniscan.export.stage import ExportStage
from omniscan.inpaint.patches import save_patches
from omniscan.typeset.fit import inscribed_box, layout_region
from tests.fixtures import images
from tests.fixtures.korean_pages import make_korean_page

SERIES = "S"
CHAPTER = "Chapter 1"
PAGE = (800, 600)  # width, height of both raw pages
STRIP_HEIGHT = 1200
PAGE_COLOR = (200, 60, 60)
PATCH_BOX = BBox(x0=100, y0=200, x1=240, y1=280)  # 140 x 80, mask covers a 100 x 40 rectangle
TEXT_BOX = BBox(x0=300, y0=60, x1=520, y1=130)
runner = CliRunner()


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


def write_pages(cfg: Config, chapter: str = CHAPTER) -> None:
    """Two solid-colour raw JPEG pages."""
    raw = cfg.paths.library_root / SERIES / chapter
    raw.mkdir(parents=True, exist_ok=True)
    images.plain_jpeg(raw / "001.jpg", size=PAGE, color=PAGE_COLOR)
    images.plain_jpeg(raw / "002.jpg", size=PAGE, color=(60, 200, 120))


def write_ingest(cfg: Config, chapter: str = CHAPTER) -> None:
    """A hand-written ingest.json for the two-page strip."""
    work = cfg.paths.work_root / SERIES / chapter
    work.mkdir(parents=True, exist_ok=True)
    IngestArtifact(
        series=SERIES,
        chapter=chapter,
        strip_width=PAGE[0],
        strip_height=STRIP_HEIGHT,
        files=[
            SourceFile(
                index=i,
                name=name,
                sha256="0" * 64,
                width=PAGE[0],
                height=PAGE[1],
                y0=PAGE[1] * i,
                y1=PAGE[1] * (i + 1),
            )
            for i, name in enumerate(("001.jpg", "002.jpg"))
        ],
    ).save(work / "ingest.json")


def write_slices(
    cfg: Config, y_ranges: list[tuple[int, int]], filtered: set[int], chapter: str = CHAPTER
) -> None:
    """A hand-written slices.json."""
    work = cfg.paths.work_root / SERIES / chapter
    work.mkdir(parents=True, exist_ok=True)
    SlicesArtifact(
        strip_width=PAGE[0],
        strip_height=STRIP_HEIGHT,
        bands=[],
        slices=[Slice(index=i, y0=a, y1=b, filtered=i in filtered) for i, (a, b) in enumerate(y_ranges)],
    ).save(work / "slices.json")


def write_flat_patch(cfg: Config, chapter: str = CHAPTER, color: int = 255) -> None:
    """A flat-fill patch for r0001 whose mask covers a 100 x 40 rectangle inside PATCH_BOX."""
    work = cfg.paths.work_root / SERIES / chapter
    work.mkdir(parents=True, exist_ok=True)
    h, w = PATCH_BOX.height, PATCH_BOX.width
    pixels = np.full((h, w, 3), color, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[20:60, 20:120] = True  # 100 x 40 rectangle at strip (120, 220)-(220, 260)
    save_patches(
        work / "patches.npz",
        {"r0001": (torch.from_numpy(pixels).permute(2, 0, 1), torch.from_numpy(mask))},
    )
    InpaintArtifact(
        items=[
            InpaintItem(
                region_id="r0001", box=PATCH_BOX, method="flat", fill=(255, 255, 255), mask_px=int(mask.sum())
            )
        ]
    ).save(work / "inpaint.json")


def write_layout(cfg: Config, chapter: str = CHAPTER, *, with_overflow: bool = False) -> None:
    """A hand-written layout.json with one dialogue item (plus an overflow-flagged one on demand)."""
    work = cfg.paths.work_root / SERIES / chapter
    work.mkdir(parents=True, exist_ok=True)
    items = [
        LayoutItem(
            region_id="r0001",
            font_role="dialogue",
            font="ComicNeue-Bold.ttf",
            size_px=32,
            lines=["Hello world"],
            box=TEXT_BOX,
            align="center",
            color=(0, 0, 0),
        )
    ]
    if with_overflow:
        items.append(
            LayoutItem(
                region_id="r0002",
                font_role="dialogue",
                font="ComicNeue-Bold.ttf",
                size_px=32,
                lines=["Overflow line"],
                box=BBox(x0=560, y0=60, x1=700, y1=120),
                overflow=True,
            )
        )
    LayoutArtifact(items=items).save(work / "layout.json")


def output_dir(cfg: Config, chapter: str = CHAPTER) -> Path:
    """The chapter's output folder."""
    return cfg.paths.output_root / SERIES / chapter


# ---------------------------------------------------------------- stage


def test_export_stage_writes_slices_and_export_json(cfg: Config) -> None:
    write_pages(cfg)
    write_ingest(cfg)
    write_slices(cfg, [(0, 400), (400, 800), (800, 1200)], filtered={1})
    write_flat_patch(cfg)
    write_layout(cfg)
    ctx = make_context(cfg, SERIES, CHAPTER)

    outcome = run_stage(ExportStage(), ctx)
    assert outcome.status == "done"
    out = output_dir(cfg)
    assert sorted(p.name for p in out.iterdir()) == ["0001.jpg", "0002.jpg"]  # the filtered slice is skipped
    export = ExportArtifact.load(ctx.paths.artifact("export.json"))
    assert export.quality == 95 and export.subsampling == "444"
    assert [(f.name, f.slice_index, f.width, f.height) for f in export.files] == [
        ("0001.jpg", 0, 800, 400),
        ("0002.jpg", 2, 800, 400),
    ]
    for f in export.files:
        assert len((out / f.name).read_bytes()) == f.bytes

    decoded = np.asarray(Image.open(out / "0001.jpg").convert("RGB")).astype(int)
    # the masked rectangle is white; the outermost ring can ring by up to 8 after JPEG (block edges
    # against the red page colour), so the tolerance-6 check covers the rectangle inset by 2 px
    assert (np.abs(decoded[222:258, 122:218] - 255) <= 6).all()
    outside = decoded[205:215, 105:115]  # inside the patch box, outside the mask
    assert (np.abs(outside - np.array(PAGE_COLOR)) <= 6).all()  # keeps the page colour
    text_area = decoded[TEXT_BOX.y0 : TEXT_BOX.y1, TEXT_BOX.x0 : TEXT_BOX.x1]
    assert (np.abs(text_area - np.array(PAGE_COLOR)).max(axis=-1) > 60).sum() > 30  # rendered English

    assert run_stage(ExportStage(), make_context(cfg, SERIES, CHAPTER)).status == "skipped"


def test_lama_patches_override_the_flat_fill(cfg: Config) -> None:
    write_pages(cfg)
    write_ingest(cfg)
    write_slices(cfg, [(0, 400), (400, 800), (800, 1200)], filtered={1})
    write_flat_patch(cfg)
    write_layout(cfg)
    work = cfg.paths.work_root / SERIES / CHAPTER
    h, w = PATCH_BOX.height, PATCH_BOX.width
    lama_color = (10, 10, 250)
    pixels = np.full((h, w, 3), lama_color, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[20:60, 20:120] = True
    save_patches(
        work / "patches_lama.npz",
        {"r0001": (torch.from_numpy(pixels).permute(2, 0, 1), torch.from_numpy(mask))},
    )
    InpaintArtifact(items=[InpaintItem(region_id="r0001", box=PATCH_BOX, method="lama")]).save(
        work / "inpaint_lama.json"
    )

    outcome = run_stage(ExportStage(), make_context(cfg, SERIES, CHAPTER))
    assert outcome.status == "done"
    assert outcome.metrics["patches"] == 2.0  # flat + LaMa
    decoded = np.asarray(Image.open(output_dir(cfg) / "0001.jpg").convert("RGB")).astype(int)
    assert (
        np.abs(decoded[222:258, 122:218] - np.array(lama_color)) <= 6
    ).all()  # inset by 2 px (JPEG ringing)


def test_reexport_deletes_only_stale_numbered_files(cfg: Config) -> None:
    write_pages(cfg)
    write_ingest(cfg)
    write_slices(cfg, [(0, 400), (400, 800), (800, 1200)], filtered={1})
    write_flat_patch(cfg)
    write_layout(cfg)
    assert run_stage(ExportStage(), make_context(cfg, SERIES, CHAPTER)).status == "done"
    out = output_dir(cfg)
    Image.new("RGB", (4, 4), (0, 0, 0)).save(out / "cover.png", format="PNG")
    (out / "notes.txt").write_text("keep me", encoding="utf-8")

    write_slices(cfg, [(0, 1200)], filtered=set())  # fewer slices: a stale 0002.jpg must go
    assert run_stage(ExportStage(), make_context(cfg, SERIES, CHAPTER)).status == "done"
    assert sorted(p.name for p in out.iterdir()) == ["0001.jpg", "cover.png", "notes.txt"]
    assert run_stage(ExportStage(), make_context(cfg, SERIES, CHAPTER)).status == "skipped"


def test_quality_change_reruns_and_changes_bytes(cfg: Config) -> None:
    write_pages(cfg)
    write_ingest(cfg)
    write_slices(cfg, [(0, 400), (400, 800), (800, 1200)], filtered={1})
    write_flat_patch(cfg)
    write_layout(cfg)
    assert run_stage(ExportStage(), make_context(cfg, SERIES, CHAPTER)).status == "done"
    first = len((output_dir(cfg) / "0001.jpg").read_bytes())

    cfg60 = cfg.model_copy(update={"export": ExportConfig(jpeg_quality=60)})
    assert run_stage(ExportStage(), make_context(cfg60, SERIES, CHAPTER)).status == "done"
    second = len((output_dir(cfg) / "0001.jpg").read_bytes())
    assert second != first
    assert ExportArtifact.load(cfg60.paths.work_root / SERIES / CHAPTER / "export.json").quality == 60
    assert run_stage(ExportStage(), make_context(cfg60, SERIES, CHAPTER)).status == "skipped"


def test_layout_change_reruns_export(cfg: Config) -> None:
    write_pages(cfg)
    write_ingest(cfg)
    write_slices(cfg, [(0, 400), (400, 800), (800, 1200)], filtered={1})
    write_flat_patch(cfg)
    write_layout(cfg)
    assert run_stage(ExportStage(), make_context(cfg, SERIES, CHAPTER)).status == "done"
    assert run_stage(ExportStage(), make_context(cfg, SERIES, CHAPTER)).status == "skipped"

    write_layout(cfg, with_overflow=True)  # changed layout.json re-runs the stage
    outcome = run_stage(ExportStage(), make_context(cfg, SERIES, CHAPTER))
    assert outcome.status == "done"
    assert outcome.metrics["overflow_items"] == 1.0


def test_missing_input_fails_with_documented_message(cfg: Config) -> None:
    write_pages(cfg)
    write_ingest(cfg)
    write_slices(cfg, [(0, 400), (400, 800), (800, 1200)], filtered={1})
    write_flat_patch(cfg)  # layout.json is never written
    outcome = run_stage(ExportStage(), make_context(cfg, SERIES, CHAPTER))
    assert outcome.status == "failed"
    assert outcome.error is not None
    assert "layout.json missing — run the typeset stage first" in outcome.error
    assert (
        Manifest.load(cfg.paths.work_root / SERIES / CHAPTER / "manifest.json").stages["export"].status
        == "failed"
    )


def test_export_metrics(cfg: Config) -> None:
    write_pages(cfg)
    write_ingest(cfg)
    write_slices(cfg, [(0, 400), (400, 800), (800, 1200)], filtered={1})
    write_flat_patch(cfg)
    write_layout(cfg, with_overflow=True)
    outcome = run_stage(ExportStage(), make_context(cfg, SERIES, CHAPTER))
    assert outcome.status == "done"
    export = ExportArtifact.load(cfg.paths.work_root / SERIES / CHAPTER / "export.json")
    assert outcome.metrics["slices"] == 2.0
    assert outcome.metrics["bytes"] == sum(f.bytes for f in export.files)
    assert outcome.metrics["patches"] == 1.0
    assert outcome.metrics["glyph_items"] == 2.0
    assert outcome.metrics["overflow_items"] == 1.0


# ---------------------------------------------------------------- CLI


def test_cli_export_runs_only_the_export_stage(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    for chapter in ("Chapter 1", "Chapter 2"):
        write_pages(cfg, chapter)
        write_ingest(cfg, chapter)
        write_slices(cfg, [(0, 400), (400, 800), (800, 1200)], filtered={1}, chapter=chapter)
        write_flat_patch(cfg, chapter)
        write_layout(cfg, chapter)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["export", SERIES])
    assert result.exit_code == 0
    for chapter in ("Chapter 1", "Chapter 2"):
        assert f"{SERIES}/{chapter} export: done" in result.output
        assert (cfg.paths.work_root / SERIES / chapter / "export.json").is_file()
        assert (output_dir(cfg, chapter) / "0001.jpg").is_file()
    assert "ingest:" not in result.output and "slice:" not in result.output  # only the export stage ran

    result = runner.invoke(app, ["export", SERIES])
    assert result.exit_code == 0
    assert result.output.count("skipped") == 2

    result = runner.invoke(app, ["export", SERIES, "--chapter", "Chapter 1", "--force"])
    assert result.exit_code == 0
    assert f"{SERIES}/Chapter 1 export: done" in result.output
    assert "Chapter 2" not in result.output

    result = runner.invoke(app, ["export", "NoSuchSeries"])
    assert result.exit_code == 2
    assert "no chapters found" in result.output

    result = runner.invoke(app, ["slice", SERIES])  # the slice command is unaffected
    assert result.exit_code == 0


# ---------------------------------------------------------------- synthetic Korean chapter


ENGLISH = (
    "This dungeon opened faster than expected!",
    "Stay behind me, I will hold the line.",
    "Nobody believed me about the gate.",
)


def test_synthetic_korean_page_export(cfg: Config) -> None:
    page = make_korean_page(11, width=800, height=1400, n_bubbles=3, n_free=0, n_sfx=0, background="flat")
    raw = cfg.paths.library_root / SERIES / CHAPTER
    raw.mkdir(parents=True, exist_ok=True)
    page.image.save(raw / "001.jpg", format="JPEG", quality=95)
    work = cfg.paths.work_root / SERIES / CHAPTER
    work.mkdir(parents=True, exist_ok=True)
    IngestArtifact(
        series=SERIES,
        chapter=CHAPTER,
        strip_width=800,
        strip_height=1400,
        files=[SourceFile(index=0, name="001.jpg", sha256="0" * 64, width=800, height=1400, y0=0, y1=1400)],
    ).save(work / "ingest.json")
    SlicesArtifact(strip_width=800, strip_height=1400, bands=[], slices=[Slice(index=0, y0=0, y1=1400)]).save(
        work / "slices.json"
    )

    clean = np.array(page.clean)
    items: list[InpaintItem] = []
    layout_items: list[LayoutItem] = []
    patches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for i, truth in enumerate(page.regions):
        region_id = f"r{i + 1:04d}"
        box = BBox(
            x0=max(truth.bbox.x0 - 8, 0),
            y0=max(truth.bbox.y0 - 8, 0),
            x1=min(truth.bbox.x1 + 8, 800),
            y1=min(truth.bbox.y1 + 8, 1400),
        )
        pixels = np.ascontiguousarray(clean[box.y0 : box.y1, box.x0 : box.x1])
        mask = np.ascontiguousarray(page.text_mask[box.y0 : box.y1, box.x0 : box.x1])
        patches[region_id] = (torch.from_numpy(pixels).permute(2, 0, 1), torch.from_numpy(mask))
        items.append(
            InpaintItem(region_id=region_id, box=box, method="flat", fill=truth.fill, mask_px=int(mask.sum()))
        )
        bbox = truth.bubble_bbox
        assert bbox is not None  # n_bubbles=3, n_free=0, n_sfx=0: every region is a bubble
        target = inscribed_box(bbox, truth.bubble_polygon, margin_px=6)
        layout_items.append(layout_region(region_id, ENGLISH[i], target, role="dialogue"))
    save_patches(work / "patches.npz", patches)
    InpaintArtifact(items=items).save(work / "inpaint.json")
    LayoutArtifact(items=layout_items).save(work / "layout.json")

    outcome = run_stage(ExportStage(), make_context(cfg, SERIES, CHAPTER))
    assert outcome.status == "done"
    assert outcome.metrics["patches"] == 3.0 and outcome.metrics["glyph_items"] == 3.0

    exported = np.asarray(Image.open(output_dir(cfg) / "0001.jpg").convert("RGB")).astype(int)
    diff = np.abs(exported - clean).max(axis=-1)
    outside = np.ones((1400, 800), dtype=bool)
    for item in layout_items:
        outside[item.box.y0 : item.box.y1, item.box.x0 : item.box.x1] = False
    assert diff[outside].mean() < 3  # the page equals page.clean outside the layout boxes
    for item in layout_items:
        inside = diff[item.box.y0 : item.box.y1, item.box.x0 : item.box.x1]
        assert (inside > 60).sum() >= 30  # rendered English inside every layout box

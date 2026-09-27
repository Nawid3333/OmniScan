"""The `qa` stage: exported pages decoded back into the strip and re-read with a fake OCR reader (CPU torch)."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image, ImageDraw

from omniscan.core.config import Config, GpuConfig, OcrConfig, PathsConfig
from omniscan.core.schemas import (
    BBox,
    ExportArtifact,
    ExportFile,
    IngestArtifact,
    QaArtifact,
    Region,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
    SourceFile,
)
from omniscan.core.stage import ChapterContext, make_context, run_stage
from omniscan.qa.stage import QaStage, lettered_strip

SERIES, CHAPTER = "S", "Chapter 1"
ORIGINAL = "여기가 어디지?"


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
        ocr=OcrConfig(engine="manga_ocr"),
    )


class FakeScheduler:
    """Hands out the vision group's pre-built models."""

    def __init__(self, models: dict[str, Any]) -> None:
        self.models = models

    def acquire(self, group: str) -> dict[str, Any]:
        assert group == "vision"
        return self.models

    def reset_peak(self) -> None:
        pass

    def peak_gib(self) -> float:
        return 0.0


class ColourReader:
    """Reads the original text where the crop is mostly red (the source text left over), English elsewhere."""

    def __init__(self) -> None:
        self.crops = 0

    def read(self, crops: list[torch.Tensor]) -> list[tuple[str, float]]:
        self.crops += len(crops)
        out = []
        for crop in crops:
            red, blue = crop[0].float().mean().item(), crop[2].float().mean().item()
            out.append((ORIGINAL, 0.9) if red - blue > 60 else ("WHERE AM I?", 0.9))
        return out


def region(rid: str, y0: int, text: str = ORIGINAL) -> Region:
    return Region(
        id=rid,
        slice_index=0 if y0 < 300 else 1,
        kind="bubble_text",
        bbox=BBox(x0=100, y0=y0, x1=300, y1=y0 + 60),
        text=text,
    )


def prepared(cfg: Config) -> ChapterContext:
    """A two-slice chapter already exported: slice 1's region still shows red 'source text'."""
    ctx = make_context(cfg, SERIES, CHAPTER)
    (cfg.paths.library_root / SERIES / CHAPTER).mkdir(parents=True, exist_ok=True)
    work = ctx.paths
    IngestArtifact(
        series=SERIES,
        chapter=CHAPTER,
        strip_width=400,
        strip_height=600,
        files=[SourceFile(index=0, name="001.jpg", sha256="x", width=400, height=600, y0=0, y1=600)],
    ).save(work.artifact("ingest.json"))
    SlicesArtifact(
        strip_width=400,
        strip_height=600,
        bands=[],
        slices=[Slice(index=0, y0=0, y1=300), Slice(index=1, y0=300, y1=600)],
    ).save(work.artifact("slices.json"))
    RegionsArtifact(regions=[region("r0001", 100), region("r0002", 400), region("r0003", 200, "WAIT!")]).save(
        work.artifact("ocr.json")
    )
    files = []
    ctx.paths.output_dir.mkdir(parents=True, exist_ok=True)
    for number, (y0, y1) in enumerate(((0, 300), (300, 600)), start=1):
        page = Image.new("RGB", (400, y1 - y0), "white")
        if y0 == 300:  # the region at rows 400-460 was never cleaned: red "glyphs" remain
            ImageDraw.Draw(page).rectangle((100, 100, 300, 160), fill=(220, 20, 20))
        buf = io.BytesIO()
        page.save(buf, format="JPEG", quality=95)
        name = f"{number:04d}.jpg"
        (ctx.paths.output_dir / name).write_bytes(buf.getvalue())
        files.append(
            ExportFile(
                name=name, slice_index=number - 1, width=400, height=y1 - y0, bytes=len(buf.getvalue())
            )
        )
    ExportArtifact(quality=95, subsampling="444", files=files).save(work.artifact("export.json"))
    return ctx


def test_qa_finds_the_region_whose_original_text_is_still_on_the_page(cfg: Config) -> None:
    ctx = prepared(cfg)
    reader = ColourReader()
    ctx.gpu = FakeScheduler({"reader": reader})
    outcome = run_stage(QaStage(), ctx)
    assert outcome.status == "done"
    assert (
        outcome.metrics["checked"] == 2.0 and outcome.metrics["source_left"] == 1.0
    )  # r0003 has no source text
    assert reader.crops == 2
    qa = QaArtifact.load(ctx.paths.artifact("qa.json"))
    assert [(i.region_id, i.kind) for i in qa.issues] == [("r0002", "source_left")]
    assert qa.checked == 2
    # the output folder changed after an edit + re-export: the stage runs again
    assert run_stage(QaStage(), make_context(cfg, SERIES, CHAPTER, ctx.gpu)).status == "skipped"
    digest = hashlib.sha256((ctx.paths.output_dir / "0002.jpg").read_bytes()).hexdigest()
    white = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(white, format="JPEG", quality=95)
    (ctx.paths.output_dir / "0002.jpg").write_bytes(white.getvalue())
    assert hashlib.sha256(white.getvalue()).hexdigest() != digest
    again = run_stage(QaStage(), make_context(cfg, SERIES, CHAPTER, ctx.gpu))
    assert again.status == "done" and QaArtifact.load(ctx.paths.artifact("qa.json")).issues == []


def test_qa_without_an_export_fails_and_says_why(cfg: Config) -> None:
    ctx = prepared(cfg)
    ctx.paths.artifact("export.json").unlink()
    ctx.gpu = FakeScheduler({"reader": ColourReader()})
    outcome = run_stage(QaStage(), ctx)
    assert outcome.status == "failed" and "export" in (outcome.error or "")


def test_rows_without_an_exported_image_are_black(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = prepared(cfg)
    ingest = IngestArtifact.load(ctx.paths.artifact("ingest.json"))
    real_empty = torch.empty
    monkeypatch.setattr(
        torch, "empty", lambda *args, **kwargs: real_empty(*args, **kwargs).fill_(7)
    )  # garbage
    strip = lettered_strip(ctx, ingest, [("0002.jpg", 300, 600)])  # slice 0 filtered: no image for rows 0-300
    assert int(strip[:, :300].max()) == 0
    assert (
        int(strip[0, 430, 200]) > 150 and int(strip[2, 430, 200]) < 100
    )  # the red block of image 2, in place
    assert int(strip[:, 300:390].min()) > 200  # white rows of image 2

"""OCR stage wiring: real ingest/slice/strip, fake models, resumability and failure records (card C4a)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from omniscan.core.config import Config, GpuConfig, OcrConfig, PathsConfig, SfxConfig
from omniscan.core.schemas import BBox, Region, RegionsArtifact
from omniscan.core.stage import ChapterContext, make_context, run_chapter, run_stage
from omniscan.ingest.stage import IngestStage
from omniscan.ocr.lines import LineBox
from omniscan.ocr.stage import OcrStage
from omniscan.slicer.stage import SliceStage
from tests.fixtures.korean_pages import KOREAN_LINES

SERIES = "S"
CHAPTER = "Chapter 1"


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """CPU-only config with all paths under tmp_path and a strip-sized OCR tile."""
    return Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
        ocr=OcrConfig(tile_px=400),
        sfx=SfxConfig(sweep=False),  # the sweep's own tests below switch it on
    )


class FakeScheduler:
    """GpuScheduler handing out pre-built models (stands in for the VramManager)."""

    def __init__(self, models: dict[str, Any]) -> None:
        self.models = models

    def acquire(self, group: str) -> dict[str, Any]:
        assert group == "vision"
        return self.models

    def reset_peak(self) -> None:
        pass

    def peak_gib(self) -> float:
        return 0.0


class FakeLineDetector:
    """Scripted line detector: per cumulative tile index a list of (box, score)."""

    def __init__(
        self, script: dict[int, list[tuple[tuple[float, float, float, float], float]]] | None = None
    ) -> None:
        self.script = script or {}
        self.next_index = 0

    def detect(self, tiles: Sequence[torch.Tensor]) -> list[list[LineBox]]:
        out = []
        for _ in tiles:
            out.append([LineBox(box=box, score=score) for box, score in self.script.get(self.next_index, [])])
            self.next_index += 1
        return out


class FakeRecognizer:
    """Scripted recognizer: the same reading for every crop."""

    def __init__(self, reading: tuple[str, float] = ("텍스트", 0.95)) -> None:
        self.reading = reading

    def read(self, crops: list[torch.Tensor]) -> list[tuple[str, float]]:
        return [self.reading for _ in crops]


def write_raw(cfg: Config) -> None:
    """Two 400x300 noise pages: a 400x600 strip."""
    rng = np.random.default_rng(0)
    raw = cfg.paths.library_root / SERIES / CHAPTER
    raw.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        Image.fromarray(rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)).save(
            raw / f"{i + 1:03d}.jpg", format="JPEG", quality=95
        )


def write_regions(ctx: ChapterContext, bbox: tuple[int, int, int, int] = (10, 50, 390, 200)) -> None:
    RegionsArtifact(
        regions=[
            Region(
                id="r0001",
                slice_index=0,
                kind="bubble_text",
                bbox=BBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
            )
        ]
    ).save(ctx.paths.artifact("regions.json"))


def prepared(cfg: Config) -> ChapterContext:
    """A chapter with ingest + slices done and a region waiting for OCR."""
    write_raw(cfg)
    ctx = make_context(cfg, SERIES, CHAPTER)
    assert [o.status for o in run_chapter([IngestStage(), SliceStage()], ctx)] == ["done", "done"]
    write_regions(ctx)
    return ctx


def test_ocr_stage_writes_ocr_json_and_is_resumable(cfg: Config) -> None:
    def scheduler() -> FakeScheduler:
        """Fresh scripted models — the fake detector counts tiles across runs."""
        return FakeScheduler(
            {
                "line_detector": FakeLineDetector({0: [((20, 60, 200, 90), 0.95)]}),
                "recognizer": FakeRecognizer(),
            }
        )

    ctx = prepared(cfg)
    ctx.gpu = scheduler()

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.status == "done"
    assert outcome.metrics["tiles"] == 1.0 and outcome.metrics["lines"] == 1.0
    assert outcome.metrics["regions"] == 1.0 and outcome.metrics["regions_empty"] == 0.0
    assert outcome.metrics["orphan_lines"] == 0.0
    built = RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions[0]
    assert built.text == "텍스트" and built.confidence == 0.95
    assert built.lines[0].bbox == BBox(x0=20, y0=60, x1=200, y1=90)
    assert built.lines[0].engine == cfg.ocr.rec_repo.split("/")[-1]

    manifest = make_context(cfg, SERIES, CHAPTER).manifest
    assert manifest.stages["ocr"].status == "done" and manifest.stages["ocr"].outputs == ["ocr.json"]
    assert run_stage(OcrStage(), make_context(cfg, SERIES, CHAPTER, scheduler())).status == "skipped"

    # changing ocr.rec_repo re-runs the stage (and the engine name follows the repo)
    rec = cfg.model_copy(update={"ocr": cfg.ocr.model_copy(update={"rec_repo": "PaddlePaddle/other_rec"})})
    run_stage(OcrStage(), make_context(rec, SERIES, CHAPTER, scheduler()))
    rebuilt = RegionsArtifact.load(rec.paths.work_root / SERIES / CHAPTER / "ocr.json").regions[0]
    assert rebuilt.lines[0].engine == "other_rec"

    # changing detect.reading_direction re-runs it
    direction = cfg.model_copy(update={"detect": cfg.detect.model_copy(update={"reading_direction": "rtl"})})
    assert run_stage(OcrStage(), make_context(direction, SERIES, CHAPTER, scheduler())).status == "done"

    # changing regions.json re-runs it
    write_regions(make_context(cfg, SERIES, CHAPTER), bbox=(10, 50, 390, 100))
    assert run_stage(OcrStage(), make_context(cfg, SERIES, CHAPTER, scheduler())).status == "done"


def test_ocr_stage_missing_upstream_artifacts_fail_and_are_recorded(cfg: Config) -> None:
    write_raw(cfg)
    scheduler = FakeScheduler({"line_detector": FakeLineDetector(), "recognizer": FakeRecognizer()})

    outcome = run_stage(OcrStage(), make_context(cfg, SERIES, CHAPTER, scheduler))

    assert outcome.status == "failed"
    assert outcome.error is not None and "regions.json missing — run the detect stage first" in outcome.error
    manifest = make_context(cfg, SERIES, CHAPTER).manifest
    assert manifest.stages["ocr"].status == "failed"
    assert (
        manifest.stages["ocr"].error is not None
        and "run the detect stage first" in manifest.stages["ocr"].error
    )

    ctx = make_context(cfg, SERIES, CHAPTER)
    assert [o.status for o in run_chapter([IngestStage(), SliceStage()], ctx)] == ["done", "done"]
    write_regions(ctx)
    (ctx.paths.work_dir / "ingest.json").unlink()
    outcome = run_stage(OcrStage(), make_context(cfg, SERIES, CHAPTER, scheduler))
    assert outcome.status == "failed"
    assert outcome.error is not None and "ingest.json missing — run the slice stage first" in outcome.error


def _scheduler_with(lines: dict[int, list[tuple[tuple[float, float, float, float], float]]]) -> FakeScheduler:
    return FakeScheduler({"line_detector": FakeLineDetector(lines), "recognizer": FakeRecognizer()})


def test_regions_below_drop_conf_are_dropped_and_counted(cfg: Config) -> None:
    strict = cfg.model_copy(update={"ocr": OcrConfig(tile_px=400, drop_conf=0.99)})
    ctx = prepared(strict)
    ctx.gpu = _scheduler_with({0: [((20, 60, 200, 90), 0.95)]})

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.status == "done"
    assert outcome.metrics["regions"] == 1.0 and outcome.metrics["regions_dropped"] == 1.0
    assert RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions == []


def test_regions_without_any_text_are_dropped(cfg: Config) -> None:
    ctx = prepared(cfg)
    ctx.gpu = _scheduler_with({})  # the line detector finds nothing

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.metrics["regions_empty"] == 1.0 and outcome.metrics["regions_dropped"] == 1.0
    assert RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions == []


def test_confident_regions_are_kept(cfg: Config) -> None:
    ctx = prepared(cfg)
    ctx.gpu = _scheduler_with({0: [((20, 60, 200, 90), 0.95)]})

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.metrics["regions_dropped"] == 0.0
    assert len(RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions) == 1


# ---------------------------------------------------------------- crop-reading engine (card O1b, test 7)


class FakeReader:
    """Scripted crop-reading engine: one reading for every region crop."""

    def __init__(self, reading: tuple[str, float]) -> None:
        self.reading = reading
        self.n_crops = 0

    def read(self, crops: list[torch.Tensor]) -> list[tuple[str, float]]:
        self.n_crops = len(crops)
        return [self.reading for _ in crops]


MANGA_CFG = OcrConfig(tile_px=400, engine="manga_ocr")


def test_manga_ocr_stage_reads_whole_region_crops(cfg: Config) -> None:
    manga = cfg.model_copy(update={"ocr": MANGA_CFG})
    ctx = prepared(manga)
    ctx.gpu = FakeScheduler({"reader": FakeReader(("あいう、", 0.9))})

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.status == "done"
    assert outcome.metrics["tiles"] == 0.0 and outcome.metrics["lines"] == 1.0
    assert outcome.metrics["regions_dropped"] == 0.0
    built = RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions[0]
    assert built.text == "あいう、" and built.confidence == 0.9
    assert built.lines[0].engine == "ocr-rec-manga-ocr-2025"  # the engine's default rec model id
    assert built.lines[0].bbox == BBox(x0=10, y0=50, x1=390, y1=200)  # one line per region: its bbox
    assert run_stage(OcrStage(), make_context(manga, SERIES, CHAPTER, ctx.gpu)).status == "skipped"


def test_manga_ocr_stage_drops_low_confidence_regions(cfg: Config) -> None:
    strict = cfg.model_copy(update={"ocr": MANGA_CFG.model_copy(update={"drop_conf": 0.99})})
    ctx = prepared(strict)
    ctx.gpu = FakeScheduler({"reader": FakeReader(("あ", 0.9))})

    assert run_stage(OcrStage(), ctx).metrics["regions_dropped"] == 1.0
    assert RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions == []


def test_manga_ocr_stage_drops_regions_without_text(cfg: Config) -> None:
    ctx = prepared(cfg.model_copy(update={"ocr": MANGA_CFG}))
    ctx.gpu = FakeScheduler({"reader": FakeReader(("", 0.8))})

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.metrics["regions_empty"] == 1.0 and outcome.metrics["regions_dropped"] == 1.0
    assert RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions == []


def test_manga_ocr_stage_reruns_when_engine_or_rec_model_change(cfg: Config) -> None:
    ctx = prepared(cfg)
    ctx.gpu = FakeScheduler({"reader": FakeReader(("あ", 0.9))})
    manga = cfg.model_copy(update={"ocr": MANGA_CFG})
    assert run_stage(OcrStage(), make_context(manga, SERIES, CHAPTER, ctx.gpu)).status == "done"

    rec_model = manga.model_copy(
        update={"ocr": MANGA_CFG.model_copy(update={"rec_model": "ocr-rec-manga-ocr-base"})}
    )
    outcome = run_stage(OcrStage(), make_context(rec_model, SERIES, CHAPTER, ctx.gpu))

    assert outcome.status == "done"  # engine and rec_model are part of the config hash
    built = RegionsArtifact.load(rec_model.paths.work_root / SERIES / CHAPTER / "ocr.json").regions[0]
    assert built.lines[0].engine == "ocr-rec-manga-ocr-base"  # the label follows the rec model id


# ---------------------------------------------------------------- paddleocr_vl (card O1d, test 5)


VL_CFG = OcrConfig(tile_px=400, engine="paddleocr_vl")


def test_paddleocr_vl_stage_reads_whole_region_crops(cfg: Config) -> None:
    vl = cfg.model_copy(update={"ocr": VL_CFG})
    ctx = prepared(vl)
    ctx.gpu = FakeScheduler({"reader": FakeReader(("I even brought my potions!", 0.9))})

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.status == "done"
    assert outcome.metrics["tiles"] == 0.0 and outcome.metrics["lines"] == 1.0
    assert outcome.metrics["regions_dropped"] == 0.0
    built = RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions[0]
    assert built.text == "I even brought my potions!" and built.confidence == 0.9
    assert built.lines[0].engine == "ocr-vl-1.6"  # the engine's default catalog id
    assert run_stage(OcrStage(), make_context(vl, SERIES, CHAPTER, ctx.gpu)).status == "skipped"


def test_paddleocr_vl_stage_drops_low_confidence_regions(cfg: Config) -> None:
    strict = cfg.model_copy(update={"ocr": VL_CFG.model_copy(update={"drop_conf": 0.99})})
    ctx = prepared(strict)
    ctx.gpu = FakeScheduler({"reader": FakeReader(("あ", 0.9))})

    assert run_stage(OcrStage(), ctx).metrics["regions_dropped"] == 1.0
    assert RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions == []


def test_paddleocr_vl_stage_drops_regions_without_text(cfg: Config) -> None:
    ctx = prepared(cfg.model_copy(update={"ocr": VL_CFG}))
    ctx.gpu = FakeScheduler({"reader": FakeReader(("", 0.8))})

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.metrics["regions_empty"] == 1.0 and outcome.metrics["regions_dropped"] == 1.0
    assert RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions == []


# ---------------------------------------------------------------- watermark text reclassification (F2b)

AD_1 = '구글검색 "먹튀검증 스포위키"'
AD_2 = "라이브스코어 스포츠중계 가상토토 전문가 정기/오목 웹툰"


def write_patterns(path: Path, *patterns: str) -> Path:
    """A watermark_text.toml with `patterns` (created on demand)."""
    entries = ", ".join(f'"{pattern}"' for pattern in patterns)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"[watermark_text]\npatterns = [{entries}]\n", encoding="utf-8")
    return path


class FakeCropReader:
    """Scripted crop-reading engine: one reading per region crop, in region order."""

    def __init__(self, readings: list[tuple[str, float]]) -> None:
        self.readings = readings

    def read(self, crops: list[torch.Tensor]) -> list[tuple[str, float]]:
        return [self.readings[index] for index in range(len(crops))]


def write_mixed_regions(ctx: ChapterContext) -> None:
    """Four regions: dialogue bubble, two ad regions and an sfx whose text happens to be an ad."""
    RegionsArtifact(
        regions=[
            Region(id="r0001", slice_index=0, kind="bubble_text", bbox=BBox(x0=10, y0=30, x1=390, y1=120)),
            Region(id="r0002", slice_index=0, kind="bubble_text", bbox=BBox(x0=10, y0=180, x1=390, y1=270)),
            Region(id="r0003", slice_index=0, kind="free_text", bbox=BBox(x0=10, y0=330, x1=390, y1=420)),
            Region(id="r0004", slice_index=0, kind="sfx", bbox=BBox(x0=10, y0=480, x1=390, y1=560)),
        ]
    ).save(ctx.paths.artifact("regions.json"))


def test_ocr_stage_reclassifies_ad_text_as_watermark(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped = write_patterns(
        tmp_path / "config" / "watermark_text.toml", "구글검색", "라이브스코어", "스포위키"
    )
    user = tmp_path / "user" / "watermark_text.toml"  # missing on purpose: skipped by the loader
    monkeypatch.setattr("omniscan.ocr.stage.default_watermark_text_paths", lambda: [shipped, user])
    manga = cfg.model_copy(update={"ocr": MANGA_CFG})
    ctx = prepared(manga)
    write_mixed_regions(ctx)
    ctx.gpu = FakeScheduler(
        {"reader": FakeCropReader([(KOREAN_LINES[0], 0.95), (AD_1, 0.9), (AD_2, 0.85), ("쾅!", 0.9)])}
    )

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.status == "done"
    assert outcome.metrics["watermarked"] == 2.0
    regions = RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions
    assert [r.kind for r in regions] == ["bubble_text", "watermark", "watermark", "sfx"]
    assert regions[1].text == AD_1 and regions[2].text == AD_2  # watermark regions keep their OCR text


def test_stored_watermark_regions_pass_through_unread(cfg: Config) -> None:
    manga = cfg.model_copy(update={"ocr": MANGA_CFG})
    ctx = prepared(manga)
    zone = Region(id="r0001", slice_index=0, kind="watermark", bbox=BBox(x0=0, y0=0, x1=400, y1=20))
    RegionsArtifact(
        regions=[
            zone,
            Region(id="r0002", slice_index=0, kind="bubble_text", bbox=BBox(x0=10, y0=50, x1=390, y1=200)),
        ]
    ).save(ctx.paths.artifact("regions.json"))
    reader = FakeReader(("あいう", 0.9))
    ctx.gpu = FakeScheduler({"reader": reader})

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.status == "done"
    assert reader.n_crops == 1  # only the dialogue was read
    assert outcome.metrics["watermarked"] == 1.0 and outcome.metrics["regions_dropped"] == 0.0
    regions = RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions
    assert regions[0] == zone  # unread, unchanged, still first: inpaint erases its whole box
    assert regions[1].text == "あいう"


def test_ocr_stage_reruns_when_the_watermark_patterns_change(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped = write_patterns(tmp_path / "config" / "watermark_text.toml", "라이브스코어")
    user = tmp_path / "user" / "watermark_text.toml"
    monkeypatch.setattr("omniscan.ocr.stage.default_watermark_text_paths", lambda: [shipped, user])
    manga = cfg.model_copy(update={"ocr": MANGA_CFG})
    ctx = prepared(manga)
    ctx.gpu = FakeScheduler({"reader": FakeReader((AD_1, 0.9))})  # the ad text, no pattern matches it yet

    assert run_stage(OcrStage(), ctx).status == "done"
    assert RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions[0].kind == "bubble_text"
    assert run_stage(OcrStage(), make_context(manga, SERIES, CHAPTER, ctx.gpu)).status == "skipped"

    write_patterns(user, "스포위키")  # AD_1 contains "스포위키": editing the user file must re-run the stage
    rerun = run_stage(OcrStage(), make_context(manga, SERIES, CHAPTER, ctx.gpu))
    assert rerun.status == "done"
    assert RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions[0].kind == "watermark"
    assert run_stage(OcrStage(), make_context(manga, SERIES, CHAPTER, ctx.gpu)).status == "skipped"


def test_config_subset_fingerprints_the_loaded_patterns(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped = write_patterns(tmp_path / "config" / "watermark_text.toml", "구글검색", "스포위키")
    user = tmp_path / "user" / "watermark_text.toml"
    monkeypatch.setattr("omniscan.ocr.stage.default_watermark_text_paths", lambda: [shipped, user])
    stage = OcrStage()

    expected = hashlib.sha256("구글검색\n스포위키".encode()).hexdigest()
    assert stage.config_subset(cfg)["watermark_patterns"] == expected

    write_patterns(user, "라이브스코어")  # the user file contributes too: the fingerprint follows it
    expected = hashlib.sha256("구글검색\n스포위키\n라이브스코어".encode()).hexdigest()
    assert stage.config_subset(cfg)["watermark_patterns"] == expected


# ---------------------------------------------------------------- sound effects in free text (M10)


def test_ocr_stage_turns_onomatopoeia_in_free_text_into_measured_sfx(cfg: Config) -> None:
    manga = cfg.model_copy(update={"ocr": MANGA_CFG})
    ctx = prepared(manga)
    write_mixed_regions(ctx)
    ctx.gpu = FakeScheduler(
        {
            "reader": FakeCropReader(
                [(KOREAN_LINES[0], 0.95), (KOREAN_LINES[1], 0.9), ("쿠구구궁!!", 0.9), ("쾅!", 0.9)]
            )
        }
    )

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.status == "done" and outcome.metrics["sfx"] == 2.0
    regions = RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions
    assert [r.kind for r in regions] == ["bubble_text", "bubble_text", "sfx", "sfx"]


def test_ocr_stage_sfx_detection_can_be_switched_off(cfg: Config) -> None:
    manga = cfg.model_copy(update={"ocr": MANGA_CFG, "sfx": cfg.sfx.model_copy(update={"detect": False})})
    ctx = prepared(manga)
    write_mixed_regions(ctx)
    ctx.gpu = FakeScheduler(
        {
            "reader": FakeCropReader(
                [(KOREAN_LINES[0], 0.95), (KOREAN_LINES[1], 0.9), ("쿠구구궁!!", 0.9), ("쾅!", 0.9)]
            )
        }
    )
    assert run_stage(OcrStage(), ctx).status == "done"
    regions = RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions
    assert [r.kind for r in regions] == ["bubble_text", "bubble_text", "free_text", "sfx"]


def test_config_subset_fingerprints_the_sfx_lexicon(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lexicon = tmp_path / "sfx_text.toml"
    lexicon.write_text('[sfx_text]\nko = ["쾅"]\n', encoding="utf-8")
    monkeypatch.setattr("omniscan.ocr.stage.default_sfx_text_paths", lambda: [lexicon])
    first = OcrStage().config_subset(cfg)["sfx"]
    lexicon.write_text('[sfx_text]\nko = ["쾅", "쿵"]\n', encoding="utf-8")
    assert OcrStage().config_subset(cfg)["sfx"] != first
    assert "mode" not in first  # the sfx mode only concerns inpaint and typeset


# ---------------------------------------------------------------- the whole-page sweep (ocr/sweep.py)


class FakeSweeper:
    """Scripted CRAFT: the same word boxes (tile pixels) in the first tile, none elsewhere."""

    def __init__(self, words: list[tuple[tuple[float, float, float, float], float]]) -> None:
        self.words = words
        self.calls = 0

    def detect(
        self, tiles: list[torch.Tensor]
    ) -> list[list[tuple[tuple[float, float, float, float], float]]]:
        self.calls += 1
        return [self.words if i == 0 else [] for i in range(len(tiles))]


class QueueReader:
    """Crop reader answering each call with the next scripted list of readings."""

    def __init__(self, *calls: list[tuple[str, float]]) -> None:
        self.calls = list(calls)
        self.sizes: list[int] = []

    def read(self, crops: list[torch.Tensor]) -> list[tuple[str, float]]:
        self.sizes.append(len(crops))
        return self.calls.pop(0)[: len(crops)]


def test_the_sweep_adds_the_effects_the_detector_missed(cfg: Config) -> None:
    swept = cfg.model_copy(update={"ocr": MANGA_CFG, "sfx": SfxConfig()})
    ctx = prepared(swept)  # one detected region at (10, 50, 390, 200)
    words = [
        ((60.0, 100.0, 200.0, 140.0), 0.9),  # inside the detected region: its own text
        ((40.0, 230.0, 120.0, 290.0), 0.95),  # an effect in the art
        ((200.0, 240.0, 260.0, 280.0), 0.9),  # something else in the art
    ]
    # the region, then (crop, bare letters) for each of the two words outside it
    reader = QueueReader([("대사", 0.95)], [("광!", 0.9), ("광!", 0.8), ("나무", 0.9), ("나무", 0.9)])
    ctx.gpu = FakeScheduler({"reader": reader, "sfx_sweeper": FakeSweeper(words)})

    outcome = run_stage(OcrStage(), ctx)

    assert outcome.status == "done"
    assert reader.sizes == [1, 4]
    assert outcome.metrics["sweep_candidates"] == 2.0 and outcome.metrics["sweep_sfx"] == 1.0
    assert outcome.metrics["sweep_unknown"] == 1.0 and outcome.metrics["sfx"] == 1.0
    regions = RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions
    assert [(r.id, r.kind) for r in regions] == [("r0001", "bubble_text"), ("r0002", "sfx")]
    effect = regions[1]
    assert effect.bbox == BBox(x0=40, y0=230, x1=120, y1=290)
    assert effect.text == "광!" and effect.ocr_alt is None  # 꽝, 쾅, 광... are equally close: kept as read
    assert effect.lines[0].engine == "ocr-rec-manga-ocr-2025+craft"


def test_the_sweep_stays_off_without_sfx_detection(cfg: Config) -> None:
    off = cfg.model_copy(update={"ocr": MANGA_CFG, "sfx": SfxConfig(detect=False)})
    ctx = prepared(off)
    sweeper = FakeSweeper([((40.0, 230.0, 120.0, 290.0), 0.95)])
    ctx.gpu = FakeScheduler({"reader": QueueReader([("대사", 0.95)]), "sfx_sweeper": sweeper})
    assert run_stage(OcrStage(), ctx).status == "done"
    assert sweeper.calls == 0

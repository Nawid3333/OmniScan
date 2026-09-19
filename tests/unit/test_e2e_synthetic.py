"""E1: the end-to-end golden test — the real pipeline on three synthetic Korean pages vs exact truth.

One module-scoped fixture runs all ten stages over `library/E2E/Chapter 1` (pages 001-003.jpg, seeds
101-103) with a dictionary fake answering every translation request, then each test checks one
property against the ground truth: stage bookkeeping, detection+OCR quality, page separation in the
text path, pixel integrity outside the text, English ink placement, and the export manifest. It
skips (never downloads) when the detector, OCR or LaMa weights are not cached on this machine.
"""

from __future__ import annotations

import difflib
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
import pytest
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image

from omniscan.core.config import REPO_ROOT, Config, GpuConfig, InpaintConfig, PathsConfig
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import BBox, ExportArtifact, FinalArtifact, Region, RegionsArtifact
from omniscan.llm.ollama import ChatResponse
from omniscan.pipeline.runner import PipelineResult
from omniscan.pipeline.stages import STAGE_ORDER
from omniscan.translate.prompts import TRANSLATEGEMMA_TEMPLATE
from tests.fixtures.korean_pages import KOREAN_LINES, RegionTruth, make_korean_page

pytestmark = pytest.mark.gpu

SERIES = "E2E"
CHAPTER = "Chapter 1"
SEEDS = (101, 102, 103)
PAGE_HEIGHT = 1400  # every truth box of page i is shifted down by i * this many rows in strip space

# TRANSLATIONS[KOREAN_LINES[i]] is the English for that line; every sentence distinct, "hyung" kept.
_SENTENCES: tuple[str, ...] = (
    "Are you okay? The dungeon opened!",
    "Hyung, run away quickly!",
    "That cannot possibly be true...",
    "That guy is an S-rank hunter.",
    "I will protect you now.",
    "What?! Say that once again!",
    "We must leave before the gate closes.",
    "It has been a while, Sungjin.",
    "What on earth is happening here?",
    "Be quiet. Someone is coming closer.",
    "This is only just the beginning.",
    "I promise I will come back.",
)
TRANSLATIONS: dict[str, str] = dict(zip(KOREAN_LINES, _SENTENCES, strict=True))


class DictionaryClient:
    """A fake ChatClient that answers every request from TRANSLATIONS and records the calls."""

    def __init__(self, translations: dict[str, str]) -> None:
        self.translations = translations
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        cloud: bool = False,
        format: dict[str, Any] | Literal["json"] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
        think: bool | None = None,
        max_retries: int = 5,
    ) -> ChatResponse:
        """Answer in the format the request expects: JSON for chat batches, plain text for translategemma."""
        self.calls.append({"model": model, "messages": messages, "format": format})
        if len(messages) == 1:  # translategemma: one user message ending with the source text
            source = messages[0]["content"].removeprefix(TRANSLATEGEMMA_TEMPLATE)
            content = self._english(source)
        else:  # chat_json: system message + user message listing the regions as JSON
            user = messages[1]["content"]
            regions = json.loads(user.split("Regions (reading order):\n", 1)[1])
            content = json.dumps(
                {"translations": [{"id": r["id"], "text": self._english(r["text"])} for r in regions]},
                ensure_ascii=False,
            )
        return ChatResponse(
            content=content,
            model=model,
            done=True,
            total_duration_ns=None,
            prompt_eval_count=10,
            eval_count=5,
            raw={},
        )

    def _english(self, source: str) -> str:
        """The English sentence whose Korean key is closest to `source` (OCR may misread a character)."""
        key = max(self.translations, key=lambda k: difflib.SequenceMatcher(None, source, k).ratio())
        return self.translations[key]


@dataclass(frozen=True, slots=True)
class Golden:
    """Everything the golden assertions share from the one pipeline run."""

    cfg: Config
    client: DictionaryClient
    result: PipelineResult
    result2: PipelineResult
    out: np.ndarray  # exported slices concatenated vertically, uint8 [4200, 800, 3]
    raw: np.ndarray  # the drawn pages with Korean text, uint8 [4200, 800, 3]
    clean: np.ndarray  # the drawn pages without any text, uint8 [4200, 800, 3]
    truth: list[tuple[RegionTruth, BBox]]  # truth region and its bbox in strip space
    matches: list[tuple[RegionTruth, BBox, Region]]  # truth regions matched to ocr regions
    wall: float  # wall-clock seconds of the first run (models included)
    calls_after_first: int  # chat requests the first run made


@pytest.fixture(scope="module")
def golden(tmp_path_factory: pytest.TempPathFactory) -> Golden:
    """Run the whole pipeline once on three synthetic pages and collect everything the tests compare."""
    pages = [make_korean_page(seed, n_bubbles=4, n_free=1, n_sfx=0) for seed in SEEDS]
    root = tmp_path_factory.mktemp("e2e_golden")
    cfg = Config(
        gpu=GpuConfig(device="auto"),
        paths=PathsConfig(
            library_root=root / "library",
            work_root=root / "work",
            output_root=root / "output",
            promo_examples=root / "promo",
            models_dir=_real_models_dir(),
        ),
    )
    raw_dir = cfg.paths.library_root / SERIES / CHAPTER
    raw_dir.mkdir(parents=True)
    for number, page in enumerate(pages, 1):
        page.image.save(raw_dir / f"{number:03d}.jpg", format="JPEG", quality=92)

    _skip_without_cached_weights(cfg)

    from omniscan.gpu.groups import build_vram_manager  # deferred: imports torch
    from omniscan.pipeline.runner import run_pipeline

    client = DictionaryClient(TRANSLATIONS)
    gpu = build_vram_manager(cfg)
    start = time.perf_counter()
    try:
        try:
            result = run_pipeline(cfg, SERIES, [CHAPTER], lama=True, client=client, gpu=gpu)
        except httpx.HTTPError as exc:  # a weight turned out to be missing (e.g. offline): skip
            pytest.skip(f"model weights not cached: {exc}")
        calls_after_first = len(client.calls)
        result2 = run_pipeline(cfg, SERIES, [CHAPTER], lama=True, client=client, gpu=gpu)
    finally:
        gpu.release()
    wall = time.perf_counter() - start

    paths = SeriesPaths.from_config(cfg, SERIES).chapter(CHAPTER)
    out = _stack_images(Image.open(path) for path in sorted(paths.output_dir.glob("*.jpg")))
    raw = _stack_images(page.image for page in pages)
    clean = _stack_images(page.clean for page in pages)
    failures = "; ".join(f"{c}: {e}" for c, e in result.failed.items()) or result.aborted or "none"
    assert out.shape == raw.shape, (
        f"slicing/export lost or added rows: out {out.shape} vs raw {raw.shape} (pipeline failures: {failures})"
    )
    ocr = RegionsArtifact.load(paths.artifact("ocr.json")).regions
    truth = [(region, _shifted(region.bbox, i)) for i, page in enumerate(pages) for region in page.regions]
    print(
        f"\nE2E pipeline wall: {wall:.1f}s, {len(client.calls)} chat requests, "
        f"{len(out)} rows, {len(ocr)} ocr regions"
    )
    return Golden(
        cfg=cfg,
        client=client,
        result=result,
        result2=result2,
        out=out,
        raw=raw,
        clean=clean,
        truth=truth,
        matches=_matched(truth, ocr),
        wall=wall,
        calls_after_first=calls_after_first,
    )


# ---------------------------------------------------------------- assertions


def test_all_ten_stages_done_then_skipped(golden: Golden) -> None:
    """The first run does every stage in order; the second skips all ten and makes no new chat requests."""
    first = {outcome.stage: outcome for outcome in golden.result.outcomes[CHAPTER]}
    assert list(first) == list(STAGE_ORDER)
    for stage in STAGE_ORDER:
        assert first[stage].status == "done", f"{stage}: {first[stage].error}"
    assert first["judge"].metrics["requests"] == 0.0  # all profiles agree, so the judge is never asked
    second = {outcome.stage: outcome for outcome in golden.result2.outcomes[CHAPTER]}
    assert [second[stage].status for stage in STAGE_ORDER] == ["skipped"] * len(STAGE_ORDER)
    assert len(golden.client.calls) == golden.calls_after_first


def test_detection_and_ocr_find_the_text(golden: Golden) -> None:
    """Most truth regions are found and the OCR text of the matched ones is close to the truth text."""
    assert golden.matches, "no truth region matched any ocr region"
    share = len(golden.matches) / len(golden.truth)
    errors = [_cer(truth.text.replace("\n", " "), ocr.text) for truth, _box, ocr in golden.matches]
    mean_cer = sum(errors) / len(errors)
    print(
        f"\ndetection/OCR: {len(golden.matches)}/{len(golden.truth)} truth regions matched"
        f" ({share:.0%}), mean CER {mean_cer:.4f}"
    )
    assert share >= 0.8
    assert mean_cer <= 0.10


def test_translations_follow_the_page_they_came_from(golden: Golden) -> None:
    """Every matched region's final line is the translation of its own page's truth sentence."""
    paths = SeriesPaths.from_config(golden.cfg, SERIES).chapter(CHAPTER)
    final = {line.region_id: line.text for line in FinalArtifact.load(paths.artifact("final.json")).lines}
    for truth, _box, ocr in golden.matches:
        expected = _english_for(truth)
        assert final[ocr.id] == expected, f"{ocr.id}: {final.get(ocr.id)!r} != {expected!r}"


def test_pages_stay_apart_and_clean_outside_the_text(golden: Golden) -> None:
    """Outside the truth text zones the export matches the text-free pages within JPEG error."""
    zone = _text_zone([box for _truth, box in golden.truth], golden.out.shape[:2])
    diff = np.abs(golden.out.astype(np.int16) - golden.clean.astype(np.int16)).max(axis=2)
    outside = diff[~zone]
    mean, p999 = float(outside.mean()), float(np.percentile(outside, 99.9))
    print(f"\noutside text zones: mean {mean:.2f}, p99.9 {p999:.1f}")
    # Measured on the golden run: mean 7.54, p99.9 31.0. The pipeline itself changes the strip by
    # only mean ~1.2 outside the zones (export re-encode at quality 95); the rest is the quality-92
    # input JPEG error of the fixture's noisy gradient background (measured mean 7.05 against the
    # in-memory pages) — so the mean threshold sits well above that floor, the p99.9 well below 60.
    assert mean <= 12.0
    assert p999 <= 60.0


def test_english_is_drawn_where_the_korean_was(golden: Golden) -> None:
    """Every matched region's box now holds English ink and no longer shows the Korean it had."""
    ink = np.abs(golden.out.astype(np.int16) - golden.clean.astype(np.int16)).max(axis=2) > 60
    changed = np.abs(golden.out.astype(np.int16) - golden.raw.astype(np.int16)).max(axis=2) > 60
    ink_fractions: list[float] = []
    changed_fractions: list[float] = []
    for _truth, box, _ocr in golden.matches:
        grown = _grown(box, 6, ink.shape)
        ink_fractions.append(float(ink[grown.y0 : grown.y1, grown.x0 : grown.x1].mean()))
        changed_fractions.append(float(changed[grown.y0 : grown.y1, grown.x0 : grown.x1].mean()))
    print(
        f"\nink fraction per matched region: min {min(ink_fractions):.3f}, max {max(ink_fractions):.3f};"
        f" changed-vs-raw: min {min(changed_fractions):.3f}, max {max(changed_fractions):.3f}"
    )
    assert all(0.02 <= fraction <= 0.60 for fraction in ink_fractions)
    assert all(fraction >= 0.05 for fraction in changed_fractions)


def test_export_json_matches_the_files(golden: Golden) -> None:
    """export.json lists exactly the slice files on disk, with their byte sizes, all non-zero."""
    paths = SeriesPaths.from_config(golden.cfg, SERIES).chapter(CHAPTER)
    export = ExportArtifact.load(paths.artifact("export.json"))
    listed = {item.name: item.bytes for item in export.files}
    on_disk = {path.name: path.stat().st_size for path in paths.output_dir.glob("*.jpg")}
    assert listed == on_disk
    assert listed and all(size > 0 for size in listed.values())


# ---------------------------------------------------------------- helpers


def _real_models_dir() -> Path:
    """The machine's real model cache: the repo's models dir, else the main checkout's (builder worktrees)."""
    candidates = [REPO_ROOT / "models"]
    if REPO_ROOT.parent.name.endswith("-wt"):
        main = REPO_ROOT.parent.parent / REPO_ROOT.parent.name.removesuffix("-wt")
        candidates.append(main / "models")
    for candidate in candidates:
        if (candidate / "lama" / InpaintConfig().lama_file).is_file():
            return candidate
    pytest.skip("LaMa weights not cached in the repo or main-checkout models dir")


def _skip_without_cached_weights(cfg: Config) -> None:
    """Skip the module (never download) when a detector, OCR or LaMa weight file is not cached."""
    for repo in (cfg.detect.repo, cfg.ocr.det_repo, cfg.ocr.rec_repo):
        try:
            snapshot_download(repo, local_files_only=True)
        except LocalEntryNotFoundError as exc:
            pytest.skip(f"{repo} not cached: {exc}")
    if not (cfg.paths.models_dir / "lama" / InpaintConfig().lama_file).is_file():
        pytest.skip(f"LaMa weights not cached in {cfg.paths.models_dir}")


def _stack_images(images: Iterable[Image.Image]) -> np.ndarray:
    """The images concatenated vertically (they share their width), uint8 [H, W, 3]."""
    return np.vstack([np.asarray(image.convert("RGB"), dtype=np.uint8) for image in images])


def _shifted(box: BBox, page_index: int) -> BBox:
    """A page-space box moved into strip space: page i occupies rows [i*1400, (i+1)*1400)."""
    dy = page_index * PAGE_HEIGHT
    return BBox(x0=box.x0, y0=box.y0 + dy, x1=box.x1, y1=box.y1 + dy)


def _grown(box: BBox, grow: int, shape: tuple[int, int]) -> BBox:
    """The box grown by `grow` px on every side, clipped to a [H, W] image."""
    return BBox(
        x0=max(0, box.x0 - grow),
        y0=max(0, box.y0 - grow),
        x1=min(shape[1], box.x1 + grow),
        y1=min(shape[0], box.y1 + grow),
    )


def _text_zone(boxes: Sequence[BBox], shape: tuple[int, int], grow: int = 24) -> np.ndarray:
    """Bool [H, W]: the union of `boxes` grown by `grow` px on every side, clipped to the image."""
    zone = np.zeros(shape, dtype=bool)
    for box in boxes:
        clipped = _grown(box, grow, shape)
        zone[clipped.y0 : clipped.y1, clipped.x0 : clipped.x1] = True
    return zone


def _matched(
    truth: Sequence[tuple[RegionTruth, BBox]], ocr: Sequence[Region]
) -> list[tuple[RegionTruth, BBox, Region]]:
    """Each truth region paired with the ocr region of highest IoU, kept when that IoU reaches 0.5."""
    matches: list[tuple[RegionTruth, BBox, Region]] = []
    for region, box in truth:
        best = max(ocr, key=lambda r: box.iou(r.bbox), default=None)
        if best is not None and box.iou(best.bbox) >= 0.5:
            matches.append((region, box, best))
    return matches


def _english_for(truth: RegionTruth) -> str:
    """The English sentence for a truth region: its wrapped lines collapse back to the source line."""
    sentence = " ".join(truth.text.split())
    if sentence not in TRANSLATIONS:
        pytest.fail(f"truth text {sentence!r} is not one of the translated KOREAN_LINES")
    return TRANSLATIONS[sentence]


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between `a` and `b` (copied from test_ocr_pipeline.py)."""
    if not a:
        return len(b)
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        prev = row[0]
        row[0] = i + 1
        for j, cb in enumerate(b):
            cur = row[j + 1]
            row[j + 1] = min(row[j] + 1, cur + 1, prev + (ca != cb))
            prev = cur
    return row[len(b)]


def _cer(truth: str, read: str) -> float:
    """Character error rate of `read` against `truth`, both with all whitespace removed (copied)."""
    a = "".join(truth.split())
    b = "".join(read.split())
    return _levenshtein(a, b) / len(a) if a else (1.0 if b else 0.0)

"""The card's one real-GPU test (GL1 acceptance 5): the full reference pass on synthetic chapters.

Three raw chapters (one synthetic Korean page each, real detect+OCR) against three imported
reference chapters — the same pages with official English lettering drawn into the same bubbles.
The real chapter matcher (CM1) pairs the chapters, the real pipeline OCRs both sides through the
pseudo series paths, and a scripted chat client answers the extraction prompts: the term all three
chapters agree on must land `locked` with `origin="reference"`, the term only two chapters agree on
stays `proposed`. Skips (never downloads) when the detector/OCR weights are not cached; LaMa is
never needed because the reference pass only runs ingest/slice/detect/ocr."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pytest
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image, ImageDraw, ImageFont

from omniscan.core.config import REPO_ROOT, Config, GpuConfig, PathsConfig
from omniscan.core.paths import SeriesPaths
from omniscan.glossary.reference import (
    DEFAULT_EXTRACTION_MODEL,
    ReferenceSummary,
    format_summary,
    run_reference,
)
from omniscan.glossary.store import GlossaryEntry, GlossaryStore
from omniscan.llm.ollama import ChatResponse
from tests.fixtures.korean_pages import KoreanPage, RegionTruth, font_path, make_korean_page

pytestmark = pytest.mark.gpu

SERIES = "REFE2E"
SEEDS = (303, 304, 306)  # measured: raw vs reference page dhash similarity 0.91/0.92/0.98 (>= 0.75)
RAW_CHAPTERS = ("Chapter 1", "Chapter 2", "Chapter 3")
REF_CHAPTERS = ("ch_01", "ch_02", "ch_03")
FONT_NAME = "NanumGothic-Regular.ttf"
ENGLISH_LINES = ("Run away, Minjun!", "The dungeon opened!", "That guy is an S-rank hunter.")

_FULL_REPLY = json.dumps(
    {
        "terms": [
            {"source": "민준", "target": "Minjun", "type": "person"},
            {"source": "던전", "target": "dungeon", "type": "place"},
            {"source": "헌터", "target": "hunter", "type": "rank"},
        ]
    },
    ensure_ascii=False,
)
_PARTIAL_REPLY = json.dumps(
    {
        "terms": [
            {"source": "민준", "target": "Minjun", "type": "person"},
            {"source": "헌터", "target": "hunter", "type": "rank"},
        ]
    },
    ensure_ascii=False,
)


class ScriptedClient:
    """Answers the extraction prompts in chapter order: chapters 1-2 agree on three terms, 3 on two."""

    def __init__(self) -> None:
        self.replies = [_FULL_REPLY, _FULL_REPLY, _PARTIAL_REPLY]
        self.prompts: list[str] = []

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
        self.prompts.append(messages[1]["content"])
        return ChatResponse(
            content=self.replies.pop(0),
            model=model,
            done=True,
            total_duration_ns=None,
            prompt_eval_count=None,
            eval_count=None,
            raw={},
        )

    def close(self) -> None:
        return None


def _wrap(text: str, font: ImageFont.FreeTypeFont, limit: float) -> list[str]:
    """Greedy word wrap so every line's advance width fits `limit`."""
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else f"{current} {word}"
        if not current or font.getlength(candidate) <= limit:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    centre: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    """Centred multi-line text (manual bbox centring: multiline anchor support varies by Pillow)."""
    box = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    width, height = box[2] - box[0], box[3] - box[1]
    draw.multiline_text(
        (centre[0] - width / 2 - box[0], centre[1] - height / 2 - box[1]),
        text,
        font=font,
        fill=fill,
        align="center",
    )


def _reference_page(page: KoreanPage) -> Image.Image:
    """The official English release of `page`: its clean art with English drawn into every region."""
    image = page.clean.copy()
    draw = ImageDraw.Draw(image)
    for index, truth in enumerate(page.regions):
        _draw_english(draw, truth, ENGLISH_LINES[index % len(ENGLISH_LINES)])
    return image


def _draw_english(draw: ImageDraw.ImageDraw, truth: RegionTruth, text: str) -> None:
    """`text` centred inside the region's bubble (or its ink box), shrunk until it fits."""
    box = truth.bubble_bbox or truth.bbox
    interior_w, interior_h = box.width - 28, box.height - 16
    for size in range(30, 11, -2):
        font = ImageFont.truetype(str(font_path(FONT_NAME)), size)
        lines = _wrap(text, font, interior_w)
        block = "\n".join(lines)
        ink = draw.multiline_textbbox((0, 0), block, font=font, align="center")
        if ink[2] - ink[0] <= interior_w and ink[3] - ink[1] <= interior_h:
            _draw_centered(
                draw,
                ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2),
                block,
                font,
                truth.text_color,
            )
            return
    raise ValueError(f"English {text!r} does not fit its bubble {box}")


def _models_dir() -> Path:
    """The machine's real model dir: the repo's models/, else the main checkout's (builder worktrees)."""
    candidates = [REPO_ROOT / "models"]
    if REPO_ROOT.parent.name.endswith("-wt"):
        main = REPO_ROOT.parent.parent / REPO_ROOT.parent.name.removesuffix("-wt")
        candidates.append(main / "models")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    pytest.skip("no cached models dir in the repo or main checkout")


def _skip_without_cached_weights(cfg: Config) -> None:
    """Skip (never download) when a detector or OCR weight file is not cached."""
    for repo in (cfg.detect.repo, cfg.ocr.det_repo, cfg.ocr.rec_repo):
        try:
            snapshot_download(repo, local_files_only=True)
        except LocalEntryNotFoundError as exc:
            pytest.skip(f"{repo} not cached: {exc}")


@pytest.fixture(scope="module")
def reference_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ReferenceSummary, list[GlossaryEntry], ScriptedClient, SeriesPaths]:
    """Build both chapter sets, run the whole reference pass once on the GPU, collect the results."""
    pages = [make_korean_page(seed, n_bubbles=3, n_free=0, n_sfx=0) for seed in SEEDS]
    root = tmp_path_factory.mktemp("reference_gpu")
    cfg = Config(
        gpu=GpuConfig(device="auto"),
        paths=PathsConfig(
            library_root=root / "library",
            work_root=root / "work",
            output_root=root / "output",
            promo_examples=root / "promo",
            models_dir=_models_dir(),
        ),
    )
    _skip_without_cached_weights(cfg)

    sp = SeriesPaths.from_config(cfg, SERIES)
    for name, page in zip(RAW_CHAPTERS, pages, strict=True):
        raw_dir = sp.library_dir / name
        raw_dir.mkdir(parents=True)
        page.image.save(raw_dir / "001.jpg", format="JPEG", quality=92)
    for name, page in zip(REF_CHAPTERS, pages, strict=True):
        ref_dir = sp.reference_dir / name
        ref_dir.mkdir(parents=True)
        _reference_page(page).save(ref_dir / "001.jpg", format="JPEG", quality=92)

    from omniscan.gpu.groups import build_vram_manager  # deferred: imports torch

    client = ScriptedClient()
    gpu = build_vram_manager(cfg)
    try:
        summary = run_reference(cfg, SERIES, client=client, model=DEFAULT_EXTRACTION_MODEL, gpu=gpu)
    finally:
        gpu.release()
    with GlossaryStore(sp.db) as store:
        rows = store.list()
    print("\n" + "\n".join(format_summary(summary)))
    return summary, rows, client, sp


def test_reference_pass_locks_three_chapter_agreement(
    reference_run: tuple[ReferenceSummary, list[GlossaryEntry], ScriptedClient, SeriesPaths],
) -> None:
    """민준→Minjun recurs in all three reference chapters: locked, origin=reference, count=3; the
    two-chapter term stays proposed; glossary.yaml is re-exported next to the store."""
    summary, rows, client, sp = reference_run
    assert summary.matched == tuple(zip(RAW_CHAPTERS, REF_CHAPTERS, strict=True))
    assert summary.raw_only == () and summary.reference_only == ()
    assert summary.ocr_failed == ()
    assert summary.chapters_extracted == 3
    assert len(client.prompts) == 3  # one extraction request per matched chapter pair

    by_source = {row.source: row for row in rows}
    minjun = by_source["민준"]
    assert minjun.target == "Minjun"
    assert (minjun.status, minjun.origin) == ("locked", "reference")
    assert minjun.count == 3 and minjun.first_seen_chapter == 1.0
    dungeon = by_source["던전"]  # only two chapters agreed: proposed, never dropped
    assert (dungeon.target, dungeon.status, dungeon.origin) == ("dungeon", "proposed", "reference")
    assert by_source["헌터"].status == "locked"
    assert len(rows) == 3
    assert sp.glossary_yaml.is_file()

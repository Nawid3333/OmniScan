"""The acquire run loop: extract → select pages → download → completeness, one chapter at a time."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Literal

import httpx

from omniscan.acquire.completeness import Finding, check_pages, inspect_chapter, verdict
from omniscan.acquire.download import AcquireError, ImageRef, download_chapter
from omniscan.acquire.drm import DrmPlatformError
from omniscan.acquire.extractor import AuthError, ExtractError, Extraction, Extractor, QuotaError
from omniscan.acquire.pagerun import PageSelection, select_pages
from omniscan.acquire.plan import AcquirePlan, PlanItem, read_accepted, write_accepted
from omniscan.core.paths import list_chapters

SELECTION_NAME = "acquire_selection.json"


@dataclass(frozen=True, slots=True)
class ChapterOutcome:
    """One chapter's acquire result: status, page count, charged credits, findings and error."""

    name: str
    status: Literal["ok", "review", "failed", "skipped"]
    pages: int
    credits: int  # credits charged for this chapter (0 for skipped/failed extractions)
    findings: tuple[Finding, ...]
    error: str | None


def acquire_chapters(
    plan: AcquirePlan,
    series_dir: Path,
    extractor: Extractor,
    *,
    client: httpx.Client | None = None,
    max_credits: int | None = None,
    page_run: bool = True,
    apply_filters: bool = True,
    on_event: Callable[[ChapterOutcome], None] | None = None,
) -> list[ChapterOutcome]:
    """Extract, download and completeness-check every plan item in order; per-chapter problems never raise."""
    outcomes: list[ChapterOutcome] = []
    spent = 0
    stop_reason: str | None = None
    for item in plan.items:
        chapter_dir = series_dir / item.source.name
        if item.state == "done":
            record = read_accepted(chapter_dir)
            pages = record.get("pages") if isinstance(record, dict) else None
            outcome = ChapterOutcome(
                item.source.name,
                "skipped",
                pages if isinstance(pages, int) else 0,
                0,
                (),
                "already acquired",
            )
        elif stop_reason is not None:
            outcome = ChapterOutcome(item.source.name, "skipped", 0, 0, (), stop_reason)
        elif max_credits is not None and spent + item.credits > max_credits:
            stop_reason = "credit budget reached"
            outcome = ChapterOutcome(item.source.name, "skipped", 0, 0, (), stop_reason)
        else:
            try:
                extraction = extractor.extract(item.source.url, mode=plan.mode)
            except (AuthError, QuotaError) as exc:
                outcome = ChapterOutcome(item.source.name, "failed", 0, 0, (), str(exc))
                stop_reason = f"stopped: {exc}"
            except (ExtractError, DrmPlatformError) as exc:
                outcome = ChapterOutcome(item.source.name, "failed", 0, 0, (), str(exc))
            else:
                spent += extraction.credits
                outcome = _download_and_check(
                    item,
                    series_dir,
                    chapter_dir,
                    extraction,
                    page_run=page_run,
                    apply_filters=apply_filters,
                    client=client,
                )
        outcomes.append(outcome)
        if on_event is not None:
            on_event(outcome)
    return outcomes


def _download_and_check(
    item: PlanItem,
    series_dir: Path,
    chapter_dir: Path,
    extraction: Extraction,
    *,
    page_run: bool,
    apply_filters: bool,
    client: httpx.Client | None,
) -> ChapterOutcome:
    """Select the page images, download them and record the completeness verdict for one chapter."""
    name = item.source.name
    urls = [image.url for image in extraction.images]
    selection = select_pages(urls) if page_run else PageSelection(kept=tuple(urls), dropped=())
    _write_selection(chapter_dir, extraction.page_url, selection)
    if not selection.kept:
        filtered = len(urls) - len(selection.kept)
        return ChapterOutcome(
            name,
            "failed",
            0,
            extraction.credits,
            (),
            f"no page images found ({filtered} image(s) on the page were filtered out)",
        )
    refs = [ImageRef(url=url, referer=item.source.url) for url in selection.kept]
    try:
        download_chapter(refs, chapter_dir, client=client, apply_filters=apply_filters)
    except (AcquireError, DrmPlatformError) as exc:
        return ChapterOutcome(name, "failed", 0, extraction.credits, (), str(exc))
    pages, findings = inspect_chapter(chapter_dir)
    findings = [
        *findings,
        *check_pages(
            pages,
            median_pages=_median_other(series_dir, name),
            previous_pages=_previous_pages(series_dir, name),
        ),
    ]
    result = verdict(findings)
    if result == "failed":
        error = next(finding.message for finding in findings if finding.level == "error")
        return ChapterOutcome(name, "failed", len(pages), extraction.credits, tuple(findings), error)
    write_accepted(chapter_dir, result, findings, len(pages), source_url=item.source.url)
    return ChapterOutcome(name, result, len(pages), extraction.credits, tuple(findings), None)


def _write_selection(chapter_dir: Path, page_url: str, selection: PageSelection) -> None:
    """Record kept and dropped image URLs in acquire_selection.json before any download starts."""
    record = {
        "page_url": page_url,
        "kept": list(selection.kept),
        "dropped": [{"url": url, "reason": reason} for url, reason in selection.dropped],
    }
    chapter_dir.mkdir(parents=True, exist_ok=True)
    (chapter_dir / SELECTION_NAME).write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _median_other(series_dir: Path, name: str) -> float | None:
    """Median accepted page count of the series' other chapters, when at least 3 of them have one."""
    counts: list[int] = []
    for chapter in list_chapters(series_dir):
        if chapter.name == name:
            continue
        record = read_accepted(chapter)
        pages = record.get("pages") if isinstance(record, dict) else None
        if isinstance(pages, int):
            counts.append(pages)
    return median(counts) if len(counts) >= 3 else None


def _previous_pages(series_dir: Path, name: str) -> int | None:
    """Accepted page count of the nearest earlier chapter in reading order, or None when there is none."""
    chapters = list_chapters(series_dir)
    index = next((position for position, chapter in enumerate(chapters) if chapter.name == name), None)
    if index is None:
        return None
    for chapter in reversed(chapters[:index]):
        record = read_accepted(chapter)
        pages = record.get("pages") if isinstance(record, dict) else None
        if isinstance(pages, int):
            return pages
    return None
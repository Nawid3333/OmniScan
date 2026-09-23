"""Computing and storing per-chapter story summaries (`omniscan story summarize`).

A manual, explicit command — never an extra pipeline stage (an unconditional LLM call per chapter
would break the cost-conscious default). Reading an existing summary back into a later chapter's
translate/judge prompt is free and automatic once a summary exists (see pipeline/stages.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omniscan.core.config import Config
from omniscan.core.paths import ChapterPaths, SeriesPaths, natural_key
from omniscan.core.schemas import FinalArtifact, RegionsArtifact
from omniscan.story.prompts import SUMMARY_SCHEMA, parse_summary_reply, summary_messages
from omniscan.story.store import SummaryStore
from omniscan.translate.run import ChatClient

DEFAULT_SUMMARY_MODEL = "gemma4:31b-cloud"


def chapter_final_lines(paths: ChapterPaths) -> list[str]:
    """The chapter's judged English lines in reading order; [] if final.json or ocr.json is missing."""
    final_path = paths.artifact("final.json")
    ocr_path = paths.artifact("ocr.json")
    if not final_path.is_file() or not ocr_path.is_file():
        return []
    order = {
        region.id: (region.slice_index, region.reading_order, natural_key(region.id))
        for region in RegionsArtifact.load(ocr_path).regions
    }
    usable: list[tuple[tuple[object, ...], str]] = [
        (order[line.region_id], line.text)
        for line in FinalArtifact.load(final_path).lines
        if line.text.strip() and line.region_id in order
    ]
    usable.sort(key=lambda pair: pair[0])
    return [text for _key, text in usable]


def summarize_chapter(client: ChatClient, model: str, lines: Sequence[str]) -> str | None:
    """One chat call over the chapter's lines; None when there are no lines or the reply is unusable."""
    if not lines:
        return None
    response = client.chat(
        model, summary_messages(lines), cloud=False, format=SUMMARY_SCHEMA, options={"temperature": 0.0}
    )
    return parse_summary_reply(response.content)


@dataclass(frozen=True, slots=True)
class SummarizeSummary:
    """What one summarise pass did (see `omniscan story summarize --json`)."""

    done: tuple[str, ...]  # chapters (re)computed this call
    skipped_no_final: tuple[str, ...]  # no final.json yet, or the model gave an unusable reply
    skipped_existing: tuple[str, ...]  # already had a summary and force=False


def run_summarize(
    cfg: Config,
    series: str,
    *,
    client: ChatClient,
    chapters: Sequence[str] | None = None,
    model: str = DEFAULT_SUMMARY_MODEL,
    force: bool = False,
) -> SummarizeSummary:
    """Summarise the chosen chapters' judged English text into the series db; see the module docstring."""
    sp = SeriesPaths.from_config(cfg, series)
    names = list(chapters) if chapters is not None else sp.chapters()
    sp.work_dir.mkdir(parents=True, exist_ok=True)
    done: list[str] = []
    skipped_no_final: list[str] = []
    skipped_existing: list[str] = []
    with SummaryStore(sp.db) as store:
        for chapter in names:
            if not force and store.get(chapter) is not None:
                skipped_existing.append(chapter)
                continue
            lines = chapter_final_lines(sp.chapter(chapter))
            if not lines:
                skipped_no_final.append(chapter)
                continue
            summary = summarize_chapter(client, model, lines)
            if summary is None:
                skipped_no_final.append(chapter)
                continue
            store.set(chapter, summary, model)
            done.append(chapter)
    return SummarizeSummary(
        done=tuple(done),
        skipped_no_final=tuple(skipped_no_final),
        skipped_existing=tuple(skipped_existing),
    )

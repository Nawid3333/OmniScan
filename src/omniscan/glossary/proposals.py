"""Glossary proposals from a chapter's own OCR text (`omniscan glossary propose`).

Most series have no official English release, so reference mode (`glossary/reference.py`) can
never run; this pass asks the chat model to propose terms from the raw Korean OCR text alone and
merges them into the store always as `status="proposed"`, `origin="llm"` — never auto-locked,
because there is no official translation to agree with. A human reviews them with
`omniscan glossary list --status proposed`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omniscan.core.config import Config
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import GlossaryEntry, RegionsArtifact
from omniscan.glossary.proposal_prompts import proposal_messages
from omniscan.glossary.reference import (
    AggregatedTerm,
    MergeReport,
    TermCandidate,
    aggregate,
    merge_into_store,
)
from omniscan.glossary.reference_prompts import TERMS_SCHEMA, parse_terms_reply
from omniscan.glossary.store import GlossaryStore
from omniscan.glossary.yaml_io import export_yaml
from omniscan.translate.languages import chapter_language
from omniscan.translate.prompts import source_text, translatable
from omniscan.translate.run import ChatClient

DEFAULT_PROPOSAL_MODEL = "gemma4:31b-cloud"
DEFAULT_MIN_CHAPTERS = 2  # an aggregated term must recur in at least this many chapters to be written at all
_NEVER_LOCK = 1_000_000  # above any realistic chapter count: proposals never auto-lock


def chapter_lines(sp: SeriesPaths, chapter: str) -> list[str]:
    """The chapter's OCR'd source lines worth extracting from, reading order; [] when ocr.json is missing."""
    path = sp.chapter(chapter).artifact("ocr.json")
    if not path.is_file():
        return []
    return [source_text(region) for region in translatable(RegionsArtifact.load(path).regions)]


def extract_proposals(
    client: ChatClient, model: str, chapter: str, lines: Sequence[str], lang: str = "ko"
) -> list[TermCandidate]:
    """[] for no lines; else one chat call over the chapter's lines -> its TermCandidate list."""
    if not lines:
        return []
    response = client.chat(
        model,
        proposal_messages(lines, lang),
        cloud=False,
        format=TERMS_SCHEMA,
        options={"temperature": 0.0},
    )
    return [
        TermCandidate(term.source, term.target, term.type, chapter)
        for term in parse_terms_reply(response.content)
    ]


@dataclass(frozen=True, slots=True)
class ProposalSummary:
    """What one proposal pass found and (unless dry) wrote (see `omniscan glossary propose --json`)."""

    chapters_scanned: tuple[str, ...]
    chapters_with_lines: int
    lines_scanned: int
    aggregated_above_threshold: int
    merge: MergeReport


def run_proposals(
    cfg: Config,
    series: str,
    *,
    client: ChatClient,
    chapters: Sequence[str] | None = None,
    model: str = DEFAULT_PROPOSAL_MODEL,
    min_chapters: int = DEFAULT_MIN_CHAPTERS,
    dry_run: bool = False,
) -> ProposalSummary:
    """Propose glossary terms from the chosen chapters' OCR text; see the module docstring."""
    sp = SeriesPaths.from_config(cfg, series)
    names = list(chapters) if chapters is not None else sp.chapters()
    candidates: list[TermCandidate] = []
    lines_scanned = 0
    chapters_with_lines = 0
    for chapter in names:
        lines = chapter_lines(sp, chapter)
        lines_scanned += len(lines)
        if not lines:
            continue
        chapters_with_lines += 1
        candidates.extend(
            extract_proposals(client, model, chapter, lines, lang=chapter_language(sp.chapter(chapter)))
        )
    aggregated = [term for term in aggregate(candidates) if term.chapters >= min_chapters]
    report = _merge_proposals(sp, aggregated, dry_run=dry_run)
    return ProposalSummary(
        chapters_scanned=tuple(names),
        chapters_with_lines=chapters_with_lines,
        lines_scanned=lines_scanned,
        aggregated_above_threshold=len(aggregated),
        merge=report,
    )


def _merge_proposals(sp: SeriesPaths, terms: Sequence[AggregatedTerm], *, dry_run: bool) -> MergeReport:
    """Merge the proposals (never auto-locked) into the store and re-export glossary.yaml."""
    if not dry_run:
        sp.work_dir.mkdir(parents=True, exist_ok=True)
        with GlossaryStore(sp.db) as store:
            report = merge_into_store(store, terms, min_locks=_NEVER_LOCK, write=True, origin="llm")
            export_yaml(store, sp.glossary_yaml)
        return report
    if not sp.db.is_file():
        return merge_into_store(_NullSink(), terms, min_locks=_NEVER_LOCK, write=False, origin="llm")
    with GlossaryStore(sp.db) as store:
        return merge_into_store(store, terms, min_locks=_NEVER_LOCK, write=False, origin="llm")


class _NullSink:
    """A read-free sink for dry runs on a series with no db yet: every term counts as new."""

    def find_by_source(self, source: str) -> GlossaryEntry | None:
        return None

    def add(self, entry: GlossaryEntry) -> GlossaryEntry:
        return entry

    def update(self, entry: GlossaryEntry) -> None:
        raise AssertionError("a dry run on a missing db never updates an entry")

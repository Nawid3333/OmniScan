"""Reference-mode glossary bootstrap (`omniscan reference`): learn the glossary from official chapters.

Given a series whose library has raw chapters and an official English translation of some of them
imported under `library_root/<series>/_reference_en/`, the command matches chapters by page art
(CM1's `match_chapters`), OCRs both sides through the normal pipeline stages, pairs the OCR'd regions
of every matched chapter pair (aligned pages, regions by reading-order rank), and asks the chat
model for the (source, target) terms each pair's text contains. A (source, target) pair that recurs
identically in at least `min_locks` (default 3, decision D7) distinct reference chapters is written
to the glossary as `status="locked"`, `origin="reference"`; everything else is written `proposed`
for human review, with the runner-up targets recorded in the notes. Entries that are already
`locked` are never silently overwritten: agreement raises their `count`, disagreement is flagged as
a conflict and the entry is left untouched — same for entries the owner `rejected`, and for
`origin="user"` rows.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from omniscan.core.config import Config, series_config
from omniscan.core.paths import REFERENCE_DIR, SeriesPaths, chapter_number, natural_key
from omniscan.core.schemas import (
    GlossaryEntry,
    IngestArtifact,
    Region,
    RegionsArtifact,
    SourceFile,
    TermType,
)
from omniscan.core.stage import GpuScheduler
from omniscan.glossary.reference_prompts import (
    TERMS_SCHEMA,
    PairedLine,
    parse_terms_reply,
    terms_messages,
)
from omniscan.glossary.store import GlossaryStore
from omniscan.glossary.yaml_io import export_yaml
from omniscan.match.chapters import ChapterMapping, PagePair, match_chapters
from omniscan.pipeline.runner import PipelineResult, run_pipeline
from omniscan.translate.run import ChatClient

DEFAULT_MIN_LOCKS = 3  # decision D7: auto-lock terms consistent across >= 3 reference chapters
DEFAULT_EXTRACTION_MODEL = "gemma4:31b-cloud"  # strongest shipped chat_json model (proxied locally)
MAX_REGION_COUNT_DIFF = 1  # page pairs whose region counts differ by more than this are skipped
MAX_SOURCE_CHARS = 30  # a longer "term" is a sentence the model mislabelled, not a term
MAX_TARGET_CHARS = 40
_OCR_STAGES = ("ingest", "slice", "detect", "ocr")


# ---------------------------------------------------------------- pairing OCR'd regions


@dataclass(frozen=True, slots=True)
class PairingResult:
    """The paired lines of one matched chapter pair, plus what could not be paired confidently."""

    lines: tuple[PairedLine, ...]
    pages_paired: int
    pages_skipped: int  # aligned page pairs whose region counts differed too much


def _region_page(region: Region, files: Sequence[SourceFile]) -> int | None:
    """The page (SourceFile.index) holding most of the region's vertical span; None when none overlaps."""
    best: int | None = None
    best_overlap = 0
    for file in files:
        overlap = min(region.bbox.y1, file.y1) - max(region.bbox.y0, file.y0)
        if overlap > best_overlap:
            best, best_overlap = file.index, overlap
    return best


def _regions_by_page(regions: Sequence[Region], files: Sequence[SourceFile]) -> dict[int, list[Region]]:
    """Regions grouped by their page (strip-space bbox vs the ingest layout), each page in reading order."""
    by_page: dict[int, list[Region]] = {}
    for region in regions:
        page = _region_page(region, files)
        if page is not None:
            by_page.setdefault(page, []).append(region)
    for page_regions in by_page.values():
        page_regions.sort(key=lambda region: (region.reading_order, natural_key(region.id)))
    return by_page


def pair_chapter(
    regions_a: Sequence[Region],
    regions_b: Sequence[Region],
    ingest_a: IngestArtifact,
    ingest_b: IngestArtifact,
    pages: Sequence[PagePair],
    *,
    max_count_diff: int = MAX_REGION_COUNT_DIFF,
) -> PairingResult:
    """Pair the OCR'd regions of one matched chapter pair: aligned pages, regions by reading-order rank.

    The same panel layout re-lettered keeps its bubble order and usually its bubble count, so ranks
    pair; a page pair whose region counts differ by more than `max_count_diff` is skipped rather
    than paired by guessing — beyond one stray detection the two sides' regionations diverged and
    rank pairing would pair unrelated lines. Watermarked and empty regions never pair."""
    by_page_a = _regions_by_page(_pairable(regions_a), ingest_a.files)
    by_page_b = _regions_by_page(_pairable(regions_b), ingest_b.files)
    lines: list[PairedLine] = []
    skipped = 0
    for page in pages:
        side_a = by_page_a.get(page.page_a, [])
        side_b = by_page_b.get(page.page_b, [])
        if abs(len(side_a) - len(side_b)) > max_count_diff:
            skipped += 1
            continue
        for region_a, region_b in zip(side_a, side_b, strict=False):
            source, target = _collapse(region_a.text), _collapse(region_b.text)
            if source and target:
                lines.append(PairedLine(source, target, page.page_a, page.page_b))
    return PairingResult(tuple(lines), len(pages) - skipped, skipped)


def _pairable(regions: Sequence[Region]) -> list[Region]:
    """The regions worth pairing: not watermarks, with text."""
    return [region for region in regions if region.kind != "watermark" and region.text.strip()]


def _collapse(text: str) -> str:
    """Every run of whitespace (newlines included) collapsed to one space."""
    return " ".join(text.split())


# ---------------------------------------------------------------- extraction and aggregation


@dataclass(frozen=True, slots=True)
class TermCandidate:
    """One (source, target) term extracted from one matched chapter pair's paired text."""

    source: str
    target: str
    type: TermType
    chapter: str  # the raw chapter folder name the pair came from


def extract_candidates(
    client: ChatClient, model: str, chapter: str, lines: Sequence[PairedLine]
) -> list[TermCandidate]:
    """Ask the chat model for the terms of one chapter's paired lines (no request without lines)."""
    if not lines:
        return []
    response = client.chat(
        model, terms_messages(lines), cloud=False, format=TERMS_SCHEMA, options={"temperature": 0.0}
    )
    return [
        TermCandidate(term.source, term.target, term.type, chapter)
        for term in parse_terms_reply(response.content)
    ]


@dataclass(frozen=True, slots=True)
class AggregatedTerm:
    """One source term's cross-chapter aggregation: the strongest target plus the runner-ups."""

    source: str
    target: str
    type: TermType
    chapters: int  # distinct reference chapters agreeing on this (source, target) pair
    occurrences: int  # total extracted occurrences of the pair (extraction is chapter-level)
    first_seen_chapter: float | None  # earliest raw chapter number the source was seen in
    alternatives: tuple[str, ...]  # other targets seen for the same source, strongest first


def aggregate(candidates: Sequence[TermCandidate]) -> list[AggregatedTerm]:
    """Fold candidates into one AggregatedTerm per source: agreeing targets counted, runner-ups kept.

    Sources compare after whitespace collapse (Korean has no case); targets compare case-folded,
    because "Minjun"/"MINJUN" are one term with different casing conventions — the stored target is
    the most frequent original spelling. A source longer than MAX_SOURCE_CHARS or a target longer
    than MAX_TARGET_CHARS is dropped as a mislabelled line, not a term."""
    by_source: dict[str, list[TermCandidate]] = {}
    for candidate in candidates:
        if _usable(candidate.source, candidate.target):
            by_source.setdefault(_collapse(candidate.source), []).append(candidate)
    aggregated: list[AggregatedTerm] = []
    for source, group in by_source.items():
        variants: dict[str, list[TermCandidate]] = {}
        for candidate in group:
            variants.setdefault(_target_key(candidate.target), []).append(candidate)
        ordered = sorted(variants.values(), key=_variant_rank)
        winner = ordered[0]
        aggregated.append(
            AggregatedTerm(
                source=source,
                target=_most_common(winner),
                type=_most_common_type(winner),
                chapters=len({candidate.chapter for candidate in winner}),
                occurrences=len(winner),
                first_seen_chapter=_first_seen(group),
                alternatives=tuple(dict.fromkeys(_most_common(variant) for variant in ordered[1:])),
            )
        )
    return sorted(aggregated, key=lambda term: term.source)


def _usable(source: str, target: str) -> bool:
    return len(_collapse(source)) <= MAX_SOURCE_CHARS and len(_collapse(target)) <= MAX_TARGET_CHARS


def _target_key(target: str) -> str:
    """The case-folded, whitespace-collapsed form two targets must share to count as the same term."""
    return _collapse(target).casefold()


def _variant_rank(group: list[TermCandidate]) -> tuple[int, int, str]:
    """Most agreeing chapters first, then most occurrences, then alphabetical target spelling."""
    return (-len({candidate.chapter for candidate in group}), -len(group), group[0].target.casefold())


def _most_common(group: list[TermCandidate]) -> str:
    """The group's most frequent original target spelling (ties: first seen)."""
    return Counter(candidate.target for candidate in group).most_common(1)[0][0]


def _most_common_type(group: list[TermCandidate]) -> TermType:
    return Counter(candidate.type for candidate in group).most_common(1)[0][0]


def _first_seen(group: Sequence[TermCandidate]) -> float | None:
    numbers = [number for candidate in group if (number := chapter_number(candidate.chapter)) is not None]
    return min(numbers) if numbers else None


# ---------------------------------------------------------------- merging into the store


class GlossarySink(Protocol):
    """The part of `GlossaryStore` the merge needs (dry runs substitute a no-op sink)."""

    def find_by_source(self, source: str) -> GlossaryEntry | None: ...

    def add(self, entry: GlossaryEntry) -> GlossaryEntry: ...

    def update(self, entry: GlossaryEntry) -> None: ...


@dataclass(frozen=True, slots=True)
class Conflict:
    """A glossary entry the reference extraction disagreed with (flagged, never touched)."""

    source: str
    existing_target: str
    extracted_target: str
    existing_status: str  # "locked" | "proposed" (a human-proposed row) | "rejected"


@dataclass(frozen=True, slots=True)
class MergeReport:
    """What one merge did (or would do, on a dry run) to the series' glossary."""

    locked: int  # entries with status locked after the merge (new, updated, or pre-existing+agreeing)
    proposed: int  # entries this pass wrote/updated as proposed (pre-existing proposed included)
    conflicts: tuple[Conflict, ...]
    rejected: tuple[str, ...]  # sources still extracted although the owner rejected them


def merge_into_store(
    store: GlossarySink,
    terms: Sequence[AggregatedTerm],
    *,
    min_locks: int = DEFAULT_MIN_LOCKS,
    write: bool = True,
) -> MergeReport:
    """Write the aggregated terms into the store; existing `locked` entries are never overwritten.

    Agreement with a locked entry raises its `count` (never lower — idempotent under re-runs);
    disagreement is a conflict and leaves the entry untouched. Rejected entries are left alone.
    An `origin="user"` proposed entry keeps its human target: agreement bumps only its count.
    `write=False` computes the same report without touching the store (dry runs)."""
    locked = proposed = 0
    conflicts: list[Conflict] = []
    rejected: list[str] = []
    for term in terms:
        want: Literal["locked", "proposed"] = "locked" if term.chapters >= min_locks else "proposed"
        existing = store.find_by_source(term.source)
        if existing is None:
            if write:
                store.add(
                    GlossaryEntry(
                        source=term.source,
                        target=term.target,
                        type=term.type,
                        status=want,
                        origin="reference",
                        first_seen_chapter=term.first_seen_chapter,
                        count=term.occurrences,
                        notes=_entry_notes(term),
                    )
                )
            if want == "locked":
                locked += 1
            else:
                proposed += 1
        elif existing.status == "locked":
            if _target_key(existing.target) == _target_key(term.target):
                if write and term.occurrences > existing.count:
                    store.update(existing.model_copy(update={"count": term.occurrences}))
                locked += 1
            else:
                conflicts.append(Conflict(term.source, existing.target, term.target, "locked"))
        elif existing.status == "rejected":
            rejected.append(term.source)
        elif existing.origin == "user":
            if _target_key(existing.target) == _target_key(term.target):
                if write and term.occurrences > existing.count:
                    store.update(existing.model_copy(update={"count": term.occurrences}))
                proposed += 1
            else:
                conflicts.append(Conflict(term.source, existing.target, term.target, "proposed"))
        else:  # proposed from an earlier machine pass: update it in place, never duplicate the row
            if write:
                store.update(
                    existing.model_copy(
                        update={
                            "target": term.target,
                            "type": term.type,
                            "status": want,
                            "origin": "reference",
                            "first_seen_chapter": term.first_seen_chapter,
                            "count": term.occurrences,
                            "notes": _entry_notes(term, replaced=existing.target),
                        }
                    )
                )
            if want == "locked":
                locked += 1
            else:
                proposed += 1
    return MergeReport(
        locked=locked,
        proposed=proposed,
        conflicts=tuple(conflicts),
        rejected=tuple(rejected),
    )


def _entry_notes(term: AggregatedTerm, *, replaced: str | None = None) -> str | None:
    """The human-readable notes for a reference entry: runner-up targets and a replaced target."""
    parts: list[str] = []
    if term.alternatives:
        parts.append("other targets seen: " + ", ".join(term.alternatives))
    if replaced is not None:
        parts.append(f"replaced target {replaced!r}")
    return "; ".join(parts) or None


class _NullSink:
    """A read-free sink for dry runs on a series with no db yet: every term counts as new."""

    def find_by_source(self, source: str) -> GlossaryEntry | None:
        return None

    def add(self, entry: GlossaryEntry) -> GlossaryEntry:
        return entry

    def update(self, entry: GlossaryEntry) -> None:
        raise AssertionError("a dry run on a missing db never updates an entry")


# ---------------------------------------------------------------- orchestration


@dataclass(frozen=True, slots=True)
class ReferenceSummary:
    """What one reference pass found and wrote (see `format_summary`)."""

    matched: tuple[tuple[str, str], ...]  # (raw chapter, reference chapter) folder names
    raw_only: tuple[str, ...]
    reference_only: tuple[str, ...]
    chapters_extracted: int  # matched pairs whose paired text was extracted
    pages_skipped: int
    lines_paired: int
    merge: MergeReport
    ocr_failed: tuple[tuple[str, str, str], ...]  # (side, chapter, error)
    dry_run: bool


def run_reference(
    cfg: Config,
    series: str,
    *,
    client: ChatClient,
    model: str = DEFAULT_EXTRACTION_MODEL,
    min_locks: int = DEFAULT_MIN_LOCKS,
    dry_run: bool = False,
    force: bool = False,
    gpu: GpuScheduler | None = None,
) -> ReferenceSummary:
    """Match chapters, OCR both sides, extract and lock glossary terms; see the module docstring."""
    # merge the REAL series' series.toml once, here: run_pipeline re-merges against whatever series
    # name it is given, and the pseudo series (below) has no series.toml — so without this merge the
    # reference side would silently OCR with the base config's engine (the same trap cli.py's
    # _run_stages and queue/executor.py each hit once; asserted in test_reference_cli.py)
    cfg = series_config(cfg, SeriesPaths.from_config(cfg, series).library_dir)
    sp = SeriesPaths.from_config(cfg, series)
    # the pseudo series name carries the reference folder: `Path` joins the embedded "/" as a normal
    # separator, so its library_dir IS sp.reference_dir and its work artifacts land under
    # work_root/<series>/_reference_en/ (asserted in test_reference_cli.py and by the GPU e2e test)
    ref_sp = SeriesPaths.from_config(cfg, f"{series}/{REFERENCE_DIR}")
    mapping = match_chapters(sp.library_dir, sp.reference_dir)
    candidates: list[TermCandidate] = []
    lines_paired = pages_skipped = chapters_extracted = 0
    ocr_failed: list[tuple[str, str, str]] = []
    if mapping.matched:
        raw_ok, raw_bad = _ocr(cfg, series, [m.a for m in mapping.matched], gpu=gpu, force=force)
        ref_ok, ref_bad = _ocr(
            cfg, f"{series}/{REFERENCE_DIR}", [m.b for m in mapping.matched], gpu=gpu, force=force
        )
        ocr_failed = [("raw", name, raw_bad[name]) for name in raw_bad]
        ocr_failed += [("reference", name, ref_bad[name]) for name in ref_bad]
        for match in mapping.matched:
            if match.a not in raw_ok or match.b not in ref_ok:
                continue
            pairing = pair_chapter(
                _ocr_regions(sp, match.a),
                _ocr_regions(ref_sp, match.b),
                _ingest(sp, match.a),
                _ingest(ref_sp, match.b),
                match.pages,
            )
            lines_paired += len(pairing.lines)
            pages_skipped += pairing.pages_skipped
            if pairing.lines:
                candidates.extend(extract_candidates(client, model, match.a, pairing.lines))
                chapters_extracted += 1
    report = _merge(
        cfg,
        sp,
        aggregate(candidates),
        min_locks=min_locks,
        dry_run=dry_run,
    )
    return ReferenceSummary(
        matched=tuple((m.a, m.b) for m in mapping.matched),
        raw_only=tuple(mapping.unmatched_a),
        reference_only=tuple(mapping.unmatched_b),
        chapters_extracted=chapters_extracted,
        pages_skipped=pages_skipped,
        lines_paired=lines_paired,
        merge=report,
        ocr_failed=tuple(ocr_failed),
        dry_run=dry_run,
    )


def _ocr(
    cfg: Config,
    series: str,
    chapters: Sequence[str],
    *,
    gpu: GpuScheduler | None,
    force: bool,
) -> tuple[set[str], dict[str, str]]:
    """Run the OCR stages over `chapters`; returns (chapters with a written ocr.json, failures)."""
    result: PipelineResult = run_pipeline(
        cfg, series, list(chapters), stages=list(_OCR_STAGES), gpu=gpu, force=force
    )
    return set(chapters) - set(result.failed), dict(result.failed)


def _ocr_regions(sp: SeriesPaths, chapter: str) -> list[Region]:
    """The chapter's OCR'd regions (empty when its ocr.json is missing, e.g. all detections dropped)."""
    path = sp.chapter(chapter).artifact("ocr.json")
    return RegionsArtifact.load(path).regions if path.is_file() else []


def _ingest(sp: SeriesPaths, chapter: str) -> IngestArtifact:
    """The chapter's ingest layout (the region -> page mapping needs it)."""
    return IngestArtifact.load(sp.chapter(chapter).artifact("ingest.json"))


def _merge(
    cfg: Config,
    sp: SeriesPaths,
    terms: Sequence[AggregatedTerm],
    *,
    min_locks: int,
    dry_run: bool,
) -> MergeReport:
    """Merge into the store and re-export glossary.yaml; a dry run computes the same report read-only."""
    if not dry_run:
        sp.work_dir.mkdir(parents=True, exist_ok=True)
        with GlossaryStore(sp.db) as store:
            report = merge_into_store(store, terms, min_locks=min_locks)
            export_yaml(store, sp.glossary_yaml)
        return report
    if not sp.db.is_file():
        return merge_into_store(_NullSink(), terms, min_locks=min_locks, write=False)
    with GlossaryStore(sp.db) as store:
        return merge_into_store(store, terms, min_locks=min_locks, write=False)


def format_summary(summary: ReferenceSummary) -> list[str]:
    """Human-readable summary lines for the CLI: counts, unmatched chapters, failures, conflicts."""
    lines = [
        f"reference: {len(summary.matched)} matched chapter pair(s), {len(summary.raw_only)} raw-only,"
        f" {len(summary.reference_only)} reference-only, {summary.lines_paired} paired line(s),"
        f" {summary.pages_skipped} page pair(s) skipped"
    ]
    lines.extend(f"reference:   raw chapter without reference match: {name}" for name in summary.raw_only)
    lines.extend(
        f"reference:   reference chapter without raw match: {name}" for name in summary.reference_only
    )
    lines.extend(
        f"reference:   {side} chapter {name} OCR failed: {error}"
        for side, name, error in summary.ocr_failed
    )
    lines.append(f"reference: {summary.merge.locked} term(s) locked, {summary.merge.proposed} proposed")
    lines.extend(
        f"reference:   conflict: {conflict.source}: store has {conflict.existing_target!r}"
        f" ({conflict.existing_status}), reference extraction says {conflict.extracted_target!r}"
        " — entry left untouched"
        for conflict in summary.merge.conflicts
    )
    lines.extend(
        f"reference:   rejected entry still extracted (left untouched): {name}"
        for name in summary.merge.rejected
    )
    if summary.dry_run:
        lines.append("reference: dry run — nothing written")
    return lines
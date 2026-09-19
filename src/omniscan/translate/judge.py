"""The judge: choose / merge / rewrite between candidate translations per region.

To save tokens the judge model is asked only about regions whose candidates disagree or violate a
locked glossary term; agreeing regions are picked deterministically. Answers that still break a
locked term get a repair round (`max_repair_rounds`). Reply parsing is as tolerant as the
translation runner's (cloud models fence, add prose, omit ids); anything unresolved falls back to
the deterministic candidate choice. Client errors propagate unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from omniscan.core.schemas import FinalLine, GlossaryEntry, Region
from omniscan.llm.ollama import ChatResponse
from omniscan.translate.agree import candidates_agree, normalize_line
from omniscan.translate.judge_config import JudgeConfig
from omniscan.translate.judge_prompts import JUDGE_SCHEMA, JudgeItem, judge_messages, label_for
from omniscan.translate.parse import extract_list
from omniscan.translate.postcheck import check_locked_terms
from omniscan.translate.prompts import source_text, translatable
from omniscan.translate.run import ChatClient

_RATIONALE_MAX = 300  # the reply's rationale, whitespace-collapsed and cut to this many characters

# (analysis, previous, problems) per judged region: round 1 has no rejected answer (None), a repair
# round carries the rejected text (or "" when the answer was invalid/missing) and what was wrong.
type _Triple = tuple[_Analysis, str | None, tuple[str, ...]]


@dataclass(slots=True)
class JudgeStats:
    """Counters over one `judge_regions` call (all int except `seconds`)."""

    regions: int
    judged: int
    auto_picked: int
    untranslated: int
    violations_left: int
    requests: int
    repair_requests: int
    prompt_tokens: int
    completion_tokens: int
    seconds: float


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One candidate translation of one region: its label, source run and text."""

    label: str
    run_id: str
    text: str


@dataclass(slots=True)
class _Analysis:
    """The candidates of one region: every run's text in priority order and the unique ones labelled."""

    region: Region
    all_candidates: list[_Candidate]
    unique: list[_Candidate]


@dataclass(slots=True)
class _Resolution:
    """What the judge model decided for one region in one round."""

    decision: Literal["pick", "merge", "rewrite"]
    text: str
    sources: list[str]
    rationale: str


class _Usage:
    """Token/request counters over the requests made in one `judge_regions` call."""

    __slots__ = ("completion_tokens", "prompt_tokens", "repair_requests", "requests")

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.requests = 0
        self.repair_requests = 0

    def record(self, response: ChatResponse, *, repair: bool) -> None:
        self.prompt_tokens += response.prompt_eval_count or 0
        self.completion_tokens += response.eval_count or 0
        self.requests += 1
        if repair:
            self.repair_requests += 1


def judge_regions(
    client: ChatClient,
    cfg: JudgeConfig,
    regions: Sequence[Region],
    runs: Mapping[str, Mapping[str, str]],
    entries: Sequence[GlossaryEntry],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[FinalLine], JudgeStats]:
    """Judge the chapter's translatable regions into final lines, asking the model only where needed."""
    start = clock()
    targets = translatable(regions)
    analyses = [_analysis(region, runs, cfg) for region in targets]

    decided: dict[str, FinalLine] = {}
    judged: list[_Analysis] = []
    auto_picked = untranslated = 0
    for analysis in analyses:
        line = _auto_line(analysis, cfg, entries)
        if line is None:
            judged.append(analysis)
            continue
        if line.decision == "manual":
            untranslated += 1
        else:
            auto_picked += 1
        decided[analysis.region.id] = line

    usage = _Usage()

    def chat(triples: Sequence[_Triple], *, repair: bool) -> dict[str, _Resolution | None]:
        """One judge request over `triples` and the resolutions (or unresolved marks) it produced."""
        items = [
            JudgeItem(
                region=a.region,
                candidates={c.label: c.text for c in a.unique},
                previous=previous,
                problems=problems,
            )
            for a, previous, problems in triples
        ]
        response = client.chat(
            cfg.model,
            judge_messages(items, entries),
            cloud=(cfg.endpoint == "cloud"),
            format=JUDGE_SCHEMA,
            options={"temperature": cfg.temperature},
            think=cfg.think,
        )
        usage.record(response, repair=repair)
        return _resolve([a for a, _, _ in triples], response.content)

    resolutions: dict[str, _Resolution | None] = {}
    first_round: list[_Triple] = [(a, None, ()) for a in judged]
    for chunk in _chunks(first_round, cfg.chunk_regions):
        resolutions.update(chat(chunk, repair=False))
    repair_rounds_used = 0
    pending = _repair_triples(judged, resolutions, entries)
    while pending and repair_rounds_used < cfg.max_repair_rounds:
        for chunk in _chunks(pending, cfg.chunk_regions):
            resolutions.update(chat(chunk, repair=True))
        repair_rounds_used += 1
        pending = _repair_triples(judged, resolutions, entries)

    for analysis in judged:
        decided[analysis.region.id] = _final_line(analysis, resolutions.get(analysis.region.id), entries)
    lines = [decided[region.id] for region in targets]
    return lines, JudgeStats(
        regions=len(targets),
        judged=len(judged),
        auto_picked=auto_picked,
        untranslated=untranslated,
        violations_left=sum(1 for line in lines if "glossary_violation" in line.flags),
        requests=usage.requests,
        repair_requests=usage.repair_requests,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        seconds=clock() - start,
    )


def _analysis(region: Region, runs: Mapping[str, Mapping[str, str]], cfg: JudgeConfig) -> _Analysis:
    """The candidates of one region: runs in priority order, unique ones labelled A, B, ..."""
    preferred = set(cfg.prefer)
    priority = [run_id for run_id in dict.fromkeys(cfg.prefer) if run_id in runs]
    priority += sorted(run_id for run_id in runs if run_id not in preferred)
    found: list[_Candidate] = []
    for run_id in priority:
        text = runs[run_id].get(region.id, "")
        if not text.strip():
            continue
        found.append(_Candidate(label="", run_id=run_id, text=text))
    unique: list[_Candidate] = []
    keys: set[str] = set()
    for candidate in found:
        key = normalize_line(candidate.text) or candidate.text
        if key in keys:
            continue
        keys.add(key)
        unique.append(_Candidate(label_for(len(unique)), candidate.run_id, candidate.text))
    return _Analysis(region=region, all_candidates=found, unique=unique)


def _auto_line(analysis: _Analysis, cfg: JudgeConfig, entries: Sequence[GlossaryEntry]) -> FinalLine | None:
    """The deterministic line for a region that needs no judge request, or None when it is judged."""
    if not analysis.all_candidates:
        return FinalLine(
            region_id=analysis.region.id,
            text="",
            decision="manual",
            sources=[],
            rationale="no candidate",
            flags=["untranslated"],
        )
    if cfg.always_judge and len(analysis.unique) >= 2:
        return None
    if not candidates_agree([c.text for c in analysis.all_candidates], cfg.agree_threshold):
        return None
    source = source_text(analysis.region)
    clean = [c for c in analysis.all_candidates if not check_locked_terms(source, c.text, entries)]
    if not clean:
        return None
    first = clean[0]
    return FinalLine(
        region_id=analysis.region.id,
        text=first.text,
        decision="pick",
        sources=[first.run_id],
        rationale="candidates agree" if len(analysis.all_candidates) >= 2 else "single candidate",
        flags=[],
    )


def _resolve(analyses: Sequence[_Analysis], content: str) -> dict[str, _Resolution | None]:
    """Per-region resolutions of one reply; regions the reply did not resolve map to None."""
    wanted = {a.region.id for a in analyses}
    by_id: dict[str, dict[str, Any]] = {}  # first entry per id wins, unknown/malformed ids ignored
    for item in extract_list(content, "judgements") or []:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or item_id not in wanted or item_id in by_id:
            continue
        by_id[item_id] = item
    return {
        a.region.id: _resolution(by_id[a.region.id], a) if a.region.id in by_id else None for a in analyses
    }


def _resolution(item: Mapping[str, Any], analysis: _Analysis) -> _Resolution | None:
    """One reply entry resolved against the item's candidates, or None when it does not resolve."""
    rationale_value = item.get("rationale")
    rationale = " ".join(rationale_value.split())[:_RATIONALE_MAX] if isinstance(rationale_value, str) else ""
    decision = item.get("decision")
    if decision == "pick" and isinstance(item.get("pick"), str):
        for candidate in analysis.unique:
            if candidate.label == item["pick"]:
                return _Resolution("pick", candidate.text, [candidate.run_id], rationale)
        return None
    text = item.get("text")
    if decision in ("merge", "rewrite") and isinstance(text, str) and text.strip():
        sources = [c.run_id for c in analysis.unique] if decision == "merge" else []
        return _Resolution(decision, " ".join(text.split()), sources, rationale)
    return None


def _repair_triples(
    analyses: Sequence[_Analysis],
    resolutions: Mapping[str, _Resolution | None],
    entries: Sequence[GlossaryEntry],
) -> list[_Triple]:
    """The judged items needing another round: unresolved ones and resolved-but-violating ones."""
    triples: list[_Triple] = []
    for analysis in analyses:
        resolution = resolutions.get(analysis.region.id)
        if resolution is None:
            triples.append((analysis, "", ("the previous answer was invalid or missing",)))
            continue
        violations = check_locked_terms(source_text(analysis.region), resolution.text, entries)
        if violations:
            problems = tuple(f"missing binding term: {v.source} -> {v.expected}" for v in violations)
            triples.append((analysis, resolution.text, problems))
    return triples


def _final_line(
    analysis: _Analysis, resolution: _Resolution | None, entries: Sequence[GlossaryEntry]
) -> FinalLine:
    """The final line of a judged region: the resolution, or the deterministic fallback."""
    source = source_text(analysis.region)
    if resolution is not None:
        return FinalLine(
            region_id=analysis.region.id,
            text=resolution.text,
            decision=resolution.decision,
            sources=resolution.sources,
            rationale=resolution.rationale,
            flags=["glossary_violation"] if check_locked_terms(source, resolution.text, entries) else [],
        )
    clean = [c for c in analysis.all_candidates if not check_locked_terms(source, c.text, entries)]
    fallback = clean[0] if clean else analysis.all_candidates[0]
    flags = ["judge_failed"]
    if check_locked_terms(source, fallback.text, entries):
        flags.append("glossary_violation")
    return FinalLine(
        region_id=analysis.region.id,
        text=fallback.text,
        decision="pick",
        sources=[fallback.run_id],
        rationale="judge failed",
        flags=flags,
    )


def _chunks[T](items: Sequence[T], size: int) -> list[list[T]]:
    """Slices of `items` with at most `size` elements, in order."""
    return [list(items[i : i + size]) for i in range(0, len(items), size)]

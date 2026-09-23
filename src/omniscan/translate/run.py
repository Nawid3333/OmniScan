"""The translation runner: one candidate run per profile over a chapter's regions, resumable per chunk.

Errors from the chat client (including `OllamaRateLimitError`) propagate unchanged after the partial
file has been saved, so a re-run resumes from what completed. Tolerant parsing handles the malformed
answers cloud models produce; ids the model never returns get no candidate.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from omniscan.core.schemas import Candidate, CandidateRun, GlossaryEntry, Region
from omniscan.llm.ollama import ChatResponse
from omniscan.translate.parse import parse_translations
from omniscan.translate.profiles import TranslationProfile
from omniscan.translate.prompts import (
    TRANSLATIONS_SCHEMA,
    chat_json_messages,
    source_text,
    substitute_binding,
    translatable,
    translategemma_prompt,
)

_PARTIAL_EVERY_REGIONS = 20  # translategemma saves the partial after this many regions
_QUOTE_PREFIXES = ('"', "“", "「", "『")  # a wrapping quote is only stripped if the source is unquoted


class ChatClient(Protocol):
    """The part of `OllamaClient.chat` the runner uses (tests substitute a fake)."""

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
    ) -> ChatResponse: ...


class _Usage:
    """Token/request counters over the requests made in one `run_profile` call."""

    __slots__ = ("completion_tokens", "prompt_tokens", "repair_requests", "requests")

    def __init__(self) -> None:
        self.prompt_tokens = 0.0
        self.completion_tokens = 0.0
        self.requests = 0.0
        self.repair_requests = 0.0

    def record(self, response: ChatResponse, *, repair: bool) -> None:
        self.prompt_tokens += float(response.prompt_eval_count or 0)
        self.completion_tokens += float(response.eval_count or 0)
        self.requests += 1.0
        if repair:
            self.repair_requests += 1.0

    def as_dict(
        self, targets: Sequence[Region], by_id: dict[str, Candidate], seconds: float
    ) -> dict[str, float]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "requests": self.requests,
            "repair_requests": self.repair_requests,
            "regions": float(len(targets)),
            "missing": float(sum(1 for r in targets if r.id not in by_id)),
            "seconds": seconds,
        }


def run_profile(
    client: ChatClient,
    profile: TranslationProfile,
    regions: Sequence[Region],
    entries: Sequence[GlossaryEntry],
    *,
    partial_path: Path | None = None,
    max_repair_rounds: int = 2,
    story_summary: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> CandidateRun:
    """Translate the chapter's translatable regions with one profile and return its candidate run."""
    start = clock()
    targets = translatable(regions)
    by_id: dict[str, Candidate] = {}
    for candidate in _restored_candidates(partial_path, profile):
        if candidate.region_id in {r.id for r in targets}:
            by_id[candidate.region_id] = candidate
    remaining = [r for r in targets if r.id not in by_id]
    usage = _Usage()

    def save_partial() -> None:
        if partial_path is not None:
            # Intermediate saves don't call `clock()` — only start/end of the whole run consume ticks.
            _save_partial(partial_path, profile, targets, by_id, usage, 0.0)

    try:
        if profile.style == "chat_json":
            _run_chat_json(
                client,
                profile,
                remaining,
                entries,
                by_id,
                save_partial,
                usage,
                max_repair_rounds,
                story_summary,
            )
        else:
            _run_translategemma(client, profile, remaining, entries, by_id, save_partial, usage)
    except BaseException:
        if partial_path is not None:
            _save_partial(partial_path, profile, targets, by_id, usage, clock() - start)
        raise

    return CandidateRun(
        run_id=profile.name,
        profile=profile.name,
        model=profile.model,
        candidates=[by_id[r.id] for r in targets if r.id in by_id],
        usage=usage.as_dict(targets, by_id, clock() - start),
    )


def _run_chat_json(
    client: ChatClient,
    profile: TranslationProfile,
    remaining: Sequence[Region],
    entries: Sequence[GlossaryEntry],
    by_id: dict[str, Candidate],
    save_partial: Callable[[], None],
    usage: _Usage,
    max_repair_rounds: int,
    story_summary: str | None = None,
) -> None:
    """Request the regions in chunks; repair missing ids, then save the partial after every chunk."""
    for chunk in (
        remaining[i : i + profile.chunk_regions] for i in range(0, len(remaining), profile.chunk_regions)
    ):
        texts = _request_translations(
            client, profile, chunk, entries, usage, repair=False, story_summary=story_summary
        )
        missing = [r for r in chunk if r.id not in texts]
        for _ in range(max_repair_rounds):
            if not missing:
                break
            repair = _request_translations(
                client, profile, missing, entries, usage, repair=True, story_summary=story_summary
            )
            texts.update(repair)
            missing = [r for r in missing if r.id not in repair]
        for region in chunk:
            if region.id in texts:
                by_id[region.id] = Candidate(region_id=region.id, text=texts[region.id], notes=None)
        save_partial()


def _request_translations(
    client: ChatClient,
    profile: TranslationProfile,
    regions: Sequence[Region],
    entries: Sequence[GlossaryEntry],
    usage: _Usage,
    *,
    repair: bool,
    story_summary: str | None = None,
) -> dict[str, str]:
    """One chat_json request for `regions`; returns the usable id -> text pairs of the reply."""
    response = client.chat(
        profile.model,
        chat_json_messages(regions, entries, story_summary=story_summary),
        cloud=(profile.endpoint == "cloud"),
        format=TRANSLATIONS_SCHEMA,
        options={"temperature": profile.temperature},
        think=profile.think,
    )
    usage.record(response, repair=repair)
    return parse_translations(response.content, [r.id for r in regions]).texts


def _run_translategemma(
    client: ChatClient,
    profile: TranslationProfile,
    remaining: Sequence[Region],
    entries: Sequence[GlossaryEntry],
    by_id: dict[str, Candidate],
    save_partial: Callable[[], None],
    usage: _Usage,
) -> None:
    """One request per region; saves the partial every `_PARTIAL_EVERY_REGIONS` regions."""
    for done, region in enumerate(remaining, 1):
        text = _unwrap_reply(
            _chat(
                client, profile, [{"role": "user", "content": _prompt_for(region, entries)}], usage
            ).content.strip(),
            source_text(region),
        )
        if text:
            by_id[region.id] = Candidate(region_id=region.id, text=text, notes=None)
        if done % _PARTIAL_EVERY_REGIONS == 0:
            save_partial()


def _prompt_for(region: Region, entries: Sequence[GlossaryEntry]) -> str:
    """The translategemma prompt for one region: locked glossary terms pre-substituted."""
    return translategemma_prompt(substitute_binding(source_text(region), entries))


def _chat(
    client: ChatClient, profile: TranslationProfile, messages: list[dict[str, Any]], usage: _Usage
) -> ChatResponse:
    """One translategemma request (no `format` — the model answers plain text) and its counters."""
    response = client.chat(
        profile.model,
        messages,
        cloud=(profile.endpoint == "cloud"),
        options={"temperature": profile.temperature},
        think=profile.think,
    )
    usage.record(response, repair=False)
    return response


def _unwrap_reply(text: str, source: str) -> str:
    """Drop one pair of wrapping straight quotes the model added (unless the source is quoted)."""
    if (
        len(text) >= 2
        and text.startswith('"')
        and text.endswith('"')
        and not source.startswith(_QUOTE_PREFIXES)
    ):
        return text[1:-1].strip()
    return text


def _restored_candidates(partial_path: Path | None, profile: TranslationProfile) -> list[Candidate]:
    """Candidates of a usable same-profile/same-model partial; any other partial file is ignored."""
    if partial_path is None or not partial_path.is_file():
        return []
    try:
        partial = CandidateRun.load(partial_path)
    except Exception:
        return []
    if partial.profile != profile.name or partial.model != profile.model:
        return []
    return partial.candidates


def _save_partial(
    partial_path: Path,
    profile: TranslationProfile,
    targets: Sequence[Region],
    by_id: dict[str, Candidate],
    usage: _Usage,
    seconds: float,
) -> None:
    """Rewrite the partial file with the candidates collected so far (in target order)."""
    CandidateRun(
        run_id=profile.name,
        profile=profile.name,
        model=profile.model,
        candidates=[by_id[r.id] for r in targets if r.id in by_id],
        usage=usage.as_dict(targets, by_id, seconds),
    ).save(partial_path)

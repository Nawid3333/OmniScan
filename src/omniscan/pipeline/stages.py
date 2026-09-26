"""The ten pipeline stages as `core.stage.Stage` objects, plus `build_stage` — the stage registry.

The eight torch/JSON stages are the stage modules' own classes; translate and judge get thin adapters
here so the runner can treat all ten uniformly. Stage classes are imported inside `build_stage` so
importing this module — and `omniscan --help` — stays free of torch and transformers.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from omniscan.core.config import Config
from omniscan.core.paths import SeriesPaths
from omniscan.core.stage import ChapterContext, Stage
from omniscan.edits.store import FINAL_AUTO_FILE
from omniscan.glossary.store import GlossaryStore
from omniscan.llm.ollama import OllamaRateLimitError
from omniscan.story.store import SummaryStore
from omniscan.translate.chapter import translate_chapter
from omniscan.translate.judge_chapter import judge_chapter

if TYPE_CHECKING:
    from omniscan.core.schemas import CandidateRun, GlossaryEntry
    from omniscan.translate.judge_config import JudgeConfig
    from omniscan.translate.profiles import TranslationProfile
    from omniscan.translate.run import ChatClient

log = logging.getLogger(__name__)

STAGE_ORDER: tuple[str, ...] = (
    "ingest",
    "slice",
    "detect",
    "ocr",
    "translate",
    "judge",
    "inpaint",
    "inpaint_lama",
    "typeset",
    "export",
)
PASS_OF: dict[str, str] = {
    "ingest": "vision",
    "slice": "vision",
    "detect": "vision",
    "ocr": "vision",
    "translate": "text",
    "judge": "text",
    "inpaint": "render",
    "inpaint_lama": "render",
    "typeset": "render",
    "export": "render",
}


def _series_entries(series: SeriesPaths) -> list[GlossaryEntry]:
    """All glossary entries of the series, or none when its db does not exist (never created here)."""
    if not series.db.is_file():
        return []
    with GlossaryStore(series.db) as store:
        return store.list()


def _series_story_context(series: SeriesPaths, chapter: str, *, max_chapters: int = 3) -> str | None:
    """Up to the last `max_chapters` prior chapters' stored summaries, oldest first; None when none exist."""
    if not series.db.is_file():
        return None
    order = series.chapters()
    if chapter not in order:
        return None
    prior = order[: order.index(chapter)]
    pairs: list[tuple[str, str]] = []
    with SummaryStore(series.db) as store:
        for name in prior:
            row = store.get(name)
            if row is not None:  # gaps are fine: an unsummarised chapter just contributes nothing
                pairs.append((name, row.summary))
    if len(pairs) > max_chapters:
        pairs = pairs[-max_chapters:]
    if not pairs:
        return None
    return "\n".join(f"- {name}: {summary}" for name, summary in pairs)


def _needs_local_gpu(endpoint: str, model: str) -> bool:
    """True when a model occupies local VRAM: local endpoint and not an Ollama cloud model (`x:cloud`, `x:31b-cloud`)."""
    return endpoint == "local" and not model.endswith((":cloud", "-cloud"))


@runtime_checkable
class OllamaEvictor(Protocol):
    """A GpuScheduler that can unload local Ollama models (omniscan.gpu.vram.VramManager)."""

    def evict_ollama(self, keep: str | None = None) -> list[str]: ...


def _make_sole_local_model(ctx: ChapterContext, endpoint: str, model: str) -> None:
    """Unload every other local Ollama model before `model` is used, so only one is ever in VRAM.

    Ollama keeps a second model resident whenever its estimate says both fit, and two 12B models on a
    16 GB card then run far slower than either alone (measured: gemma4:12b 34 s -> 87 s for one chapter
    while translategemma:12b stayed loaded). A swap costs one load, which the shared VRAM forced anyway.
    """
    if _needs_local_gpu(endpoint, model) and isinstance(ctx.gpu, OllamaEvictor):
        ctx.gpu.evict_ollama(keep=model)


class TranslateStage:
    """Run every enabled translation profile over a chapter's ocr.json (satisfies core.stage.Stage).

    A profile with a fallback that hits the Ollama rate limit (cloud tokens exhausted) is replaced by its
    fallback for this chapter and for every later chapter of the same pass, so the pass does not pay the
    rate-limit retries again per chapter. The chapter's record still lists the primary's output, which
    then does not exist: the next run translates that chapter again with the primary.
    """

    name: ClassVar[str] = "translate"
    version: ClassVar[int] = 1
    gpu_group: str | None  # OLLAMA_GROUP when a profile (or a fallback) needs a local model, else None

    def __init__(
        self,
        client: ChatClient,
        profiles: Sequence[TranslationProfile],
        fallbacks: Mapping[str, TranslationProfile] | None = None,
    ) -> None:
        self._client = client
        self._profiles = list(profiles)
        self._fallbacks = dict(fallbacks or {})  # primary profile name -> its fallback profile
        self._rate_limited: set[str] = set()  # primaries that hit the rate limit during this pass
        from omniscan.gpu.vram import OLLAMA_GROUP  # deferred: importing this module must not pull torch

        every = [*self._profiles, *self._fallbacks.values()]
        self.gpu_group = (
            OLLAMA_GROUP
            if any(_needs_local_gpu(profile.endpoint, profile.model) for profile in every)
            else None
        )

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        """Files whose content determines this stage's output (raw images and/or upstream artifacts)."""
        inputs = [ctx.paths.artifact("ocr.json")]
        if ctx.series.db.is_file():
            inputs.append(ctx.series.db)  # the glossary: a locked term changes the translations
        return inputs

    def outputs(self, ctx: ChapterContext) -> list[str]:
        """Artifact names (relative to the chapter work dir) this stage writes."""
        return [f"translations/{profile.name}.json" for profile in self._profiles]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        return {
            "profiles": [profile.model_dump() for profile in self._profiles],
            "fallbacks": {name: fallback.model_dump() for name, fallback in sorted(self._fallbacks.items())},
        }

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        entries = _series_entries(ctx.series)
        story_summary = _series_story_context(ctx.series, ctx.paths.chapter)
        regions = fallbacks_used = 0.0
        for profile in self._profiles:
            fallback = self._fallbacks.get(profile.name)
            run: CandidateRun | None = None
            if fallback is None or profile.name not in self._rate_limited:
                try:
                    run = self._translate(ctx, profile, entries, story_summary)
                except OllamaRateLimitError:
                    if fallback is None:
                        raise
                    log.warning(
                        "%s/%s: %s hit the Ollama rate limit; %s translates instead for the rest of this run",
                        ctx.paths.series,
                        ctx.paths.chapter,
                        profile.name,
                        fallback.name,
                    )
                    self._rate_limited.add(profile.name)
            if run is None and fallback is not None:
                run = self._translate(ctx, fallback, entries, story_summary)
                fallbacks_used += 1.0
                ctx.paths.artifact(f"translations/{profile.name}.json").unlink(missing_ok=True)  # stale
            elif fallback is not None:
                ctx.paths.artifact(f"translations/{fallback.name}.json").unlink(missing_ok=True)  # stale
            if run is not None:
                regions += float(run.usage["regions"])
        return {"profiles": float(len(self._profiles)), "regions": regions, "fallbacks_used": fallbacks_used}

    def _translate(
        self,
        ctx: ChapterContext,
        profile: TranslationProfile,
        entries: Sequence[GlossaryEntry],
        story_summary: str | None,
    ) -> CandidateRun:
        """One profile over the chapter (the only local model in VRAM while it runs); its written run."""
        _make_sole_local_model(ctx, profile.endpoint, profile.model)
        _status, run = translate_chapter(
            self._client, ctx.paths, profile, entries, force=True, story_summary=story_summary
        )
        if run is None:
            raise RuntimeError(f"translate_chapter skipped {profile.name} despite force=True")
        return run


class JudgeStage:
    """Judge a chapter's candidate runs into final.json (satisfies core.stage.Stage).

    `run_ids` (the translate stage's profiles and their fallbacks) limits the candidates to runs this
    pipeline produces, so a run left over from an earlier profile set is never judged; None judges every
    run in translations/. A judge rate limit does not stop the run: the regions it did not answer get the
    deterministic pick, flagged "judge_failed".
    """

    name: ClassVar[str] = "judge"
    version: ClassVar[int] = 1
    gpu_group: str | None  # OLLAMA_GROUP for a local judge model, else None

    def __init__(
        self, client: ChatClient, judge_cfg: JudgeConfig, run_ids: Sequence[str] | None = None
    ) -> None:
        self._client = client
        self._judge_cfg = judge_cfg
        self._run_ids = None if run_ids is None else list(run_ids)
        from omniscan.gpu.vram import OLLAMA_GROUP  # deferred: importing this module must not pull torch

        self.gpu_group = OLLAMA_GROUP if _needs_local_gpu(judge_cfg.endpoint, judge_cfg.model) else None

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        """Files whose content determines this stage's output (raw images and/or upstream artifacts)."""
        inputs = [ctx.paths.artifact("ocr.json")]
        translations = ctx.paths.artifact("translations")
        if translations.is_dir():
            inputs.extend(
                sorted(
                    path
                    for path in translations.glob("*.json")
                    if not path.name.startswith(".")  # dot-prefixed partials are never candidate runs
                )
            )
        if ctx.series.db.is_file():
            inputs.append(ctx.series.db)
        return inputs

    def outputs(self, ctx: ChapterContext) -> list[str]:
        """Artifact names (relative to the chapter work dir) this stage writes."""
        return ["final.json", FINAL_AUTO_FILE]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        return self._judge_cfg.model_dump()

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        _make_sole_local_model(ctx, self._judge_cfg.endpoint, self._judge_cfg.model)
        run_ids = None
        if self._run_ids is not None:
            present = [r for r in self._run_ids if ctx.paths.artifact(f"translations/{r}.json").is_file()]
            run_ids = present or None  # none of ours on disk: judge whatever is there (the old behaviour)
        _status, _artifact, stats = judge_chapter(
            self._client,
            ctx.paths,
            self._judge_cfg,
            _series_entries(ctx.series),
            run_ids=run_ids,
            force=True,
            story_summary=_series_story_context(ctx.series, ctx.paths.chapter),
            rate_limit_fallback=True,
        )
        if stats is None:
            return {}
        return {
            "regions": float(stats.regions),
            "judged": float(stats.judged),
            "auto_picked": float(stats.auto_picked),
            "untranslated": float(stats.untranslated),
            "violations_left": float(stats.violations_left),
            "requests": float(stats.requests),
            "rate_limited": float(stats.rate_limited),
        }


def build_stage(name: str, cfg: Config, *, client: ChatClient | None = None) -> Stage:
    """A fresh stage object for `name`; the translate/judge stages need a chat client."""
    if name in ("translate", "judge"):
        if client is None:
            raise ValueError(f"stage {name!r} needs a chat client")
        from omniscan.translate.profiles import default_profile_paths, load_profiles, resolve_fallbacks

        known = load_profiles(default_profile_paths())
        profiles = [p for p in known.values() if p.enabled]
        fallbacks = resolve_fallbacks(profiles, known)
        if name == "translate":
            # gpu_group is a per-instance attribute on the adapters while core.stage.Stage declares
            # it ClassVar; the runtime contract is identical, so only that declaration-level
            # mismatch is silenced here.
            return TranslateStage(client, profiles, fallbacks)  # pyright: ignore[reportReturnType]
        from omniscan.translate.judge_config import default_judge_paths, load_judge_config

        run_ids = [p.name for p in profiles] + [f.name for f in fallbacks.values()]
        return JudgeStage(client, load_judge_config(default_judge_paths()), run_ids)  # pyright: ignore[reportReturnType]
    if name == "ingest":
        from omniscan.ingest.stage import IngestStage

        return IngestStage()
    if name == "slice":
        from omniscan.slicer.stage import SliceStage

        return SliceStage()
    if name == "detect":
        from omniscan.detect.stage import DetectStage

        return DetectStage()
    if name == "ocr":
        from omniscan.ocr.stage import OcrStage

        return OcrStage()
    if name == "inpaint":
        from omniscan.inpaint.stage import InpaintStage

        return InpaintStage()
    if name == "inpaint_lama":
        from omniscan.inpaint.lama_stage import LamaStage

        return LamaStage()
    if name == "typeset":
        from omniscan.typeset.stage import TypesetStage

        return TypesetStage()
    if name == "export":
        from omniscan.export.stage import ExportStage

        return ExportStage()
    raise ValueError(f"unknown stage {name!r} (known: {', '.join(STAGE_ORDER)})")

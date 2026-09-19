"""The ten pipeline stages as `core.stage.Stage` objects, plus `build_stage` — the stage registry.

The eight torch/JSON stages are the stage modules' own classes; translate and judge get thin adapters
here so the runner can treat all ten uniformly. Stage classes are imported inside `build_stage` so
importing this module — and `omniscan --help` — stays free of torch and transformers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from omniscan.core.config import Config
from omniscan.core.paths import SeriesPaths
from omniscan.core.stage import ChapterContext, Stage
from omniscan.glossary.store import GlossaryStore
from omniscan.translate.chapter import translate_chapter
from omniscan.translate.judge_chapter import judge_chapter

if TYPE_CHECKING:
    from omniscan.core.schemas import GlossaryEntry
    from omniscan.translate.judge_config import JudgeConfig
    from omniscan.translate.profiles import TranslationProfile
    from omniscan.translate.run import ChatClient

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


def _needs_local_gpu(endpoint: str, model: str) -> bool:
    """True when a model occupies local VRAM: local endpoint and not an Ollama cloud model (`x:cloud`, `x:31b-cloud`)."""
    return endpoint == "local" and not model.endswith((":cloud", "-cloud"))


class TranslateStage:
    """Run every enabled translation profile over a chapter's ocr.json (satisfies core.stage.Stage)."""

    name: ClassVar[str] = "translate"
    version: ClassVar[int] = 1
    gpu_group: str | None  # OLLAMA_GROUP when a profile needs a local model, else None

    def __init__(self, client: ChatClient, profiles: Sequence[TranslationProfile]) -> None:
        self._client = client
        self._profiles = list(profiles)
        from omniscan.gpu.vram import OLLAMA_GROUP  # deferred: importing this module must not pull torch

        self.gpu_group = (
            OLLAMA_GROUP
            if any(_needs_local_gpu(profile.endpoint, profile.model) for profile in self._profiles)
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
        return {"profiles": [profile.model_dump() for profile in self._profiles]}

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        entries = _series_entries(ctx.series)
        regions = 0.0
        for profile in self._profiles:
            _status, run = translate_chapter(self._client, ctx.paths, profile, entries, force=True)
            if run is not None:
                regions += float(run.usage["regions"])
        return {"profiles": float(len(self._profiles)), "regions": regions}


class JudgeStage:
    """Judge a chapter's candidate runs into final.json (satisfies core.stage.Stage)."""

    name: ClassVar[str] = "judge"
    version: ClassVar[int] = 1
    gpu_group: str | None  # OLLAMA_GROUP for a local judge model, else None

    def __init__(self, client: ChatClient, judge_cfg: JudgeConfig) -> None:
        self._client = client
        self._judge_cfg = judge_cfg
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
        return ["final.json"]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        return self._judge_cfg.model_dump()

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        _status, _artifact, stats = judge_chapter(
            self._client, ctx.paths, self._judge_cfg, _series_entries(ctx.series), force=True
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
        }


def build_stage(name: str, cfg: Config, *, client: ChatClient | None = None) -> Stage:
    """A fresh stage object for `name`; the translate/judge stages need a chat client."""
    if name in ("translate", "judge"):
        if client is None:
            raise ValueError(f"stage {name!r} needs a chat client")
        if name == "translate":
            from omniscan.translate.profiles import default_profile_paths, load_profiles

            profiles = [p for p in load_profiles(default_profile_paths()).values() if p.enabled]
            # gpu_group is a per-instance attribute on the adapters while core.stage.Stage declares
            # it ClassVar; the runtime contract is identical, so only that declaration-level
            # mismatch is silenced here.
            return TranslateStage(client, profiles)  # pyright: ignore[reportReturnType]
        from omniscan.translate.judge_config import default_judge_paths, load_judge_config

        return JudgeStage(client, load_judge_config(default_judge_paths()))  # pyright: ignore[reportReturnType]
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

"""The pipeline runner: ordered passes over all chapters, so each model group is loaded once.

`plan_passes` splits the selected stages into three passes (vision, text, render) and `run_pipeline`
executes them with the core stage runner — one `run_series` call per pass, so a model group stays
resident across every chapter of its pass. Chapters whose stage fails are dropped from the later
passes; an Ollama rate limit stops the whole run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from omniscan.core.paths import SeriesPaths
from omniscan.core.stage import GpuScheduler, StageOutcome, run_series
from omniscan.pipeline.stages import PASS_OF, STAGE_ORDER, build_stage

if TYPE_CHECKING:
    from omniscan.core.config import Config
    from omniscan.translate.run import ChatClient

ReportFn = Callable[[str, StageOutcome], None]


def plan_passes(stages: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Group stage names into consecutive passes sharing a `PASS_OF` label, ordered by `STAGE_ORDER`."""
    unknown = next((name for name in stages if name not in STAGE_ORDER), None)
    if unknown is not None:
        raise ValueError(f"unknown stage {unknown!r} (known: {', '.join(STAGE_ORDER)})")
    selected = set(stages)
    passes: list[tuple[str, list[str]]] = []
    for name in (name for name in STAGE_ORDER if name in selected):
        label = PASS_OF[name]
        if passes and passes[-1][0] == label:
            passes[-1][1].append(name)
        else:
            passes.append((label, [name]))
    return passes


@dataclass(slots=True)
class PipelineResult:
    """What one `run_pipeline` call did: per-chapter outcomes, first failures, rate-limit abort."""

    outcomes: dict[str, list[StageOutcome]] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)  # chapter -> "<stage>: <error>"
    aborted: str | None = None  # "rate limit" when an Ollama rate limit stopped the run

    @property
    def ok(self) -> bool:
        """True when no chapter failed and the run was not aborted."""
        return not self.failed and self.aborted is None


def run_pipeline(
    cfg: Config,
    series: str,
    chapters: Sequence[str] | None = None,
    *,
    stages: Sequence[str] | None = None,
    lama: bool = True,
    force: bool = False,
    client: ChatClient | None = None,
    gpu: GpuScheduler | None = None,
    report: ReportFn | None = None,
) -> PipelineResult:
    """Run the selected stages over the chapters in passes; failed chapters skip the later passes.

    The caller owns `client` and `gpu`: the runner never builds either. `report` is called once per
    outcome after its pass finished (in chapter order, then stage order).
    """
    names = list(stages) if stages is not None else list(STAGE_ORDER)
    if not lama:
        names = [name for name in names if name != "inpaint_lama"]
    passes = plan_passes(names)  # validates the names and puts them in STAGE_ORDER
    active = list(chapters) if chapters is not None else SeriesPaths.from_config(cfg, series).chapters()
    if not active:
        raise ValueError(f"no chapters found for series {series!r}")
    needs_client = next(
        (name for _label, pass_stages in passes for name in pass_stages if PASS_OF[name] == "text"), None
    )
    if needs_client is not None and client is None:
        raise ValueError(f"stage {needs_client!r} needs a chat client")

    result = PipelineResult()
    for _label, pass_stages in passes:
        stage_objects = [build_stage(name, cfg, client=client) for name in pass_stages]
        pass_results = run_series(stage_objects, cfg, series, active, gpu=gpu, force=force)
        for chapter, outcomes in pass_results.items():
            result.outcomes.setdefault(chapter, []).extend(outcomes)
            if report is not None:
                for outcome in outcomes:
                    report(chapter, outcome)
            first_failure = next((o for o in outcomes if o.status == "failed"), None)
            if first_failure is not None and first_failure.error is not None:
                result.failed.setdefault(chapter, f"{first_failure.stage}: {first_failure.error}")
                if first_failure.error.startswith("OllamaRateLimitError"):
                    result.aborted = "rate limit"
        if result.aborted is not None:
            break
        active = [chapter for chapter in active if chapter not in result.failed]
    return result


def needs_gpu(names: Sequence[str], cfg: Config, client: ChatClient | None) -> bool:
    """True when any stage built for `names` needs a GPU model group (translate/judge need a client)."""
    return any(build_stage(name, cfg, client=client).gpu_group is not None for name in names)

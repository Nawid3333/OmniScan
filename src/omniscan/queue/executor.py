"""Executor that runs a queued job's stages through the pipeline runner (one run_pipeline per job)."""

from __future__ import annotations

from omniscan.core.config import Config, get_secrets
from omniscan.core.paths import SeriesPaths
from omniscan.llm.ollama import OllamaClient
from omniscan.pipeline.runner import needs_gpu, run_pipeline
from omniscan.pipeline.stages import PASS_OF, STAGE_ORDER
from omniscan.queue.store import Job
from omniscan.queue.worker import Executor, PermanentJobError


def stage_executor(cfg: Config) -> Executor:
    """Run a queued job's stages over its series via run_pipeline; unknown stages never retry."""

    def _run(job: Job) -> None:
        unknown = next((name for name in job.stages if name not in STAGE_ORDER), None)
        if unknown is not None:
            raise PermanentJobError(f"stage {unknown!r} is not implemented yet")
        if job.chapters is None and not SeriesPaths.from_config(cfg, job.series).chapters():
            raise PermanentJobError(f"no chapters found for series {job.series!r}")
        names = list(job.stages)
        client = (
            OllamaClient(cfg.ollama, get_secrets())
            if any(PASS_OF[name] == "text" for name in names)
            else None
        )
        gpu = None
        try:
            if needs_gpu(names, cfg, client):
                from omniscan.gpu.groups import (
                    build_vram_manager,  # deferred: importing this module must not pull torch
                )

                gpu = build_vram_manager(cfg)
            result = run_pipeline(
                cfg, job.series, job.chapters, stages=names, force=job.force, client=client, gpu=gpu
            )
        finally:
            if gpu is not None:
                gpu.release()  # the models leave VRAM when the job ends
            if client is not None:
                client.close()
        if not result.ok:
            if result.aborted is not None:
                raise RuntimeError("Ollama rate limit reached")  # the worker retries under its attempt rules
            raise RuntimeError(
                "; ".join(f"{chapter}: {message}" for chapter, message in result.failed.items())
            )

    return _run

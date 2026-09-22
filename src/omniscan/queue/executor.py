"""Executor that runs a queued job's stages through the pipeline runner (one run_pipeline per job)."""

from __future__ import annotations

from omniscan.core.config import Config, get_secrets, series_config
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
        # merge series.toml before building the VRAM manager: it decides which OCR/detect/inpaint
        # models to load, and make_context() inside run_pipeline would otherwise merge it too late.
        # A new name, not a reassigned `cfg` — `cfg` is the enclosing stage_executor's parameter,
        # captured here by closure, and reassigning it would make it a local of `_run` for the whole
        # function (UnboundLocalError at the SeriesPaths.from_config call above).
        job_cfg = series_config(cfg, SeriesPaths.from_config(cfg, job.series).library_dir)
        names = list(job.stages)
        client = (
            OllamaClient(job_cfg.ollama, get_secrets())
            if any(PASS_OF[name] == "text" for name in names)
            else None
        )
        gpu = None
        hw_lock = None
        try:
            if needs_gpu(names, job_cfg, client):
                # exclusive real-GPU access first: build_vram_manager's warm-up thread already
                # touches the device, so the lock must be held before it is called, not after
                # (see gpu/lock.py) — two queued jobs, or a job and a live CLI run, must never
                # touch the real GPU at the same instant
                if job_cfg.gpu.device != "cpu":
                    from omniscan.gpu.lock import acquire_gpu_lock

                    hw_lock = acquire_gpu_lock()
                from omniscan.gpu.groups import (
                    build_vram_manager,  # deferred: importing this module must not pull torch
                )

                gpu = build_vram_manager(job_cfg)
            result = run_pipeline(
                job_cfg,
                job.series,
                job.chapters,
                stages=names,
                force=job.force,
                client=client,
                gpu=gpu,
            )
        finally:
            if gpu is not None:
                gpu.release()  # the models leave VRAM when the job ends
            if hw_lock is not None:
                from omniscan.gpu.lock import release_gpu_lock

                release_gpu_lock(hw_lock)
            if client is not None:
                client.close()
        if not result.ok:
            if result.aborted is not None:
                raise RuntimeError("Ollama rate limit reached")  # the worker retries under its attempt rules
            raise RuntimeError(
                "; ".join(f"{chapter}: {message}" for chapter, message in result.failed.items())
            )

    return _run

"""Where does the GPU sit idle? Per-stage wall time, kernel time and duty cycle, plus first-call stalls.

Usage (repo root, uses the machine config and the models in `paths.models_dir`):
  uv run python scripts/gpu_duty.py SERIES CHAPTER STAGE [STAGE ...]      # e.g. ingest slice detect ocr
  uv run python scripts/gpu_duty.py --ops                                  # time the first vs second call of the op families

Why: a GPU that alternates between long idle gaps and bursts of tiny kernels whines (coil whine) and wastes time.
`duty` = summed kernel time of `torch.profiler` / wall time of the stage. WARNING: the profiler perturbs the run on this
ROCm-Windows build (inflated numbers, and one profiled render pass produced a corrupted export) — treat a profiled
run as a diagnosis only, never keep its outputs, and prefer plain wall-clock timings. The `--ops` first-call table
is exact wall time and safe. Measured facts and the fixes are in docs/GPU_NOTES.md.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F  # noqa: N812 — torch's standard alias
from torch.profiler import ProfilerActivity, profile

import omniscan.core.stage as core_stage
from omniscan.core.config import get_config
from omniscan.gpu.device import resolve_device
from omniscan.gpu.groups import build_vram_manager
from omniscan.pipeline.runner import run_pipeline


def stage_duty(series: str, chapter: str, stages: list[str]) -> None:
    """Run `stages` (forced) for one chapter and print wall/kernel time and duty per stage."""
    original = core_stage.run_stage
    rows: list[tuple[str, str, float, float, int]] = []

    def wrapped(stage, ctx, *, force=False):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            started = time.perf_counter()
            outcome = original(stage, ctx, force=force)
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
        averages = prof.key_averages()
        kernel_s = sum(e.self_device_time_total for e in averages) / 1e6
        kernels = sum(e.count for e in averages if e.self_device_time_total > 0)
        rows.append((stage.name, outcome.status, wall, kernel_s, kernels))
        return outcome

    core_stage.run_stage = wrapped
    cfg = get_config()
    gpu = build_vram_manager(cfg)
    try:
        run_pipeline(cfg, series, [chapter], stages=stages, force=True, gpu=gpu, client=None)
    finally:
        core_stage.run_stage = original
        gpu.release()
    print(f"{'stage':14}{'status':8}{'wall s':>8}{'gpu s':>8}{'duty %':>8}{'kernels':>9}")
    for name, status, wall, kernel_s, kernels in rows:
        duty = 100 * kernel_s / wall if wall else 0.0
        print(f"{name:14}{status:8}{wall:8.2f}{kernel_s:8.2f}{duty:8.1f}{kernels:9d}")


def first_calls() -> None:
    """Time the first and the second call of the op families the pipeline uses (library initialisation stalls)."""
    device = resolve_device(get_config().gpu.device)
    torch.cuda.set_device(device)

    def timed(name: str, fn) -> None:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        first = time.perf_counter() - started
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        print(f"{name:32} first {first:6.2f}s  second {time.perf_counter() - started:7.3f}s")

    x = torch.randn(1, 64, 256, 256, device=device)
    timed(
        "conv3x3 (MIOpen + CK init)", lambda: F.conv2d(x, torch.randn(64, 64, 3, 3, device=device), padding=1)
    )
    timed("rfft2 (rocFFT plan)", lambda: torch.fft.rfft2(x))
    timed("irfft2", lambda: torch.fft.irfft2(torch.fft.rfft2(x), s=(256, 256)))
    timed(
        "conv_transpose2d",
        lambda: F.conv_transpose2d(x, torch.randn(64, 32, 4, 4, device=device), stride=2, padding=1),
    )
    timed(
        "bmm (rocBLAS)",
        lambda: torch.bmm(torch.randn(4, 128, 128, device=device), torch.randn(4, 128, 128, device=device)),
    )
    timed("max_pool2d", lambda: F.max_pool2d(x, 9, 1, 4))


def main() -> None:
    """Parse the command line and run the requested measurement."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--ops", action="store_true", help="time first vs second call of the op families and exit"
    )
    parser.add_argument("series", nargs="?")
    parser.add_argument("chapter", nargs="?")
    parser.add_argument("stages", nargs="*")
    args = parser.parse_args()
    if args.ops:
        first_calls()
    elif args.series and args.chapter and args.stages:
        stage_duty(args.series, args.chapter, args.stages)
    else:
        parser.error("give SERIES CHAPTER STAGE... or --ops")


if __name__ == "__main__":
    main()

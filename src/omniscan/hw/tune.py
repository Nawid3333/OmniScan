"""The tuning plan: what OmniScan should be set to on a given machine, from its hardware snapshot alone.

A pure function from `HardwareInfo` to a `Plan`: the torch backend extra to install, the device, the VRAM
budget, which OCR engine and batch sizes fit, whether LaMa inpainting is worth running, whether translation
should stay in the cloud, and the config overrides that make it so (`omniscan tune --apply`, the settings
screen's *Optimise for this PC*). The numbers come from the model catalogue's memory needs and the
benchmarks in docs/benchmarks/; every rule is a plain threshold so a profile test can pin it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from omniscan.hw.detect import GpuInfo, HardwareInfo

Tier = Literal["workstation", "gaming", "light", "cpu"]
TorchExtra = Literal["cuda", "rocm-gfx1201", "xpu", "mps", "cpu"]

VLM_OCR_MIN_VRAM_GB = 8.0  # PaddleOCR-VL (catalogue: ocr-vl, 8 GB)
LOCAL_LLM_MIN_VRAM_GB = 9.0  # translategemma:12b (catalogue: llm-translategemma-12b, 9 GB)
LAMA_MIN_VRAM_GB = 3.0  # big-lama needs 2 GB plus the page window
VRAM_HEADROOM_GB = 1.5  # left to the display driver and other programs
WORKSTATION_VRAM_GB = 16.0
GAMING_VRAM_GB = 8.0
RX_9070_SERIES = ("RX 9070",)  # the only AMD cards with a Windows ROCm extra today (gfx1201)


@dataclass(frozen=True, slots=True)
class Plan:
    """What to set on this machine, and why."""

    tier: Tier
    torch_extra: TorchExtra
    device: str  # the gpu.device to set ("cuda:1", "xpu:0", "mps", "cpu")
    gpu_name: str | None  # the GPU the plan is for, None on a CPU machine
    vram_gb: float | None  # its memory, None on a CPU machine
    vram_budget_gib: (
        float | None
    )  # gpu.vram_budget_gib to set, None when the GPU is not torch-managed (mps/cpu)
    ocr_engine: Literal["ppocr", "paddleocr_vl"]
    rec_batch_size: int
    crop_batch_size: int
    detect_batch_size: int
    lama: bool  # run the LaMa inpaint stage by default
    translation: Literal["cloud", "local-or-cloud"]
    notes: tuple[str, ...] = ()
    overrides: dict[tuple[str, str], object] = field(default_factory=dict)  # (section, key) -> value to set


def main_gpu(hw: HardwareInfo) -> GpuInfo | None:
    """The GPU the pipeline would run on: the first discrete one (an integrated-only machine counts as none)."""
    return next((gpu for gpu in hw.gpus if not gpu.integrated), None)


def torch_extra_for(hw: HardwareInfo, gpu: GpuInfo | None) -> tuple[TorchExtra, str | None]:
    """The backend extra to install for this machine, and a note when the GPU cannot be used yet."""
    if gpu is None:
        return "cpu", None
    if gpu.backend == "cuda":
        return "cuda", None
    if gpu.backend == "xpu":
        return "xpu", None
    if gpu.backend == "mps":
        return "mps", None
    if gpu.backend == "rocm":
        if hw.os == "windows" and any(marker in gpu.name for marker in RX_9070_SERIES):
            return "rocm-gfx1201", None
        if hw.os == "linux":
            return "cpu", (
                f"{gpu.name}: PyTorch's Linux ROCm wheels run it, but OmniScan has no extra for it yet "
                "(docs/ROADMAP.md X1); install torch from PyTorch's ROCm index by hand"
            )
        return "cpu", f"{gpu.name}: no ROCm build for this card on {hw.os} yet; the CPU build runs"
    return "cpu", f"{gpu.name}: unknown backend {gpu.backend!r}; the CPU build runs"


def plan_for(hw: HardwareInfo) -> Plan:
    """The tuning plan for a hardware snapshot."""
    gpu = main_gpu(hw)
    extra, extra_note = torch_extra_for(hw, gpu)
    notes: list[str] = []
    if extra_note:
        notes.append(extra_note)
    integrated = next((g for g in hw.gpus if g.integrated), None)
    if gpu is None and integrated is not None:
        notes.append(
            f"only an integrated GPU ({integrated.name}) was found; it is skipped, the CPU runs the models"
        )
    usable = gpu if extra != "cpu" else None
    if gpu is not None and usable is None:
        notes.append("the GPU is not usable by this build, so the plan is the CPU plan")
    vram = usable.vram_gb if usable is not None else None
    device = usable.device if usable is not None else "cpu"
    if vram is None:
        tier: Tier = "cpu"
    elif vram >= WORKSTATION_VRAM_GB:
        tier = "workstation"
    elif vram >= GAMING_VRAM_GB:
        tier = "gaming"
    else:
        tier = "light"
    budget = None
    if usable is not None and usable.backend != "mps":
        budget = round(max(1.0, usable.vram_gb - VRAM_HEADROOM_GB), 1)
    if vram is not None and vram >= VLM_OCR_MIN_VRAM_GB:
        engine: Literal["ppocr", "paddleocr_vl"] = "paddleocr_vl"
        rec, crop, det = 64, 16, 8
    elif vram is not None and vram >= 4.0:
        engine, rec, crop, det = "ppocr", 32, 8, 4
    elif vram is not None:
        engine, rec, crop, det = "ppocr", 16, 4, 2
        notes.append(
            f"{usable.name if usable else 'the GPU'} has {vram:g} GB: small batches, PP-OCR models only"
        )
    else:
        engine, rec, crop, det = "ppocr", 16, 4, 2
    lama = vram is not None and vram >= LAMA_MIN_VRAM_GB
    if not lama:
        notes.append(
            "LaMa inpainting is off by default (about a minute per page on a CPU); flat fills clean the "
            "balloons and the Studio's tools fix the rest"
        )
    translation: Literal["cloud", "local-or-cloud"] = (
        "local-or-cloud" if vram is not None and vram >= LOCAL_LLM_MIN_VRAM_GB else "cloud"
    )
    if translation == "cloud":
        notes.append(
            f"translation stays with the cloud profile: a local model needs {LOCAL_LLM_MIN_VRAM_GB:g} GB of GPU "
            "memory for its fallback (Ollama on this machine would run it on the CPU)"
        )
    if hw.ram_gb < 16:
        notes.append(f"{hw.ram_gb:g} GB of RAM is tight: close other programs during a run")
    overrides: dict[tuple[str, str], object] = {
        ("gpu", "device"): device,
        ("ocr", "engine"): engine,
        ("ocr", "rec_batch_size"): rec,
        ("ocr", "crop_batch_size"): crop,
        ("detect", "batch_size"): det,
        ("inpaint", "lama"): lama,
    }
    if budget is not None:
        overrides[("gpu", "vram_budget_gib")] = budget
    return Plan(
        tier=tier,
        torch_extra=extra,
        device=device,
        gpu_name=usable.name if usable is not None else None,
        vram_gb=vram,
        vram_budget_gib=budget,
        ocr_engine=engine,
        rec_batch_size=rec,
        crop_batch_size=crop,
        detect_batch_size=det,
        lama=lama,
        translation=translation,
        notes=tuple(notes),
        overrides=overrides,
    )


def describe(plan: Plan) -> list[str]:
    """The plan as the lines `omniscan tune` prints."""
    lines = [
        f"tier: {plan.tier}",
        f"torch backend to install: uv sync --extra {plan.torch_extra} --extra gui",
        f"device: {plan.device}" + (f" ({plan.gpu_name}, {plan.vram_gb:g} GB)" if plan.gpu_name else ""),
    ]
    if plan.vram_budget_gib is not None:
        lines.append(f"GPU memory budget: {plan.vram_budget_gib:g} GiB")
    lines.append(
        f"OCR: {plan.ocr_engine} (recognition batch {plan.rec_batch_size}, crop batch {plan.crop_batch_size}, "
        f"detector batch {plan.detect_batch_size})"
    )
    lines.append(f"LaMa inpainting: {'on' if plan.lama else 'off'}")
    lines.append(f"translation: {plan.translation}")
    lines.extend(f"note: {note}" for note in plan.notes)
    return lines


def plan_json(plan: Plan) -> dict[str, object]:
    """The plan as JSON-ready data (overrides keyed "section.key")."""
    return {
        "tier": plan.tier,
        "torch_extra": plan.torch_extra,
        "device": plan.device,
        "gpu_name": plan.gpu_name,
        "vram_gb": plan.vram_gb,
        "vram_budget_gib": plan.vram_budget_gib,
        "ocr_engine": plan.ocr_engine,
        "rec_batch_size": plan.rec_batch_size,
        "crop_batch_size": plan.crop_batch_size,
        "detect_batch_size": plan.detect_batch_size,
        "lama": plan.lama,
        "translation": plan.translation,
        "notes": list(plan.notes),
        "overrides": {f"{section}.{key}": value for (section, key), value in plan.overrides.items()},
    }

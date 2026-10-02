"""The machine profiles and the tuning plan for each: every GPU vendor, CPU-only and integrated-only machines."""

from __future__ import annotations

from typing import cast

import pytest

from omniscan.core.config import Config, get_config
from omniscan.hw import profiles
from omniscan.hw.assess import assess
from omniscan.hw.detect import detect_hardware
from omniscan.hw.tune import Plan, describe, main_gpu, plan_for, plan_json
from omniscan.models.catalog import load_catalog

EXPECTED: dict[str, dict[str, object]] = {
    "rtx4090": {
        "tier": "workstation",
        "torch_extra": "cuda",
        "device": "cuda:0",
        "ocr_engine": "paddleocr_vl",
        "lama": True,
        "translation": "local-or-cloud",
        "vram_budget_gib": 22.5,
    },
    "rtx3060": {
        "tier": "gaming",
        "torch_extra": "cuda",
        "device": "cuda:0",
        "ocr_engine": "paddleocr_vl",
        "lama": True,
        "translation": "local-or-cloud",
        "vram_budget_gib": 10.5,
    },
    "gtx1650-laptop": {
        "tier": "light",
        "torch_extra": "cuda",
        "device": "cuda:0",
        "ocr_engine": "ppocr",
        "lama": True,
        "translation": "cloud",
        "vram_budget_gib": 2.5,
        "rec_batch_size": 32,
    },
    "rx9070xt": {
        "tier": "workstation",
        "torch_extra": "rocm-gfx1201",
        "device": "cuda:1",
        "ocr_engine": "paddleocr_vl",
        "lama": True,
        "translation": "local-or-cloud",
        "vram_budget_gib": 14.5,
    },
    "rx7800xt-linux": {
        "tier": "cpu",
        "torch_extra": "cpu",
        "device": "cpu",
        "ocr_engine": "ppocr",
        "lama": False,
        "translation": "cloud",
        "vram_budget_gib": None,
    },
    "arc-b580": {
        "tier": "gaming",
        "torch_extra": "xpu",
        "device": "xpu:1",
        "ocr_engine": "paddleocr_vl",
        "lama": True,
        "translation": "local-or-cloud",
        "vram_budget_gib": 10.5,
    },
    "m2-air": {
        "tier": "workstation",
        "torch_extra": "mps",
        "device": "mps",
        "ocr_engine": "paddleocr_vl",
        "lama": True,
        "translation": "local-or-cloud",
        "vram_budget_gib": None,
    },
    "m4-max": {
        "tier": "workstation",
        "torch_extra": "mps",
        "device": "mps",
        "ocr_engine": "paddleocr_vl",
        "lama": True,
        "translation": "local-or-cloud",
        "vram_budget_gib": None,
    },
    "cpu-laptop": {
        "tier": "cpu",
        "torch_extra": "cpu",
        "device": "cpu",
        "ocr_engine": "ppocr",
        "lama": False,
        "translation": "cloud",
        "vram_budget_gib": None,
        "rec_batch_size": 16,
    },
    "igpu-only": {
        "tier": "cpu",
        "torch_extra": "cpu",
        "device": "cpu",
        "ocr_engine": "ppocr",
        "lama": False,
        "translation": "cloud",
        "vram_budget_gib": None,
    },
    "cpu-server-linux": {
        "tier": "cpu",
        "torch_extra": "cpu",
        "device": "cpu",
        "ocr_engine": "ppocr",
        "lama": False,
        "translation": "cloud",
        "vram_budget_gib": None,
    },
}


def test_every_profile_has_an_expectation_and_vice_versa() -> None:
    assert set(EXPECTED) == set(profiles.profile_names())


@pytest.mark.parametrize("name", profiles.profile_names())
def test_the_plan_for_each_profile(name: str) -> None:
    plan = plan_for(profiles.profile(name))
    got = {key: getattr(plan, key) for key in EXPECTED[name]}
    assert got == EXPECTED[name], name
    assert plan.overrides[("gpu", "device")] == plan.device
    assert plan.overrides[("inpaint", "lama")] == plan.lama
    assert plan.overrides[("ocr", "engine")] == plan.ocr_engine
    assert (("gpu", "vram_budget_gib") in plan.overrides) == (plan.vram_budget_gib is not None)
    assert describe(plan)[0] == f"tier: {plan.tier}"
    assert cast(dict[str, object], plan_json(plan)["overrides"])["gpu.device"] == plan.device


@pytest.mark.parametrize("name", profiles.profile_names())
def test_the_plan_only_sets_values_the_config_accepts(name: str) -> None:
    """Every override validates against the config model (so `tune --apply` can never write a refused value)."""
    plan = plan_for(profiles.profile(name))
    data: dict[str, dict[str, object]] = {}
    for (section, key), value in plan.overrides.items():
        data.setdefault(section, {})[key] = value
    cfg = Config.model_validate(data)
    assert cfg.gpu.device == plan.device and cfg.inpaint.lama is plan.lama


@pytest.mark.parametrize("name", profiles.profile_names())
def test_every_catalog_model_can_be_assessed_on_each_profile(name: str) -> None:
    """The model fit check never crashes on a profile, and a machine without a GPU never runs one on a GPU."""
    hw = profiles.profile(name)
    for entry in load_catalog():
        compat = assess(entry, hw)
        assert compat.level in ("ok", "slow", "warn", "incompatible")
        if not hw.gpus and compat.device not in (None, "cpu", "ollama"):
            raise AssertionError(f"{entry.id} would run on {compat.device} on GPU-less {name}")


def test_notes_explain_the_cpu_and_integrated_cases() -> None:
    igpu = plan_for(profiles.profile("igpu-only"))
    assert any("integrated" in note for note in igpu.notes)
    linux_amd = plan_for(profiles.profile("rx7800xt-linux"))
    assert any("no extra for it yet" in note for note in linux_amd.notes)
    laptop = plan_for(profiles.profile("cpu-laptop"))
    assert any("RAM is tight" in note for note in laptop.notes)
    assert any("LaMa" in note for note in laptop.notes)
    assert main_gpu(profiles.profile("rx9070xt")).name == "AMD Radeon RX 9070 XT"  # type: ignore[union-attr]


def test_unknown_profile_names_the_known_ones() -> None:
    with pytest.raises(ValueError, match="rtx4090"):
        profiles.profile("commodore64")


def test_the_environment_variable_simulates_a_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(profiles.SIMULATE_ENV, "arc-b580")
    hw = detect_hardware()
    assert hw.best_device == "xpu:1" and [g.vendor for g in hw.gpus] == ["intel", "intel"]
    assert detect_hardware(simulate="m2-air").gpus[0].backend == "mps"
    monkeypatch.setenv(profiles.SIMULATE_ENV, "")
    assert profiles.simulated_profile() is None


def test_plans_are_plain_data() -> None:
    plan = plan_for(profiles.profile("rtx3060"))
    assert isinstance(plan, Plan) and get_config().inpaint.lama is True

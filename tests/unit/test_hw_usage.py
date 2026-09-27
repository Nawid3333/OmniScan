"""Tests for omniscan.hw.usage: what each usage level means and what applying it changes (CPU only)."""

from __future__ import annotations

from typing import Any

import psutil
import pytest
import torch

from omniscan.core.config import Config, GpuConfig
from omniscan.gpu.codec.select import get_codec
from omniscan.gpu.codec.turbo import TurboCodec
from omniscan.hw import usage
from omniscan.hw.usage import USAGE_LEVELS, apply_process_limits, usage_limits, vram_budget_gib


def test_levels_scale_cpu_threads_and_vram_share() -> None:
    assert usage_limits("full", 16) == usage.UsageLimits("full", 16, 1.0, False)
    assert usage_limits("balanced", 16) == usage.UsageLimits("balanced", 12, 0.75, False)
    assert usage_limits("background", 16) == usage.UsageLimits("background", 4, 0.5, True)


def test_a_single_core_machine_keeps_one_thread() -> None:
    for level in USAGE_LEVELS:
        assert usage_limits(level, 1).cpu_threads == 1


def test_vram_budget_is_capped_by_the_levels_share_of_the_gpu() -> None:
    assert vram_budget_gib(usage_limits("full", 8), 14.5, 16.0) == 14.5
    assert vram_budget_gib(usage_limits("balanced", 8), 14.5, 16.0) == 12.0
    assert vram_budget_gib(usage_limits("background", 8), 14.5, 16.0) == 8.0
    assert (
        vram_budget_gib(usage_limits("background", 8), 4.0, 16.0) == 4.0
    )  # a smaller configured budget wins
    assert vram_budget_gib(usage_limits("background", 8), 14.5, None) == 14.5  # CPU/MPS: nothing to cap


def test_full_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any) -> None:
        raise AssertionError("full must not touch process limits")

    monkeypatch.setattr(torch, "set_num_threads", fail)
    monkeypatch.setattr(usage, "_lower_priority", fail)
    apply_process_limits(usage_limits("full", 8))


def test_balanced_limits_threads_but_keeps_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    threads: list[int] = []
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    monkeypatch.setattr(usage, "_lower_priority", lambda: pytest.fail("balanced must keep the priority"))
    apply_process_limits(usage_limits("balanced", 8))
    assert threads == [6]


def test_background_limits_threads_and_lowers_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    threads: list[int] = []
    lowered: list[bool] = []
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    monkeypatch.setattr(usage, "_lower_priority", lambda: lowered.append(True))
    apply_process_limits(usage_limits("background", 8))
    assert threads == [2]
    assert lowered == [True]


def test_lower_priority_uses_nice_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int | None] = []

    class FakeProcess:
        def nice(self, value: int | None = None) -> int:
            calls.append(value)
            return 0

    monkeypatch.setattr(usage.sys, "platform", "linux")
    monkeypatch.setattr(psutil, "Process", FakeProcess)
    usage._lower_priority()
    assert calls == [None, 10]


def test_codec_workers_follow_the_usage_level() -> None:
    background = get_codec(Config(gpu=GpuConfig(device="cpu", usage="background")))
    assert isinstance(background, TurboCodec)
    assert background._max_workers == usage_limits("background").cpu_threads

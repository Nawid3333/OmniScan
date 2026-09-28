"""pipeline/on_demand.group_models: one model group for one Studio action, under the GPU lock. The VRAM manager
and the lock are stand-ins; omniscan.gpu.groups imports torch, so a stand-in module takes its place and the test
runs without torch."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

import omniscan.gpu.lock as gpu_lock
from omniscan.core.config import Config, GpuConfig
from omniscan.pipeline.on_demand import group_models


class Manager:
    def __init__(self) -> None:
        self.log: list[object] = []

    def acquire(self, group: str) -> dict[str, str]:
        self.log.append(("acquire", group))
        return {"model": group}

    def release(self) -> None:
        self.log.append("release")


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> tuple[Manager, list[object]]:
    manager, locks = Manager(), []
    groups = ModuleType("omniscan.gpu.groups")
    groups.__dict__["build_vram_manager"] = lambda cfg: manager
    monkeypatch.setitem(sys.modules, "omniscan.gpu.groups", groups)
    monkeypatch.setattr(gpu_lock, "acquire_gpu_lock", lambda: locks.append("lock") or "handle")
    monkeypatch.setattr(gpu_lock, "release_gpu_lock", lambda handle: locks.append(("unlock", handle)))
    return manager, locks


def test_a_group_is_loaded_under_the_gpu_lock_and_released(events: tuple[Manager, list[object]]) -> None:
    manager, locks = events
    with group_models(Config(), "vision") as models:
        assert models == {"model": "vision"} and locks == ["lock"]
    assert manager.log == [("acquire", "vision"), "release"] and locks == ["lock", ("unlock", "handle")]


def test_everything_is_released_when_the_action_fails(events: tuple[Manager, list[object]]) -> None:
    manager, locks = events
    with pytest.raises(RuntimeError, match="the OCR broke"), group_models(Config(), "inpaint"):
        raise RuntimeError("the OCR broke")
    assert manager.log == [("acquire", "inpaint"), "release"] and locks == ["lock", ("unlock", "handle")]


def test_a_cpu_config_takes_no_gpu_lock(events: tuple[Manager, list[object]]) -> None:
    manager, locks = events
    with group_models(Config(gpu=GpuConfig(device="cpu")), "vision"):
        pass
    assert locks == [] and manager.log == [("acquire", "vision"), "release"]

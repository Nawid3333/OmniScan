"""G3 stage prefetching: `_prefetch_groups` queues later groups' loads, first group and doubles skipped."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import omniscan.pipeline.runner as runner_module
from omniscan.core.config import Config, GpuConfig, OllamaConfig, PathsConfig


class SpyPrefetchScheduler:
    """Records prefetch calls; satisfies GpuScheduler plus the runner's PrefetchScheduler protocol."""

    def __init__(self) -> None:
        self.prefetched: list[str] = []

    def prefetch(self, group: str) -> bool:
        self.prefetched.append(group)
        return True

    def acquire(self, group: str) -> dict[str, Any]:
        return {}

    def reset_peak(self) -> None:
        pass

    def peak_gib(self) -> float:
        return 0.0


_GROUPS: dict[str, str | None] = {
    "ingest": None,
    "slice": None,
    "detect": "vision",
    "ocr": "vision",
    "translate": "ollama_local",
    "inpaint_lama": "inpaint",
}

_PASSES = [
    ("vision", ["ingest", "slice", "detect", "ocr"]),
    ("text", ["translate"]),
    ("render", ["inpaint_lama"]),
]


def prefetch_cfg(tmp_path: Path, *, warmup: bool) -> Config:
    """CPU-only config with all paths under tmp_path; only `gpu.warmup` matters to the planner."""
    return Config(
        gpu=GpuConfig(device="cpu", warmup=warmup),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
        ollama=OllamaConfig(local_url="http://127.0.0.1:9"),
    )


def fake_build_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_stage doubles carrying only the gpu_group the prefetch planner reads."""
    monkeypatch.setattr(
        runner_module,
        "build_stage",
        lambda name, cfg, *, client=None: SimpleNamespace(gpu_group=_GROUPS[name]),
    )


def test_prefetch_groups_queues_every_group_but_the_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_build_stage(monkeypatch)
    spy = SpyPrefetchScheduler()

    runner_module._prefetch_groups(spy, _PASSES, prefetch_cfg(tmp_path, warmup=True), None)

    assert spy.prefetched == ["inpaint"]  # vision (the first group) loads synchronously at its acquire


def test_prefetch_groups_needs_warmup_and_a_scheduler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_build_stage(monkeypatch)
    spy = SpyPrefetchScheduler()

    runner_module._prefetch_groups(spy, _PASSES, prefetch_cfg(tmp_path, warmup=False), None)
    assert spy.prefetched == []  # gpu.warmup off: no background loads

    plain: Any = object()  # a scheduler without prefetch (e.g. CPU/MPS) is skipped
    runner_module._prefetch_groups(plain, _PASSES, prefetch_cfg(tmp_path, warmup=True), None)
    assert spy.prefetched == []


def test_prefetch_groups_skips_ollama_group(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_build_stage(monkeypatch)
    spy = SpyPrefetchScheduler()

    runner_module._prefetch_groups(spy, _PASSES, prefetch_cfg(tmp_path, warmup=True), None)

    assert spy.prefetched == ["inpaint"]  # ollama_local is not a registered torch group

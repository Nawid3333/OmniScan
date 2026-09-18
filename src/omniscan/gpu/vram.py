"""VRAM manager: exactly one model group resident at a time, local Ollama models evicted before torch groups load.

Groups are registered with a loader returning named models (already on the device) and an estimated size.
`acquire(group)` is sticky: acquiring the resident group again is free, so a pass over 200 chapters loads each model
group once. The special group `OLLAMA_GROUP` means "free all torch memory so Windows Ollama can load a local LLM".
WDDM can silently page VRAM to system RAM instead of failing, so the budget is enforced here, not by the driver.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
import torch

from omniscan.gpu.device import resolve_device

log = logging.getLogger(__name__)

OLLAMA_GROUP = "ollama_local"
GIB = float(2**30)


@dataclass(frozen=True, slots=True)
class GroupSpec:
    loader: Callable[[torch.device], Mapping[str, Any]]
    est_gib: float


class VramManager:
    def __init__(
        self,
        device: str | torch.device = "auto",
        budget_gib: float | None = None,
        *,
        ollama_url: str | None = "http://localhost:11434",
        http: httpx.Client | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self._groups: dict[str, GroupSpec] = {}
        self._resident: str | None = None
        self._models: Mapping[str, Any] = {}
        self._ollama_url = ollama_url
        self._http = http
        self.budget_gib = budget_gib
        if self.is_gpu and budget_gib is not None:
            total = torch.cuda.get_device_properties(self.device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, budget_gib * GIB / total), self.device)

    @property
    def is_gpu(self) -> bool:
        return self.device.type == "cuda"

    @property
    def resident(self) -> str | None:
        return self._resident

    def register(
        self, group: str, loader: Callable[[torch.device], Mapping[str, Any]], est_gib: float
    ) -> None:
        """Register a model group; the loader must put every model on the given device in eval mode."""
        if group == OLLAMA_GROUP:
            raise ValueError(f"{OLLAMA_GROUP!r} is reserved")
        self._groups[group] = GroupSpec(loader, est_gib)

    def acquire(self, group: str) -> Mapping[str, Any]:
        """Make `group` the only resident group and return its models."""
        if group == self._resident:
            return self._models
        if group != OLLAMA_GROUP and group not in self._groups:
            raise KeyError(f"unknown model group {group!r}")
        self.release()
        if group == OLLAMA_GROUP:
            self._resident = OLLAMA_GROUP
            return self._models
        self.evict_ollama()
        spec = self._groups[group]
        free = self.free_gib()
        if free is not None and free < spec.est_gib:
            log.warning("group %s needs ~%.1f GiB but only %.1f GiB free", group, spec.est_gib, free)
        log.info("loading model group %s (~%.1f GiB)", group, spec.est_gib)
        self._models = spec.loader(self.device)
        self._resident = group
        return self._models

    def release(self) -> None:
        """Drop the resident group and return its memory to the driver."""
        if self._resident is not None:
            log.debug("releasing model group %s", self._resident)
        self._models = {}
        self._resident = None
        gc.collect()
        if self.is_gpu:
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

    def evict_ollama(self) -> list[str]:
        """Unload local (VRAM-resident) Ollama models so torch can use the GPU; returns unloaded names."""
        if self._ollama_url is None:
            return []
        client = self._http or httpx.Client(timeout=10.0)
        unloaded: list[str] = []
        try:
            running = client.get(f"{self._ollama_url}/api/ps").json().get("models", [])
            for model in running:
                if model.get("size_vram", 0) > 0:
                    client.post(
                        f"{self._ollama_url}/api/generate", json={"model": model["name"], "keep_alive": 0}
                    ).raise_for_status()
                    unloaded.append(model["name"])
        except httpx.HTTPError as exc:
            log.warning("could not query/evict Ollama models: %s", exc)
        finally:
            if self._http is None:
                client.close()
        if unloaded:
            log.info("evicted local Ollama models from VRAM: %s", ", ".join(unloaded))
        return unloaded

    def free_gib(self) -> float | None:
        if not self.is_gpu:
            return None
        free, _total = torch.cuda.mem_get_info(self.device)
        return free / GIB

    def reset_peak(self) -> None:
        if self.is_gpu:
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_gib(self) -> float:
        return torch.cuda.max_memory_allocated(self.device) / GIB if self.is_gpu else 0.0

"""VRAM manager: one model group resident at a time, local Ollama models evicted only when VRAM gets tight.

Groups are registered with a loader returning named models (already on the device) and an estimated size.
`acquire(group)` is sticky: acquiring the resident group again is free, so a pass over 200 chapters loads each model
group once. The special group `OLLAMA_GROUP` means "free all torch memory so Windows Ollama can load a local LLM".
WDDM can silently page VRAM to system RAM instead of failing, so the budget is enforced here, not by the driver.

A group can be preloaded with `prefetch(group)` on a background thread before anything acquires it: the loader runs
while earlier stages compute, and the later `acquire` then only frees the previous group and adopts the prefetched
models (the two groups share VRAM for that window; `prefetch` skips itself when a load would not leave enough free).
Queued prefetches load one at a time on a single worker, in request order — concurrent loaders starve each other's
library init and the pipeline's CPU stages — and the worker waits for the first `acquire` before running a loader,
so no MIOpen/kernel-find work overlaps the pipeline's own startup decode.
"""

from __future__ import annotations

import gc
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
import torch

from omniscan.gpu.device import resolve_device
from omniscan.gpu.timeline import mark

log = logging.getLogger(__name__)

OLLAMA_GROUP = "ollama_local"
GIB = float(2**30)
# Evict local Ollama models only when loading a group would leave less free VRAM than this margin; a
# resident local LLM can stay otherwise (it is reused by later translate/judge passes without a reload).
_EVICT_MARGIN_GIB = 4.0
# Do not start a prefetch unless loading the group would leave at least this much free VRAM untouched.
_PREFETCH_MARGIN_GIB = 1.0


@dataclass(frozen=True, slots=True)
class GroupSpec:
    loader: Callable[[torch.device], Mapping[str, Any]]
    est_gib: float


@dataclass
class _Prefetch:
    """One queued or running background load: `models` stays None until it finished (or on failure)."""

    event: threading.Event = field(default_factory=threading.Event)
    models: Mapping[str, Any] | None = None
    error: str | None = None


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
        self._prefetches: dict[str, _Prefetch] = {}
        self._prefetch_lock = threading.Lock()
        self._prefetch_queue: list[tuple[str, GroupSpec, _Prefetch]] = []
        self._prefetch_worker: threading.Thread | None = None
        self._first_acquire = threading.Event()
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

    @property
    def first_acquire_event(self) -> threading.Event:
        """Set once any group was first acquired; the GPU warm-up thread waits for it (startup decode first)."""
        return self._first_acquire

    def register(
        self, group: str, loader: Callable[[torch.device], Mapping[str, Any]], est_gib: float
    ) -> None:
        """Register a model group; the loader must put every model on the given device in eval mode."""
        if group == OLLAMA_GROUP:
            raise ValueError(f"{OLLAMA_GROUP!r} is reserved")
        self._groups[group] = GroupSpec(loader, est_gib)

    def prefetch(self, group: str) -> bool:
        """Queue loading `group` on the background worker so a later `acquire` can adopt it; True when queued.

        A no-op when the group is already resident, a prefetch is queued/running for it, or the load would not
        leave `_PREFETCH_MARGIN_GIB` free. The models are never made resident here; the loader failing is only
        logged (the later `acquire` falls back to a synchronous load).
        """
        if self._resident == group:
            return False
        with self._prefetch_lock:
            if group in self._prefetches:
                return False
            spec = self._groups.get(group)
            if spec is None:
                return False
            free = self.free_gib()
            if free is not None and free < spec.est_gib + _PREFETCH_MARGIN_GIB:
                log.info(
                    "prefetch of %s skipped: %.1f GiB free is below its ~%.1f GiB estimate",
                    group,
                    free,
                    spec.est_gib,
                )
                return False
            running = _Prefetch()
            self._prefetches[group] = running
            self._prefetch_queue.append((group, spec, running))
            if self._prefetch_worker is None:
                self._prefetch_worker = threading.Thread(
                    target=self._prefetch_run_all, name="prefetch-models", daemon=True
                )
                self._prefetch_worker.start()
        return True

    def _prefetch_run_all(self) -> None:
        """The prefetch worker body: run the queued loaders one at a time, never raising.

        Each loader waits for the first `acquire`: until then the pipeline is doing its own startup work
        (strip decode, GPU library init), which MIOpen/kernel-find phases would starve.
        """
        while True:
            with self._prefetch_lock:
                if not self._prefetch_queue:
                    self._prefetch_worker = None
                    return
                group, spec, running = self._prefetch_queue.pop(0)
            self._first_acquire.wait()
            try:
                running.models = spec.loader(self.device)
                log.info("prefetched model group %s", group)
                mark(f"prefetch {group} done")
            except Exception as exc:
                running.error = f"{type(exc).__name__}: {exc}"
                log.warning("prefetch of %s failed: %s", group, running.error)
                with self._prefetch_lock:
                    if self._prefetches.get(group) is running:
                        del self._prefetches[group]
            finally:
                running.event.set()

    def _take_prefetch(self, group: str) -> Mapping[str, Any] | None:
        """The models of `group`'s prefetch, waiting for it to finish (None when absent or failed)."""
        with self._prefetch_lock:
            running = self._prefetches.get(group)
        if running is None:
            return None
        running.event.wait()
        with self._prefetch_lock:
            if self._prefetches.get(group) is running:
                del self._prefetches[group]
        return running.models

    def acquire(self, group: str) -> Mapping[str, Any]:
        """Make `group` the only resident group and return its models."""
        if group == self._resident:
            return self._models
        if group != OLLAMA_GROUP and group not in self._groups:
            raise KeyError(f"unknown model group {group!r}")
        mark(f"acquire {group} begin")
        self._first_acquire.set()
        self.release()
        if group == OLLAMA_GROUP:
            self._resident = OLLAMA_GROUP
            return self._models
        prefetched = self._take_prefetch(group)
        if prefetched is not None:
            self._models = prefetched
            self._resident = group
            mark(f"acquire {group} end (prefetched)")
            return self._models
        mark(f"acquire {group}: previous group released")
        spec = self._groups[group]
        self._ensure_vram(spec)
        mark(f"acquire {group}: ollama evicted")
        free = self.free_gib()
        if free is not None and free < spec.est_gib:
            log.warning("group %s needs ~%.1f GiB but only %.1f GiB free", group, spec.est_gib, free)
        log.info("loading model group %s (~%.1f GiB)", group, spec.est_gib)
        self._models = spec.loader(self.device)
        self._resident = group
        mark(f"acquire {group} end")
        return self._models

    def _ensure_vram(self, spec: GroupSpec) -> None:
        """Evict local Ollama models only when loading `spec` would leave less than the eviction margin free.

        A scheduler without VRAM information (CPU/MPS) keeps the conservative always-evict behaviour.
        """
        free = self.free_gib()
        if free is None or free < spec.est_gib + _EVICT_MARGIN_GIB:
            self.evict_ollama()

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
        mark("release done")

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

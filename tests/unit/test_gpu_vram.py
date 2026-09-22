import json
from collections.abc import Mapping
from typing import Any

import httpx
import pytest
import torch

from omniscan.gpu.device import resolve_device
from omniscan.gpu.vram import OLLAMA_GROUP, VramManager


def ollama_client(ps_models: list[dict[str, Any]], calls: list[tuple[str, Any]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.url.path, body))
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": ps_models})
        return httpx.Response(200, json={"done": True})

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ollama")


def make_manager(calls: list[tuple[str, Any]], ps: list[dict[str, Any]] | None = None) -> VramManager:
    return VramManager("cpu", ollama_url="http://ollama", http=ollama_client(ps or [], calls))


def loading_manager(
    loads: list[str], *, fail_first_load: str | None = None, free: float | None = None
) -> VramManager:
    """A CPU manager with loaders recording `loads`; optionally a failing loader / faked free VRAM."""

    class FakeFreeManager(VramManager):
        def free_gib(self) -> float | None:
            return free

    def loader(name: str):
        def load(device: torch.device) -> Mapping[str, Any]:
            loads.append(name)
            if fail_first_load == name and loads.count(name) == 1:
                raise RuntimeError(f"{name} load failed")
            return {name: torch.zeros(1, device=device)}

        return load

    vm = FakeFreeManager("cpu", ollama_url="http://ollama", http=ollama_client([], []))
    vm.register("a", loader("a"), est_gib=3)
    vm.register("b", loader("b"), est_gib=2)
    vm.register("c", loader("c"), est_gib=2)
    return vm


def wait_prefetch(vm: VramManager) -> None:
    """Join the prefetch worker (None when it already exited)."""
    worker = vm._prefetch_worker
    if worker is not None:
        worker.join(10.0)


def test_acquire_is_sticky_and_switches() -> None:
    loads: list[str] = []

    def loader(name: str):
        def load(device: torch.device) -> Mapping[str, Any]:
            loads.append(name)
            return {name: torch.zeros(1, device=device)}

        return load

    vm = make_manager([])
    vm.register("vision", loader("vision"), est_gib=4)
    vm.register("inpaint", loader("inpaint"), est_gib=3)
    assert "vision" in vm.acquire("vision")
    vm.acquire("vision")
    assert loads == ["vision"]
    assert "inpaint" in vm.acquire("inpaint")
    assert vm.resident == "inpaint"
    assert loads == ["vision", "inpaint"]


def test_ollama_group_releases_torch_models() -> None:
    vm = make_manager([])
    vm.register("vision", lambda d: {"m": torch.zeros(1)}, est_gib=1)
    vm.acquire("vision")
    assert vm.acquire(OLLAMA_GROUP) == {}
    assert vm.resident == OLLAMA_GROUP


def test_evicts_only_vram_resident_ollama_models() -> None:
    calls: list[tuple[str, Any]] = []
    ps = [
        {"name": "translategemma:12b", "size_vram": 9_000_000_000},
        {"name": "gemma4:31b-cloud", "size_vram": 0},
    ]
    vm = make_manager(calls, ps)
    vm.register("vision", lambda d: {}, est_gib=1)
    vm.acquire("vision")
    generate_calls = [body for path, body in calls if path == "/api/generate"]
    assert generate_calls == [{"model": "translategemma:12b", "keep_alive": 0}]


def test_ollama_unreachable_does_not_block_loading() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    vm = VramManager(
        "cpu", ollama_url="http://ollama", http=httpx.Client(transport=httpx.MockTransport(handler))
    )
    vm.register("vision", lambda d: {"ok": True}, est_gib=1)
    assert vm.acquire("vision") == {"ok": True}


def test_unknown_and_reserved_groups() -> None:
    vm = make_manager([])
    with pytest.raises(KeyError):
        vm.acquire("nope")
    with pytest.raises(ValueError):
        vm.register(OLLAMA_GROUP, lambda d: {}, est_gib=1)


# ---------------------------------------------------------------- prefetch (card G3)


def test_prefetch_runs_on_a_worker_and_acquire_adopts_the_models() -> None:
    loads: list[str] = []
    vm = loading_manager(loads)
    assert vm.prefetch("b") is True
    assert loads == []  # nothing loads until the first acquire

    assert "a" in vm.acquire("a")  # the first acquire releases the worker
    wait_prefetch(vm)
    assert sorted(loads) == ["a", "b"]  # the worker loaded b behind the first acquire's own load

    assert "b" in vm.acquire("b")  # adopts the prefetched models: no second load
    assert loads.count("b") == 1
    assert vm.resident == "b"


def test_prefetch_waits_for_the_first_acquire() -> None:
    loads: list[str] = []
    vm = loading_manager(loads)
    assert vm.prefetch("b") is True
    assert loads == []  # the worker waits for the first acquire: no loader overlaps the startup decode

    assert "a" in vm.acquire("a")
    wait_prefetch(vm)
    assert sorted(loads) == ["a", "b"]


def test_prefetches_run_one_at_a_time_in_request_order() -> None:
    loads: list[str] = []
    vm = loading_manager(loads)
    assert vm.prefetch("c") is True
    assert vm.prefetch("b") is True
    assert "a" in vm.acquire("a")
    wait_prefetch(vm)
    assert sorted(loads) == ["a", "b", "c"]
    assert loads.index("c") < loads.index("b")  # the worker's own loads keep the queue order


def test_prefetch_is_a_noop_when_resident_queued_or_unknown() -> None:
    loads: list[str] = []
    vm = loading_manager(loads)
    assert vm.prefetch("b") is True
    assert vm.prefetch("b") is False  # already queued
    assert vm.prefetch("nope") is False  # not registered
    assert vm.prefetch(OLLAMA_GROUP) is False  # reserved

    assert "a" in vm.acquire("a")
    wait_prefetch(vm)
    assert vm.prefetch("a") is False  # already resident
    assert "b" in vm.acquire("b")
    assert sorted(loads) == ["a", "b"]  # the queued prefetch was adopted, never loaded twice


def test_prefetch_skipped_when_it_would_not_leave_the_margin_free() -> None:
    loads: list[str] = []
    vm = loading_manager(loads, free=2.0)  # below est 2 + 1 margin, above est alone
    assert vm.prefetch("b") is False  # would not leave _PREFETCH_MARGIN_GIB free
    assert vm._prefetch_worker is None  # nothing was queued or started
    assert "b" in vm.acquire("b")  # the synchronous load still works
    assert loads == ["b"]


def test_prefetch_failure_is_logged_and_acquire_falls_back() -> None:
    loads: list[str] = []
    vm = loading_manager(loads, fail_first_load="b")
    assert vm.prefetch("b") is True
    assert "a" in vm.acquire("a")
    wait_prefetch(vm)
    assert sorted(loads) == ["a", "b"]  # the worker's attempt failed

    assert "b" in vm.acquire("b")  # falls back to a synchronous load
    assert sorted(loads) == ["a", "b", "b"]
    assert vm.resident == "b"


def test_evicts_ollama_only_below_the_load_margin() -> None:
    """G3: a resident local LLM stays while loading leaves more than the eviction margin free."""

    def manager_with(free: float) -> tuple[VramManager, list[tuple[str, Any]]]:
        calls: list[tuple[str, Any]] = []
        ps = [{"name": "llm", "size_vram": 9_000_000_000}]

        class FakeFreeManager(VramManager):
            def free_gib(self) -> float | None:
                return free

        vm = FakeFreeManager("cpu", ollama_url="http://ollama", http=ollama_client(ps, calls))
        vm.register("a", lambda d: {"a": torch.zeros(1)}, est_gib=3)
        return vm, calls

    vm, calls = manager_with(20.0)  # est 3 + 4 margin: the LLM stays resident
    assert "a" in vm.acquire("a")
    assert [body for path, body in calls if path == "/api/generate"] == []

    vm, calls = manager_with(5.0)  # below the margin: evict before loading
    assert "a" in vm.acquire("a")
    assert [body for path, body in calls if path == "/api/generate"] == [{"model": "llm", "keep_alive": 0}]


@pytest.mark.gpu
def test_real_gpu_release_returns_memory() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    vm = VramManager(resolve_device(), ollama_url=None)
    vm.register("big", lambda d: {"t": torch.empty(2 * 2**30, dtype=torch.uint8, device=d)}, est_gib=2)
    before = vm.free_gib()
    vm.acquire("big")
    during = vm.free_gib()
    vm.release()
    after = vm.free_gib()
    assert before is not None and during is not None and after is not None
    assert during < before - 1.5
    assert after > during + 1.5

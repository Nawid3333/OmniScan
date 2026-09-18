import json
from collections.abc import Mapping
from typing import Any

import httpx
import pytest
import torch

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


@pytest.mark.gpu
def test_real_gpu_release_returns_memory() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    vm = VramManager("cuda:0", ollama_url=None)
    vm.register("big", lambda d: {"t": torch.empty(2 * 2**30, dtype=torch.uint8, device=d)}, est_gib=2)
    before = vm.free_gib()
    vm.acquire("big")
    during = vm.free_gib()
    vm.release()
    after = vm.free_gib()
    assert before is not None and during is not None and after is not None
    assert during < before - 1.5
    assert after > during + 1.5

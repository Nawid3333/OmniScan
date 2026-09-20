"""Tests for the Qt-free ModelsService (card U3a part 2): rows, download, remove, download_required."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omniscan.core.config import Config, PathsConfig
from omniscan.gui.services import models as services_models
from omniscan.gui.services.models import ModelsService
from omniscan.models.catalog import ModelEntry
from omniscan.models.download import ModelDownloadError
from omniscan.models.store import MARKER_NAME

SHA = "a" * 64


def entry(**overrides: Any) -> ModelEntry:
    """One zip catalog entry with the given overrides."""
    base: dict[str, Any] = {
        "id": "m",
        "name": "M",
        "kind": "vision",
        "format": "zip",
        "size_mb": 10,
        "license": "Apache-2.0",
        "description": "d",
        "used_by": ["detect"],
        "role": "detector",
        "mirror_url": "https://mirror/x.zip",
        "sha256": SHA,
        "bytes": 100,
        "upstream_repo": "org/x",
        "upstream_revision": "rev1",
    }
    base.update(overrides)
    return ModelEntry(**base)


def make_cfg(tmp_path: Path) -> Config:
    return Config(paths=PathsConfig(models_dir=tmp_path / "models"))


def install_zip(cfg: Config, model_id: str) -> None:
    folder = cfg.paths.models_dir / model_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MARKER_NAME).write_text(json.dumps({"id": model_id, "sha256": SHA}), encoding="utf-8")


def patch_catalog(monkeypatch: pytest.MonkeyPatch, entries: list[ModelEntry]) -> None:
    monkeypatch.setattr(services_models, "load_catalog", lambda: entries)


class FakeHW:
    """Stand-in HardwareInfo: rows() must return it untouched."""

    best_device = "cpu"
    gpus: tuple[Any, ...] = ()
    ram_gb = 32.0
    disk_free_gb = 500.0


# ---------------------------------------------------------------- rows


def test_rows_returns_rows_and_injected_hardware(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(tmp_path)
    entries = [entry(id="a", role="detector"), entry(id="b", role="recognizer", langs=["ko"])]
    patch_catalog(monkeypatch, entries)
    rows, hw = ModelsService(cfg, ollama_names=lambda: None, hardware=lambda: FakeHW()).rows(lang="ko")
    assert [row.id for row in rows] == ["b"]  # lang filter applied, catalog order kept
    assert isinstance(hw, FakeHW)
    assert rows[0].status == "missing"  # ollama_names None = unreachable, not a daemon query


def test_default_providers_query_daemon_and_detect_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(tmp_path)
    patch_catalog(monkeypatch, [entry(id="a")])
    seen: dict[str, Any] = {}

    def fake_names(url: str) -> set[str]:
        seen["url"] = url
        return {"x:1b"}

    def fake_detect(models_dir: Path) -> Any:
        seen["models_dir"] = str(models_dir)
        return FakeHW()

    monkeypatch.setattr(services_models, "ollama_model_names", fake_names)
    monkeypatch.setattr(services_models, "detect_hardware", fake_detect)
    rows, hw = ModelsService(cfg).rows()
    assert seen == {"url": cfg.ollama.local_url, "models_dir": str(cfg.paths.models_dir)}
    assert isinstance(hw, FakeHW)
    assert [row.id for row in rows] == ["a"]


# ---------------------------------------------------------------- download and remove


def test_download_calls_download_model_with_entry_and_urls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(tmp_path)
    entry_a, entry_b = entry(id="a"), entry(id="b")
    patch_catalog(monkeypatch, [entry_a, entry_b])
    calls: list[tuple[ModelEntry, Path, dict[str, Any]]] = []

    def fake_download(entry: ModelEntry, models_dir: Path, **kwargs: Any) -> str:
        calls.append((entry, models_dir, kwargs))
        return "mirror"

    monkeypatch.setattr(services_models, "download_model", fake_download)
    assert ModelsService(cfg).download("b") == "mirror"
    called_entry, called_dir, kwargs = calls[0]
    assert called_entry is entry_b
    assert called_dir == cfg.paths.models_dir
    assert kwargs["ollama_url"] == cfg.ollama.local_url
    assert kwargs["on_progress"] is None


def test_download_adapts_the_progress_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_catalog(monkeypatch, [entry(id="a")])
    seen: list[tuple[int, int | None]] = []

    def fake_download(entry: ModelEntry, models_dir: Path, **kwargs: Any) -> str:
        kwargs["on_progress"]("ignored-id", 10, 20)  # download_model's (id, done, total) shape
        kwargs["on_progress"]("ignored-id", 5, None)
        return "ollama"

    monkeypatch.setattr(services_models, "download_model", fake_download)
    service = ModelsService(make_cfg(tmp_path))
    assert service.download("a", on_progress=lambda d, t: seen.append((d, t))) == "ollama"
    assert seen == [(10, 20), (5, None)]


def test_remove_calls_remove_model_with_entry_and_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(tmp_path)
    entry_a = entry(id="a")
    patch_catalog(monkeypatch, [entry_a])
    calls: list[tuple[ModelEntry, Path, dict[str, Any]]] = []

    def fake_remove(entry: ModelEntry, models_dir: Path, **kwargs: Any) -> bool:
        calls.append((entry, models_dir, kwargs))
        return True

    monkeypatch.setattr(services_models, "remove_model", fake_remove)
    assert ModelsService(cfg).remove("a") is True
    called_entry, called_dir, kwargs = calls[0]
    assert called_entry is entry_a
    assert called_dir == cfg.paths.models_dir
    assert kwargs == {"ollama_url": cfg.ollama.local_url}


def test_unknown_id_raises_value_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_catalog(monkeypatch, [entry(id="a")])
    service = ModelsService(make_cfg(tmp_path))
    with pytest.raises(ValueError, match="unknown model 'nope'"):
        service.download("nope")
    with pytest.raises(ValueError, match="unknown model 'nope'"):
        service.remove("nope")


def test_download_error_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_catalog(monkeypatch, [entry(id="a")])

    def failing(entry: ModelEntry, models_dir: Path, **kwargs: Any) -> str:
        raise ModelDownloadError(f"{entry.id}: all sources failed")

    monkeypatch.setattr(services_models, "download_model", failing)
    with pytest.raises(ModelDownloadError, match="all sources failed"):
        ModelsService(make_cfg(tmp_path)).download("a")


# ---------------------------------------------------------------- download_required


def test_download_required_downloads_missing_in_catalog_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(tmp_path)
    entries = [
        entry(id="opt", required=False),
        entry(id="req-bad", required=True),
        entry(id="req-installed", required=True),
        entry(id="req-ok", required=True),
    ]
    install_zip(cfg, "req-installed")
    patch_catalog(monkeypatch, entries)
    pulled: list[str] = []

    def fake_download(entry: ModelEntry, models_dir: Path, **kwargs: Any) -> str:
        pulled.append(entry.id)
        if entry.id == "req-bad":
            raise ModelDownloadError(f"{entry.id}: mirror down")
        if kwargs.get("on_progress"):
            kwargs["on_progress"](entry.id, 50, 100)
        return "mirror"

    monkeypatch.setattr(services_models, "download_model", fake_download)
    seen: list[tuple[str, int, int | None]] = []
    results = ModelsService(cfg).download_required(on_progress=lambda *a: seen.append(a))
    assert results == [("req-bad", "req-bad: mirror down"), ("req-ok", None)]
    assert pulled == ["req-bad", "req-ok"]  # optional and installed are skipped
    assert seen == [("req-ok", 50, 100)]


def test_download_required_all_installed_downloads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(tmp_path)
    patch_catalog(monkeypatch, [entry(id="req-a", required=True)])
    install_zip(cfg, "req-a")

    def no_download(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("must not download an installed model")

    monkeypatch.setattr(services_models, "download_model", no_download)
    assert ModelsService(cfg).download_required() == []


def test_download_required_ollama_missing_is_pulled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`missing` (daemon reachable, tag absent) is pulled."""
    cfg = make_cfg(tmp_path)
    pulled = entry(id="pull-me", kind="llm", format="ollama", ollama_name="p:1b", required=True)
    patch_catalog(monkeypatch, [pulled])
    calls: list[str] = []
    monkeypatch.setattr(
        services_models, "download_model", lambda e, m, **k: (calls.append(e.id), "ollama")[1]
    )
    service = ModelsService(cfg, ollama_names=lambda: {"other:1b"})  # daemon reachable
    assert service.download_required() == [("pull-me", None)]
    assert calls == ["pull-me"]


def test_download_required_skips_unknown_when_daemon_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`unknown` (daemon unreachable) is not a missing/corrupt model: nothing is pulled."""
    cfg = make_cfg(tmp_path)
    patch_catalog(
        monkeypatch, [entry(id="unk", kind="llm", format="ollama", ollama_name="u:1b", required=True)]
    )

    def no_pull(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("an unknown status must not be pulled")

    monkeypatch.setattr(services_models, "download_model", no_pull)
    assert ModelsService(cfg, ollama_names=lambda: None).download_required() == []


def test_download_required_ollama_tag_present_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(tmp_path)
    patch_catalog(
        monkeypatch, [entry(id="pulled", kind="llm", format="ollama", ollama_name="p:1b", required=True)]
    )

    def no_pull(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("a pulled tag must not be downloaded again")

    monkeypatch.setattr(services_models, "download_model", no_pull)
    assert ModelsService(cfg, ollama_names=lambda: {"p:1b"}).download_required() == []

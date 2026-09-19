"""Tests for the `omniscan models` command group (card U2a, part 4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, PathsConfig
from omniscan.models.store import MARKER_NAME

runner = CliRunner()

CATALOG_TOML = """\
[[model]]
id = "det"
name = "Detector"
kind = "vision"
required = true
format = "zip"
size_mb = 10
license = "Apache-2.0"
description = "detector"
used_by = []
mirror_url = "https://mirror/det.zip"
sha256 = "{sha}"
bytes = 100
upstream_repo = "org/det"
upstream_revision = "rev1"

[[model]]
id = "lama"
name = "LaMa"
kind = "inpaint"
format = "file"
size_mb = 5
license = "Apache-2.0"
description = "inpainter"
used_by = []
mirror_url = "https://mirror/big-lama.pt"
sha256 = "{sha}"
bytes = 4
upstream_url = "https://upstream/big-lama.pt"
install_path = "lama/big-lama.pt"

[[model]]
id = "llm-x"
name = "X"
kind = "llm"
format = "ollama"
size_mb = 7
license = "Gemma terms"
description = "llm"
used_by = []
ollama_name = "x:1b"

[[model]]
id = "llm-c"
name = "C"
kind = "llm"
format = "cloud"
size_mb = 0
license = "Gemma terms"
description = "cloud llm"
used_by = []
ollama_name = "c:1b-cloud"
"""

SHA = "a" * 64
JSON_KEYS = (
    "id",
    "name",
    "kind",
    "required",
    "format",
    "size_mb",
    "license",
    "description",
    "used_by",
    "status",
    "installed_path",
)


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Temp models_dir + a small fake catalog; Ollama is treated as unreachable."""
    config = Config(paths=PathsConfig(models_dir=tmp_path / "models"))
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: config)
    monkeypatch.setattr(omniscan.cli, "_ollama_model_names", lambda *args: None)
    catalog_file = tmp_path / "catalog.toml"
    catalog_file.write_text(CATALOG_TOML.format(sha=SHA), encoding="utf-8")
    from omniscan.models import catalog as catalog_module

    monkeypatch.setattr(catalog_module, "default_catalog_path", lambda: catalog_file)
    monkeypatch.setattr(catalog_module, "machine_catalog_path", lambda: tmp_path / "machine.toml")
    return config


def install_det(cfg: Config, sha: str = SHA) -> None:
    folder = cfg.paths.models_dir / "det"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MARKER_NAME).write_text(json.dumps({"id": "det", "sha256": sha}), encoding="utf-8")


def fake_download(entry: Any, models_dir: Any, **kwargs: Any) -> str:
    """Stand-in for download_model: installs the folder + marker without network."""
    folder = models_dir / entry.id
    marker = folder / MARKER_NAME
    if marker.is_file():
        return "already installed"
    folder.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"id": entry.id, "sha256": entry.sha256, "source": "mirror"}), encoding="utf-8"
    )
    return "mirror"


# ---------------------------------------------------------------- list


def test_list_text_rows_and_footer(cfg: Config) -> None:
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "det  vision  10 MB  required  missing  detector" in result.output
    assert "lama  inpaint  5 MB  optional  missing  inpainter" in result.output
    assert "llm-x  llm  7 MB  optional  unknown  llm" in result.output  # Ollama unreachable
    assert "llm-c  llm  0 MB  optional  cloud  cloud llm" in result.output
    assert "1 required model(s) missing, 10 MB to download" in result.output


def test_list_json_keys_and_statuses(cfg: Config) -> None:
    install_det(cfg)
    result = runner.invoke(app, ["models", "list", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["models_dir"] == str(cfg.paths.models_dir)
    assert [m["id"] for m in payload["models"]] == ["det", "lama", "llm-x", "llm-c"]
    assert list(payload["models"][0]) == list(JSON_KEYS)
    by_id = {m["id"]: m for m in payload["models"]}
    assert by_id["det"]["status"] == "installed"
    assert by_id["det"]["installed_path"] == str(cfg.paths.models_dir / "det")
    assert by_id["lama"]["status"] == "missing" and by_id["lama"]["installed_path"] is None
    assert by_id["llm-x"]["status"] == "unknown" and by_id["llm-x"]["installed_path"] is None
    assert by_id["llm-c"]["status"] == "cloud"
    assert by_id["det"]["required"] is True and by_id["lama"]["required"] is False


def test_list_with_reachable_ollama(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(omniscan.cli, "_ollama_model_names", lambda *args: {"x:1b"})
    result = runner.invoke(app, ["models", "list"])
    assert "llm-x  llm  7 MB  optional  installed  llm" in result.output


# ---------------------------------------------------------------- download


def test_download_installs_and_is_listed_installed(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("omniscan.models.download.download_model", fake_download)
    result = runner.invoke(app, ["models", "download", "det"])
    assert result.exit_code == 0
    assert "det: installed from mirror" in result.output
    result = runner.invoke(app, ["models", "list", "--json"])
    by_id = {m["id"]: m for m in json.loads(result.output)["models"]}
    assert by_id["det"]["status"] == "installed"
    result = runner.invoke(app, ["models", "download", "det"])
    assert "det: already installed" in result.output


def test_download_unknown_id_exits_2(cfg: Config) -> None:
    result = runner.invoke(app, ["models", "download", "nope"])
    assert result.exit_code == 2
    assert "unknown model id(s) nope" in result.output
    assert "known: det, lama, llm-x, llm-c" in result.output


def test_download_required_adds_missing_required(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    downloaded: list[str] = []
    original = fake_download

    def spy(entry: Any, models_dir: Any, **kwargs: Any) -> str:
        downloaded.append(entry.id)
        return original(entry, models_dir, **kwargs)

    monkeypatch.setattr("omniscan.models.download.download_model", spy)
    result = runner.invoke(app, ["models", "download", "--required"])
    assert result.exit_code == 0
    assert downloaded == ["det"]  # the only required model, and it was missing
    assert "det: installed from mirror" in result.output


def test_download_required_skips_installed(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    install_det(cfg)
    downloaded: list[str] = []

    def spy(entry: Any, models_dir: Any, **kwargs: Any) -> str:
        downloaded.append(entry.id)
        return fake_download(entry, models_dir, **kwargs)

    monkeypatch.setattr("omniscan.models.download.download_model", spy)
    result = runner.invoke(app, ["models", "download", "--required"])
    assert result.exit_code == 0
    assert downloaded == []
    assert "det: already installed" in result.output


def test_download_failure_continues_and_exits_1(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    from omniscan.models.download import ModelDownloadError

    def failing(entry: Any, models_dir: Any, **kwargs: Any) -> str:
        raise ModelDownloadError(f"{entry.id}: mirror down, upstream down")

    monkeypatch.setattr("omniscan.models.download.download_model", failing)
    result = runner.invoke(app, ["models", "download", "det", "lama"])
    assert result.exit_code == 1
    assert "models: det: mirror down, upstream down" in result.output


def test_download_ollama_installed_skips_the_pull(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(omniscan.cli, "_ollama_model_names", lambda *args: {"x:1b"})

    def spying(entry: Any, models_dir: Any, **kwargs: Any) -> str:
        raise AssertionError("download_model must not be called for an installed ollama model")

    monkeypatch.setattr("omniscan.models.download.download_model", spying)
    result = runner.invoke(app, ["models", "download", "llm-x"])
    assert result.exit_code == 0
    assert "llm-x: already installed" in result.output


# ---------------------------------------------------------------- remove and verify


def test_remove(cfg: Config) -> None:
    install_det(cfg)
    result = runner.invoke(app, ["models", "remove", "det"])
    assert result.exit_code == 0
    assert "det: removed" in result.output
    assert not (cfg.paths.models_dir / "det").exists()
    result = runner.invoke(app, ["models", "remove", "det"])
    assert result.exit_code == 0
    assert "det: nothing to remove" in result.output


def test_remove_unknown_id_exits_2(cfg: Config) -> None:
    result = runner.invoke(app, ["models", "remove", "nope"])
    assert result.exit_code == 2


def test_verify_corrupt_file_exits_1(cfg: Config) -> None:
    target = cfg.paths.models_dir / "lama" / "big-lama.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"too long")  # wrong size -> corrupt
    result = runner.invoke(app, ["models", "verify", "lama"])
    assert result.exit_code == 1
    assert "lama: corrupt" in result.output


def test_verify_installed_exits_0(cfg: Config) -> None:
    install_det(cfg)
    result = runner.invoke(app, ["models", "verify", "det"])
    assert result.exit_code == 0
    assert "det: installed" in result.output


def test_verify_default_covers_non_llm(cfg: Config) -> None:
    result = runner.invoke(app, ["models", "verify"])
    assert result.exit_code == 0
    assert "det: missing" in result.output
    assert "lama: missing" in result.output
    assert "llm-x" not in result.output  # llm models are excluded by default
    assert "llm-c" not in result.output


# ---------------------------------------------------------------- progress printer


def test_progress_prints_once_per_five_percent_step(capsys: pytest.CaptureFixture[str]) -> None:
    from omniscan.cli import _models_progress

    on_progress = _models_progress({})
    total = 100_000_000
    for done in (0, 1, 4_999_999, 5_000_000, 9_999_999, total):
        on_progress("det", done, total)
    out = capsys.readouterr().out
    assert out == "det: 0% (0/100 MB)\ndet: 5% (5/100 MB)\ndet: 100% (100/100 MB)\n"
    on_progress("det", 50_000_000, None)  # unknown total: no line at all
    assert capsys.readouterr().out == ""

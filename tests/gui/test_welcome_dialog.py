"""WelcomeDialog tests (offscreen): the five-step first-run wizard with fake hardware, models, Ollama and runtime."""

from __future__ import annotations

import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from omniscan.core.config import Config, PathsConfig, ShareConfig, UserConfig
from omniscan.gui.services.hardware import HardwareReport
from omniscan.gui.welcome_dialog import STEPS, WelcomeDialog
from omniscan.hw.profiles import profile
from omniscan.hw.tune import plan_for
from omniscan.runtime import Recommendation


class FakeHardware:
    """HardwareService stand-in: one hardware profile with its plan."""

    def __init__(self, name: str = "rtx3060") -> None:
        self.name = name

    def report(self) -> HardwareReport:
        info = profile(self.name)
        return HardwareReport(info=info, warnings=(), plan=plan_for(info))


class FakeModels:
    """ModelsService stand-in: two required models (one missing) and a download that reports bytes."""

    def __init__(self) -> None:
        self.rows_ = [
            SimpleNamespace(id="det", name="Detector", required=True, status="missing", size_mb=2),
            SimpleNamespace(id="ocr", name="OCR", required=True, status="installed", size_mb=5),
            SimpleNamespace(id="extra", name="Extra", required=False, status="missing", size_mb=9),
        ]

    def rows(self) -> tuple[list[Any], object]:
        return self.rows_, object()

    def download_required(
        self, on_progress: Callable[[str, int, int | None], None]
    ) -> list[tuple[str, str | None]]:
        for done in (500_000, 2_000_000):
            on_progress("det", done, 2_000_000)
        self.rows_[0].status = "installed"
        return [("det", None)]


def _settle(qapp: QApplication, done: Callable[[], bool], seconds: float = 30.0) -> None:
    """Process events until `done()` (a slow CI runner can take seconds to start the worker threads)."""
    deadline = time.monotonic() + seconds
    while not done() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert done()


@pytest.fixture
def user_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config.toml"
    monkeypatch.setattr("omniscan.core.config.USER_TOML", path)
    monkeypatch.setattr("omniscan.gui.services.settings.USER_TOML", path)
    return path


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    home = tmp_path / "home" / "omniscan"
    return Config(
        paths=PathsConfig(
            library_root=home / "library",
            work_root=home / "work",
            output_root=home / "output",
            models_dir=tmp_path / "repo" / "models",
        )
    )


def _wizard(cfg: Config, **kwargs: Any) -> WelcomeDialog:
    kwargs.setdefault("ollama_check", lambda: "answers (fake)")
    hardware = kwargs.pop("hardware", FakeHardware())
    return WelcomeDialog(hardware, cfg=cfg, **kwargs)


def test_shows_the_plan_and_applies_it(qapp: QApplication, cfg: Config, user_toml: Path) -> None:
    dialog = _wizard(cfg, ollama_check=lambda: "Ollama answers (fake)")
    applied: list[int] = []
    dialog.config_changed.connect(lambda: applied.append(1))
    assert dialog.apply_plan() == 0  # nothing detected yet
    _settle(qapp, lambda: dialog.optimise_button.isEnabled() and "fake" in dialog.ollama_label.text())
    assert "RTX 3060" in dialog.hardware_label.text() and "tier: gaming" in dialog.plan_label.text()
    assert "PyTorch runs on NVIDIA GeForce RTX 3060 (cuda)" in dialog.runtime_label.text()
    assert not dialog.runtime_button.isVisibleTo(dialog)
    assert dialog.apply_plan() == 7 and applied == [1]
    assert 'device = "cuda:0"' in user_toml.read_text(encoding="utf-8")
    assert not dialog.optimise_button.isEnabled() and dialog.status_label.text().startswith("wrote 7")


def test_a_failed_check_is_shown_not_raised(qapp: QApplication, cfg: Config) -> None:
    def boom() -> str:
        raise RuntimeError("no network")

    dialog = _wizard(cfg, ollama_check=boom)
    _settle(qapp, lambda: "check failed" in dialog.ollama_label.text())
    assert "RuntimeError: no network" in dialog.ollama_label.text()


def test_the_steps_go_forward_and_back(qapp: QApplication, cfg: Config) -> None:
    dialog = _wizard(cfg)
    assert dialog.step_label.text() == "Step 1 of 5: This PC" and not dialog.back_button.isEnabled()
    for _ in STEPS[1:]:
        dialog.next_button.click()
    assert dialog.step_label.text() == "Step 5 of 5: Sharing and you"
    assert dialog.finish_button.isVisibleTo(dialog) and not dialog.next_button.isVisibleTo(dialog)
    dialog.back_button.click()
    assert dialog.stack.currentIndex() == 3 and dialog.next_button.isVisibleTo(dialog)


def test_the_packaged_app_downloads_the_runtime_a_card_needs(
    qapp: QApplication, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The packaged app's CPU PyTorch sees no card: the OS found one, so the wizard offers its runtime, and holds
    the plan back (it would pin the CPU)."""
    monkeypatch.setenv("OMNISCAN_RUNTIME_DIR", str(tmp_path / "runtime"))
    calls: list[str] = []

    def install(backend: str, on_line: Callable[[str], None]) -> Path:
        calls.append(backend)
        on_line("Downloading torch")
        return tmp_path / "runtime" / f"{backend}-torch2.13.0"

    dialog = _wizard(
        cfg,
        hardware=FakeHardware("cpu-laptop"),
        recommend=lambda _hw: Recommendation(backend="cuda", gpu="NVIDIA GeForce RTX 4070", note=None),
        runtime_install=install,
        packaged=True,
    )
    _settle(qapp, lambda: dialog.runtime_button.isVisibleTo(dialog))
    assert "needs the cuda build of PyTorch" in dialog.runtime_label.text()
    assert dialog.plan() is None and not dialog.optimise_button.isEnabled()
    assert "once PyTorch runs on it" in dialog.plan_label.text()
    dialog.runtime_button.click()
    _settle(qapp, lambda: "Restart OmniScan" in dialog.runtime_label.text())
    assert calls == ["cuda"] and "cuda-torch2.13.0 is installed" in dialog.runtime_label.text()
    assert not dialog.runtime_button.isEnabled()


def test_a_failed_runtime_download_can_be_tried_again(
    qapp: QApplication, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMNISCAN_RUNTIME_DIR", str(tmp_path / "runtime"))

    def install(_backend: str, _on_line: Callable[[str], None]) -> Path:
        raise RuntimeError("uv exit code 2")

    dialog = _wizard(
        cfg,
        hardware=FakeHardware("cpu-laptop"),
        recommend=lambda _hw: Recommendation(backend="xpu", gpu="Intel Arc B580", note=None),
        runtime_install=install,
        packaged=True,
    )
    _settle(qapp, lambda: dialog.runtime_button.isVisibleTo(dialog))
    dialog.runtime_button.click()
    _settle(qapp, lambda: "failed" in dialog.runtime_label.text())
    assert "uv exit code 2" in dialog.runtime_label.text() and dialog.runtime_button.isEnabled()


def test_a_source_checkout_names_the_extra_instead(qapp: QApplication, cfg: Config) -> None:
    dialog = _wizard(
        cfg,
        hardware=FakeHardware("cpu-laptop"),
        recommend=lambda _hw: Recommendation(backend="rocm-gfx1201", gpu="AMD Radeon RX 9070 XT", note=None),
        packaged=False,
    )
    _settle(qapp, lambda: "uv sync" in dialog.runtime_label.text())
    assert "`uv sync --extra rocm-gfx1201`" in dialog.runtime_label.text()
    assert not dialog.runtime_button.isVisibleTo(dialog)


def test_one_data_folder_holds_the_others(
    qapp: QApplication, cfg: Config, user_toml: Path, tmp_path: Path
) -> None:
    dialog = _wizard(cfg, packaged=False)
    assert dialog.folder_edit.text() == str(tmp_path / "home" / "omniscan")  # the folders' common parent
    assert not dialog.models_check.isChecked()  # a source checkout keeps its models next to the code
    changed: list[int] = []
    dialog.config_changed.connect(lambda: changed.append(1))
    target = tmp_path / "data"
    dialog.folder_edit.setText(str(target))
    assert dialog.use_folder() == ["library_root", "work_root", "output_root"] and changed == [1]
    paths = tomllib.loads(user_toml.read_text(encoding="utf-8"))["paths"]
    assert paths == {
        "library_root": str(target / "library"),
        "work_root": str(target / "work"),
        "output_root": str(target / "output"),
    }
    dialog.models_check.setChecked(True)
    assert "models_dir" in dialog.use_folder()
    assert dialog.status_label.text() == f"library, work, output, models now in {target}"
    dialog.folder_edit.setText("  ")
    assert dialog.use_folder() == [] and dialog.status_label.text() == "choose a folder first"


def test_the_required_models_download_with_the_time_left(qapp: QApplication, cfg: Config) -> None:
    dialog = _wizard(cfg, models=FakeModels())
    dialog.go_to(STEPS.index("Models"))
    _settle(qapp, lambda: dialog.download_button.isEnabled())
    assert dialog.models_label.text().startswith(
        "Models: 1 required model(s) to download, about 2 MB: Detector."
    )
    dialog.download_button.click()
    _settle(qapp, lambda: dialog.download_label.text() == "the required models are installed")
    assert not dialog.download_progress.isVisibleTo(dialog) and not dialog.download_button.isEnabled()
    dialog.go_to(STEPS.index("Models"))  # shown again: looked up again
    _settle(qapp, lambda: "all 2 required models are installed" in dialog.models_label.text())


def test_the_progress_line_names_the_model_and_the_time_left(qapp: QApplication, cfg: Config) -> None:
    dialog = _wizard(cfg)
    dialog._on_download({"id": "det"}, 1_000_000, 4_000_000, time.monotonic() - 10)
    assert dialog.download_label.text() == "det: 1 of 4 MB, about 30 s left"
    assert dialog.download_progress.value() == 250


def test_translation_offers_another_model(qapp: QApplication, cfg: Config) -> None:
    dialog = _wizard(cfg)
    asked: list[int] = []
    dialog.profiles_requested.connect(lambda: asked.append(1))
    dialog.profiles_button.click()
    assert asked == [1] and "OpenAI or Anthropic" in dialog.translation_label.text()


def test_finish_writes_the_sharing_choice_and_the_name(
    qapp: QApplication, cfg: Config, user_toml: Path
) -> None:
    dialog = _wizard(cfg)
    assert dialog.share_check.isChecked() and dialog.name_edit.text() == ""
    assert "CC BY 4.0" in dialog.sharing_label.text() and "nothing is uploaded" in dialog.sharing_label.text()
    changed: list[int] = []
    dialog.config_changed.connect(lambda: changed.append(1))
    dialog.share_check.setChecked(False)
    dialog.name_edit.setText("  Ana ")
    dialog.finish()
    written = tomllib.loads(user_toml.read_text(encoding="utf-8"))
    assert written["share"] == {"enabled": False} and written["user"] == {"name": "Ana"} and changed == [1]

    named = cfg.model_copy(update={"user": UserConfig(name="Ana"), "share": ShareConfig(enabled=False)})
    again = _wizard(named)
    again.name_edit.setText("")
    again.finish()  # an emptied name is dropped, so nothing is recorded
    assert "user" not in tomllib.loads(user_toml.read_text(encoding="utf-8"))

    untouched = _wizard(named)
    untouched.config_changed.connect(lambda: changed.append(2))
    untouched.finish()
    assert changed == [1]  # nothing changed: nothing written

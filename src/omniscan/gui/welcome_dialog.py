"""WelcomeDialog: the first-run wizard — five steps from a fresh install to a PC ready to translate (#44).

Shown by the main window the first time the app runs on a machine (QSettings `welcome/shown`), and from Settings →
Hardware later. The steps:
1. This PC: the hardware, the tuning plan (`omniscan tune`) to apply, and — in the packaged app, which ships the CPU
   build of PyTorch — the GPU runtime for the card it found (`omniscan runtime install`);
2. Data folder: one folder for the library, work and output folders (and the models);
3. Models: the required models, downloaded with progress and the time left;
4. Translation: whether Ollama answers, and the way to another model or an API key (Settings → Translation);
5. Sharing and you: the sharing opt-out stated plainly, and the name a group sees on your work.
Nothing is written without a click (the last step writes on Finish); closing it at any step just hides it until
asked for again. Detection, downloads and the Ollama check run on worker threads.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from omniscan import runtime
from omniscan.core.config import Config, get_config
from omniscan.gui.services import settings, setup
from omniscan.gui.services.hardware import HardwareReport, HardwareService
from omniscan.gui.workers import run_task
from omniscan.hw.detect import HardwareInfo
from omniscan.hw.tune import Plan, TorchExtra, describe, main_gpu
from omniscan.pipeline.eta import format_duration

OllamaCheck = Callable[[], str]  # a one-line status of the Ollama daemon
RuntimeInstall = Callable[
    [TorchExtra, Callable[[str], None]], Path
]  # backend, output line callback -> folder
Recommend = Callable[[HardwareInfo], runtime.Recommendation]

STEPS = ("This PC", "Data folder", "Models", "Translation", "Sharing and you")
_MB = 1_000_000


def _ollama_status() -> str:
    """Whether the configured local Ollama daemon answers (one quick request)."""
    from omniscan.core.config import get_secrets
    from omniscan.llm.ollama import OllamaClient

    cfg = get_config()
    try:
        with OllamaClient(cfg.ollama, get_secrets()) as client:
            models = client.ps()
        return f"Ollama answers at {cfg.ollama.local_url} ({len(models)} model(s) loaded)"
    except Exception as exc:  # any failure is the same one line to the user
        return f"Ollama is not reachable at {cfg.ollama.local_url}: install it from ollama.com ({type(exc).__name__})"


def _install_runtime(backend: TorchExtra, on_line: Callable[[str], None]) -> Path:
    """Download `backend`'s GPU runtime with the bundled uv, uv's output lines to `on_line`."""
    return runtime.install(backend, run=setup.quiet_runner(on_line))


def _label(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    return label


def _row(*widgets: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    for widget in widgets:
        row.addWidget(widget)
    row.addStretch(1)
    return row


class WelcomeDialog(QDialog):
    """The first-run wizard: this PC, data folder, models, translation, sharing and you."""

    config_changed = Signal()  # the wizard wrote to the config: pages should reload it
    models_requested = Signal()  # the user wants the Models page
    profiles_requested = Signal()  # the user wants Settings → Translation (another model, an API key)

    def __init__(
        self,
        hardware: HardwareService,
        *,
        cfg: Config | None = None,
        models: Any | None = None,
        ollama_check: OllamaCheck | None = None,
        recommend: Recommend | None = None,
        runtime_install: RuntimeInstall | None = None,
        packaged: bool | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build the wizard; the services, checks, runtime download and packaged flag are injectable (tests)."""
        super().__init__(parent)
        self.setWindowTitle("Welcome to OmniScan")
        self.setMinimumWidth(600)
        self._hardware = hardware
        self._cfg = cfg or get_config()
        self._models = models
        self._ollama_check: OllamaCheck = ollama_check or _ollama_status
        self._recommend: Recommend = recommend or runtime.recommend
        self._runtime_install: RuntimeInstall = runtime_install or _install_runtime
        self._packaged = bool(getattr(sys, "frozen", False)) if packaged is None else packaged
        self._plan: Plan | None = None
        self._backend: TorchExtra | None = None
        self._required: list[Any] = []  # the required ModelRows still to download
        self._signals: list[Any] = []  # keeps the running tasks' signals alive

        self.step_label = QLabel("", self)
        self.stack = QStackedWidget(self)
        for build in (
            self._build_pc,
            self._build_folder,
            self._build_models,
            self._build_translation,
            self._build_sharing,
        ):
            self.stack.addWidget(build())
        self.status_label = _label("", self)
        self.close_button = QPushButton("Close", self)
        self.back_button = QPushButton("Back", self)
        self.next_button = QPushButton("Next", self)
        self.finish_button = QPushButton("Finish", self)

        root = QVBoxLayout(self)
        root.addWidget(self.step_label)
        root.addWidget(self.stack, 1)
        root.addWidget(self.status_label)
        buttons = QHBoxLayout()
        buttons.addWidget(self.close_button)
        buttons.addStretch(1)
        for button in (self.back_button, self.next_button, self.finish_button):
            buttons.addWidget(button)
        root.addLayout(buttons)

        self.close_button.clicked.connect(self.reject)
        self.back_button.clicked.connect(lambda: self.go_to(self.stack.currentIndex() - 1))
        self.next_button.clicked.connect(lambda: self.go_to(self.stack.currentIndex() + 1))
        self.finish_button.clicked.connect(self.finish)
        self.go_to(0)
        self._detect()

    # ------------------------------------------------------------------ pages

    def _build_pc(self) -> QWidget:
        page = QWidget(self)
        self.intro_label = _label(
            "OmniScan turns raw manhwa and manga chapters into English releases. Five short steps get this PC "
            "ready; you can close this at any step and come back from Settings → Hardware.",
            page,
        )
        self.hardware_label = _label("Hardware: detecting...", page)
        self.plan_label = _label("", page)
        self.optimise_button = QPushButton("Optimise for this PC", page)
        self.optimise_button.setEnabled(False)
        self.runtime_label = _label("", page)
        self.runtime_button = QPushButton("Download the GPU runtime", page)
        self.runtime_button.setVisible(False)
        self.runtime_progress = QProgressBar(page)
        self.runtime_progress.setRange(0, 0)  # uv reports no byte counts: busy until it ends
        self.runtime_progress.setVisible(False)
        layout = QVBoxLayout(page)
        for widget in (self.intro_label, self.hardware_label, self.plan_label):
            layout.addWidget(widget)
        layout.addLayout(_row(self.optimise_button))
        layout.addWidget(self.runtime_label)
        layout.addLayout(_row(self.runtime_button))
        layout.addWidget(self.runtime_progress)
        layout.addStretch(1)
        self.optimise_button.clicked.connect(self.apply_plan)
        self.runtime_button.clicked.connect(self.install_runtime)
        return page

    def _build_folder(self) -> QWidget:
        page = QWidget(self)
        paths = self._cfg.paths
        self.folder_label = _label(
            "One folder holds your series (library), OmniScan's working files (work) and the finished releases "
            f"(output). Today: {paths.library_root}, {paths.work_root}, {paths.output_root}.",
            page,
        )
        self.folder_edit = QLineEdit(str(setup.data_folder(self._cfg)), page)
        self.browse_button = QPushButton("Browse…", page)
        self.models_check = QCheckBox(f"Keep the AI models there too (today: {paths.models_dir})", page)
        self.models_check.setChecked(self._packaged)  # a source checkout keeps its models next to the code
        self.folder_button = QPushButton("Use this folder", page)
        layout = QVBoxLayout(page)
        layout.addWidget(self.folder_label)
        row = QHBoxLayout()
        row.addWidget(self.folder_edit, 1)
        row.addWidget(self.browse_button)
        layout.addLayout(row)
        layout.addWidget(self.models_check)
        layout.addLayout(_row(self.folder_button))
        layout.addStretch(1)
        self.browse_button.clicked.connect(self._browse)
        self.folder_button.clicked.connect(self.use_folder)
        return page

    def _build_models(self) -> QWidget:
        page = QWidget(self)
        self.models_label = _label("Models: checking which required models this PC has...", page)
        self.download_button = QPushButton("Download required models", page)
        self.download_button.setEnabled(False)
        self.models_button = QPushButton("Open the Models page", page)
        self.download_progress = QProgressBar(page)
        self.download_progress.setVisible(False)
        self.download_label = _label("", page)
        layout = QVBoxLayout(page)
        layout.addWidget(self.models_label)
        layout.addLayout(_row(self.download_button, self.models_button))
        layout.addWidget(self.download_progress)
        layout.addWidget(self.download_label)
        layout.addStretch(1)
        self.download_button.clicked.connect(self.download_models)
        self.models_button.clicked.connect(self.models_requested)
        return page

    def _build_translation(self) -> QWidget:
        page = QWidget(self)
        self.translation_label = _label(
            "OmniScan translates with gemma4 on Ollama Cloud, through the Ollama app on this PC (sign in to Ollama "
            "for its cloud models), and falls back to translategemma on this PC when the cloud is busy. Install "
            "Ollama from ollama.com. Another Ollama model, or your own OpenAI or Anthropic API key, is set up in "
            "Settings → Translation.",
            page,
        )
        self.ollama_label = _label("Ollama: checking...", page)
        self.recheck_button = QPushButton("Check again", page)
        self.profiles_button = QPushButton("Use another model or an API key…", page)
        layout = QVBoxLayout(page)
        layout.addWidget(self.translation_label)
        layout.addWidget(self.ollama_label)
        layout.addLayout(_row(self.recheck_button, self.profiles_button))
        layout.addStretch(1)
        self.recheck_button.clicked.connect(self.check_ollama)
        self.profiles_button.clicked.connect(self.profiles_requested)
        return page

    def _build_sharing(self) -> QWidget:
        page = QWidget(self)
        self.sharing_label = _label(
            "Your corrections can make OmniScan better for everyone. When sharing is on, you can export the fixes "
            "you make by hand (with the pages they are on) as a file and send it in yourself: nothing is uploaded, "
            "no account is needed, and you contribute them under CC BY 4.0. Pages are never published. Turn it "
            "off here for this PC, or per series in its settings.",
            page,
        )
        self.share_check = QCheckBox("Share my corrections (on by default)", page)
        self.share_check.setChecked(self._cfg.share.enabled)
        self.name_label = _label(
            "Your name, if you work in a group (optional): it signs your edits, chapter steps and notes so the "
            "others see who did what. It never goes into a contribution.",
            page,
        )
        self.name_edit = QLineEdit(self._cfg.user.name, page)
        self.name_edit.setMaxLength(80)
        self.name_edit.setPlaceholderText("e.g. Ana")
        layout = QVBoxLayout(page)
        for widget in (self.sharing_label, self.share_check, self.name_label, self.name_edit):
            layout.addWidget(widget)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------ navigation

    def go_to(self, index: int) -> None:
        """Show step `index` (clamped); the Models step looks up what is missing each time it shows."""
        index = max(0, min(index, self.stack.count() - 1))
        self.stack.setCurrentIndex(index)
        self.step_label.setText(f"Step {index + 1} of {len(STEPS)}: {STEPS[index]}")
        last = index == self.stack.count() - 1
        self.back_button.setEnabled(index > 0)
        self.next_button.setVisible(not last)
        self.finish_button.setVisible(last)
        if STEPS[index] == "Models" and self._models is not None and self.download_progress.isHidden():
            self._look_up_models()

    def finish(self) -> None:
        """Write the sharing choice and the name when they changed, then close."""
        name = self.name_edit.text().strip()
        changes: list[tuple[str, str, Any]] = []
        if self.share_check.isChecked() != self._cfg.share.enabled:
            changes.append(("share", "enabled", self.share_check.isChecked()))
        if name != self._cfg.user.name:
            changes.append(("user", "name", name))
        try:
            for section, key, value in changes:
                if section == "user" and not value:
                    settings.clear_global(section, key)
                else:
                    settings.set_global(section, key, value)
        except settings.SettingError as error:
            self.status_label.setText(f"refused: {error}")
            return
        if changes:
            self.config_changed.emit()
        self.accept()

    # ------------------------------------------------------------------ step 1: this PC

    def plan(self) -> Plan | None:
        """The tuning plan once the hardware was detected (None while the card waits for its PyTorch build)."""
        return self._plan

    def apply_plan(self) -> int:
        """Write the plan's settings to the user config; how many (0 when there is no plan yet)."""
        plan = self._plan
        if plan is None:
            return 0
        try:
            for (section, key), value in plan.overrides.items():
                settings.set_global(section, key, value)
        except settings.SettingError as error:
            self.status_label.setText(f"refused: {error}")
            return 0
        self.status_label.setText(f"wrote {len(plan.overrides)} setting(s) to the config")
        self.optimise_button.setEnabled(False)
        self.config_changed.emit()
        return len(plan.overrides)

    def install_runtime(self) -> None:
        """Download the GPU runtime the hardware asks for, uv's latest line as the status."""
        backend = self._backend
        if backend is None or backend == "cpu":
            return
        latest: dict[str, str] = {}  # written on the worker thread, read on the GUI thread

        def task(progress: Callable[[int, int | None], None]) -> Path:
            def on_line(line: str) -> None:
                latest["line"] = line  # set before the emit the slot reads
                progress(0, None)

            return self._runtime_install(backend, on_line)

        self.runtime_button.setEnabled(False)
        self.runtime_progress.setVisible(True)
        self.runtime_label.setText(f"GPU runtime: downloading the {backend} build of PyTorch…")
        signals = run_task(task)
        self._signals.append(signals)
        signals.progress.connect(lambda _done, _total: self.status_label.setText(latest.get("line", "")))
        signals.finished.connect(self._on_runtime_done)
        signals.failed.connect(self._on_runtime_failed)

    def _on_runtime_done(self, folder: object) -> None:
        self.runtime_progress.setVisible(False)
        name = folder.name if isinstance(folder, Path) else str(folder)
        self.runtime_label.setText(f"GPU runtime: {name} is installed. Restart OmniScan to use it.")
        self.status_label.setText("")

    def _on_runtime_failed(self, error: str) -> None:
        self.runtime_progress.setVisible(False)
        self.runtime_button.setEnabled(True)
        self.runtime_label.setText(
            f"GPU runtime: the download failed ({error}). OmniScan keeps running on the CPU."
        )

    def _detect(self) -> None:
        """Hardware (with the runtime advice) and Ollama on worker threads (detection imports torch)."""

        def task(
            _progress: Callable[[int, int | None], None],
        ) -> tuple[HardwareReport, runtime.Recommendation]:
            report = self._hardware.report()
            return report, self._recommend(report.info)

        signals = run_task(task)
        self._signals.append(signals)
        signals.finished.connect(self._on_hardware)
        signals.failed.connect(
            lambda text: self.hardware_label.setText(f"Hardware: detection failed ({text})")
        )
        self.check_ollama()

    def _on_hardware(self, result: object) -> None:
        """Show the snapshot, the plan (once PyTorch can use the card) and the GPU runtime advice."""
        if not isinstance(result, tuple):
            return
        report, advice = result
        if not isinstance(report, HardwareReport) or not isinstance(advice, runtime.Recommendation):
            return
        info = report.info
        gpu = f"{info.gpus[0].name} ({info.gpus[0].vram_gb:g} GB)" if info.gpus else "no GPU"
        self.hardware_label.setText(f"Hardware: {info.cpu_name}, {info.ram_gb:g} GB RAM, {gpu}.")
        sees_gpu = main_gpu(info) is not None
        # a card that needs another PyTorch build first: a plan now would pin the CPU and outlive the download
        waiting = advice.backend != "cpu" and not sees_gpu
        self._plan = None if waiting else report.plan
        if waiting:
            self.plan_label.setText(f"The plan for {advice.gpu} shows here once PyTorch runs on it.")
        elif report.plan is not None:
            self.plan_label.setText("\n".join(describe(report.plan)))
            self.optimise_button.setEnabled(True)
        self._backend = advice.backend
        self.runtime_label.setText(f"GPU runtime: {self._runtime_text(advice, sees_gpu)}")

    def _runtime_text(self, advice: runtime.Recommendation, sees_gpu: bool) -> str:
        """What this PC's PyTorch build means for its card; shows the download button when one is needed."""
        if advice.backend == "cpu":
            return (
                advice.note
                or "no graphics card OmniScan can use was found: it runs on the processor (slower)."
            )
        if sees_gpu:
            return f"PyTorch runs on {advice.gpu} ({advice.backend})."
        if not self._packaged:
            return f"{advice.gpu} needs the {advice.backend} build of PyTorch: `uv sync --extra {advice.backend}`."
        current = runtime.active()
        if current is not None:
            return f"{current.name} is installed. Restart OmniScan to use {advice.gpu}."
        self.runtime_button.setVisible(True)
        return (
            f"{advice.gpu} needs the {advice.backend} build of PyTorch, a download of up to a few GB. Until then "
            "OmniScan runs on the processor."
        )

    # ------------------------------------------------------------------ step 2: data folder

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Data folder", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)

    def use_folder(self) -> list[str]:
        """Point the folders into the chosen data folder; the keys written (none when refused or empty)."""
        text = self.folder_edit.text().strip()
        if not text:
            self.status_label.setText("choose a folder first")
            return []
        folder = Path(text).expanduser()
        try:
            keys = setup.use_data_folder(folder, models=self.models_check.isChecked())
        except settings.SettingError as error:
            self.status_label.setText(f"refused: {error}")
            return []
        names = ", ".join(setup.SUBFOLDERS[key] for key in keys)
        self.status_label.setText(f"{names} now in {folder}")
        self.config_changed.emit()
        return keys

    # ------------------------------------------------------------------ step 3: models

    def _look_up_models(self) -> None:
        models = self._models
        if models is None:
            return
        signals = run_task(lambda _progress: models.rows()[0])
        self._signals.append(signals)
        signals.finished.connect(self._on_model_rows)
        signals.failed.connect(lambda text: self.models_label.setText(f"Models: the check failed ({text})"))

    def _on_model_rows(self, rows: object) -> None:
        required = [row for row in rows if row.required] if isinstance(rows, list) else []
        self._required = [row for row in required if row.status in ("missing", "corrupt")]
        self.download_button.setEnabled(bool(self._required))
        if not self._required:
            self.models_label.setText(f"Models: all {len(required)} required models are installed.")
            return
        size = sum(row.size_mb for row in self._required)
        names = ", ".join(row.name for row in self._required)
        self.models_label.setText(
            f"Models: {len(self._required)} required model(s) to download, about {size:,} MB: {names}. They also "
            "download on first use."
        )

    def download_models(self) -> None:
        """Download every missing required model, with the overall progress and the time left."""
        models = self._models
        if models is None or not self._required:
            return
        sizes = {row.id: row.size_mb * _MB for row in self._required}
        total = sum(sizes.values())
        finished: dict[str, int] = {}  # model id -> its bytes, once a later model started
        current: dict[str, str] = {}  # written on the worker thread, read on the GUI thread

        def task(progress: Callable[[int, int | None], None]) -> Any:
            def on_required(model_id: str, done: int, _total: int | None) -> None:
                previous = current.get("id")
                if previous is not None and previous != model_id:
                    finished[previous] = sizes.get(previous, 0)
                current["id"] = model_id
                progress(min(total, sum(finished.values()) + done), total)

            return models.download_required(on_required)

        started = time.monotonic()
        self.download_button.setEnabled(False)
        self.download_progress.setVisible(True)
        self.download_progress.setRange(0, 0)
        signals = run_task(task)
        self._signals.append(signals)
        signals.progress.connect(lambda done, _total: self._on_download(current, done, total, started))
        signals.finished.connect(self._on_downloaded)
        signals.failed.connect(self._on_download_failed)

    def _on_download(self, current: dict[str, str], done: int, total: int, started: float) -> None:
        self.download_progress.setRange(0, 1000)
        self.download_progress.setValue(round(1000 * done / total) if total else 0)
        left = setup.time_left(done, total, time.monotonic() - started)
        eta = f", about {format_duration(left)} left" if left is not None else ""
        self.download_label.setText(f"{current.get('id', '')}: {done // _MB:,} of {total // _MB:,} MB{eta}")

    def _on_downloaded(self, results: object) -> None:
        self.download_progress.setVisible(False)
        pairs = results if isinstance(results, list) else []
        failures = [f"{model_id} ({error})" for model_id, error in pairs if error]
        if failures:
            self.download_label.setText("not downloaded: " + "; ".join(failures))
            self.download_button.setEnabled(True)
        else:
            self.download_label.setText("the required models are installed")
            self._required = []

    def _on_download_failed(self, error: str) -> None:
        self.download_progress.setVisible(False)
        self.download_button.setEnabled(True)
        self.download_label.setText(f"the download failed ({error})")

    # ------------------------------------------------------------------ step 4: translation

    def check_ollama(self) -> None:
        """Ask the Ollama daemon whether it answers (worker thread)."""
        self.ollama_label.setText("Ollama: checking...")
        signals = run_task(lambda _progress: self._ollama_check())
        self._signals.append(signals)
        signals.finished.connect(lambda text: self.ollama_label.setText(f"Ollama: {text}"))
        signals.failed.connect(lambda text: self.ollama_label.setText(f"Ollama: check failed ({text})"))

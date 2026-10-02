"""WelcomeDialog: the first start — what this PC can do, and the three things to set up.

Shown by the main window the first time the app runs on a machine (QSettings `welcome/shown`), and from
Settings later. It detects the hardware on a worker thread, shows the tuning plan (`omniscan tune`) and
offers to apply it, to open the Models page for the required downloads, and says whether Ollama answers.
Nothing happens without a click; closing it just hides it until asked for again.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from omniscan.gui.services import settings
from omniscan.gui.services.hardware import HardwareReport, HardwareService
from omniscan.gui.workers import run_task
from omniscan.hw.tune import Plan, describe

OllamaCheck = Callable[[], str]  # a one-line status of the Ollama daemon


def _ollama_status() -> str:
    """Whether the configured local Ollama daemon answers (one quick request)."""
    from omniscan.core.config import get_config, get_secrets
    from omniscan.llm.ollama import OllamaClient

    cfg = get_config()
    try:
        with OllamaClient(cfg.ollama, get_secrets()) as client:
            models = client.ps()
        return f"Ollama answers at {cfg.ollama.local_url} ({len(models)} model(s) loaded)"
    except Exception as exc:  # any failure is the same one line to the user
        return f"Ollama is not reachable at {cfg.ollama.local_url}: install it from ollama.com ({type(exc).__name__})"


class WelcomeDialog(QDialog):
    """First-run checklist: hardware plan, models, Ollama."""

    plan_applied = Signal()  # the plan was written to the config: pages should reload it
    models_requested = Signal()  # the user wants the Models page

    def __init__(
        self,
        hardware: HardwareService,
        *,
        ollama_check: OllamaCheck | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build the dialog; `hardware` and `ollama_check` are injectable (tests pass fakes)."""
        super().__init__(parent)
        self.setWindowTitle("Welcome to OmniScan")
        self.setMinimumWidth(560)
        self._hardware = hardware
        self._ollama_check: OllamaCheck = ollama_check or _ollama_status
        self._plan: Plan | None = None

        self.intro_label = QLabel(
            "OmniScan turns raw manhwa and manga chapters into English releases. Three things make it run well on "
            "this PC:",
            self,
        )
        self.intro_label.setWordWrap(True)
        self.hardware_label = QLabel("1. Hardware: detecting...", self)
        self.hardware_label.setWordWrap(True)
        self.plan_label = QLabel("", self)
        self.plan_label.setWordWrap(True)
        self.optimise_button = QPushButton("Optimise for this PC", self)
        self.optimise_button.setEnabled(False)
        self.models_label = QLabel(
            "2. Models: the detector, OCR and inpainting models download on first use, or all at once on the Models "
            "page (Download required models).",
            self,
        )
        self.models_label.setWordWrap(True)
        self.models_button = QPushButton("Open the Models page", self)
        self.ollama_label = QLabel("3. Translation: checking Ollama...", self)
        self.ollama_label.setWordWrap(True)
        self.close_button = QPushButton("Close", self)
        self.status_label = QLabel("", self)

        root = QVBoxLayout(self)
        root.addWidget(self.intro_label)
        root.addWidget(self.hardware_label)
        root.addWidget(self.plan_label)
        row1 = QHBoxLayout()
        row1.addWidget(self.optimise_button)
        row1.addStretch(1)
        root.addLayout(row1)
        root.addWidget(self.models_label)
        row2 = QHBoxLayout()
        row2.addWidget(self.models_button)
        row2.addStretch(1)
        root.addLayout(row2)
        root.addWidget(self.ollama_label)
        root.addWidget(self.status_label)
        row3 = QHBoxLayout()
        row3.addStretch(1)
        row3.addWidget(self.close_button)
        root.addLayout(row3)

        self.optimise_button.clicked.connect(self.apply_plan)
        self.models_button.clicked.connect(self.models_requested)
        self.close_button.clicked.connect(self.accept)
        self._detect()

    def plan(self) -> Plan | None:
        """The tuning plan once the hardware was detected."""
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
        self.plan_applied.emit()
        return len(plan.overrides)

    def _detect(self) -> None:
        """Hardware and Ollama on worker threads (detection imports torch)."""
        signals = run_task(lambda _progress: self._hardware.report())
        signals.finished.connect(self._on_hardware)
        signals.failed.connect(
            lambda text: self.hardware_label.setText(f"1. Hardware: detection failed ({text})")
        )
        ollama = run_task(lambda _progress: self._ollama_check())
        ollama.finished.connect(lambda text: self.ollama_label.setText(f"3. Translation: {text}"))
        ollama.failed.connect(
            lambda text: self.ollama_label.setText(f"3. Translation: check failed ({text})")
        )

    def _on_hardware(self, report: object) -> None:
        """Show the snapshot and the plan."""
        if not isinstance(report, HardwareReport):
            return
        info = report.info
        gpu = f"{info.gpus[0].name} ({info.gpus[0].vram_gb:g} GB)" if info.gpus else "no GPU"
        self.hardware_label.setText(f"1. Hardware: {info.cpu_name}, {info.ram_gb:g} GB RAM, {gpu}.")
        self._plan = report.plan
        if report.plan is not None:
            self.plan_label.setText("\n".join(describe(report.plan)))
            self.optimise_button.setEnabled(True)

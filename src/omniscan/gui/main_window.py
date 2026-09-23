"""MainWindow: the desktop shell — sidebar navigation, the five pages, status bar, remembered geometry.

Pages are built once and shown via a QStackedWidget; the library's double-click jumps to the reader
and a settings write reloads the config into every page. Window geometry and the last page persist
through QSettings.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from PySide6.QtCore import QByteArray, QSettings
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QStatusBar,
)

from omniscan.core.config import Config, load_config
from omniscan.gui.import_view import ImportView
from omniscan.gui.library_view import LibraryView
from omniscan.gui.models_view import ModelsView
from omniscan.gui.reader_view import ReaderView
from omniscan.gui.run_view import RunView
from omniscan.gui.services.hardware import HardwareService
from omniscan.gui.services.importer import ImporterService
from omniscan.gui.services.models import ModelsService
from omniscan.gui.settings_view import SettingsView

# "Import" is last so the other four keep their existing indices (scripts/gui_screenshots.py hard-codes them).
PAGES = ("Library", "Reader", "Run", "Models", "Settings", "Import")


class MainWindow(QMainWindow):
    """One window: sidebar over the five pages plus the status bar (device, running job)."""

    def __init__(
        self,
        cfg: Config | None = None,
        *,
        models_service: Any | None = None,
        hardware_service: HardwareService | None = None,
        importer_service: Any | None = None,
        qsettings: QSettings | None = None,
        config_loader: Callable[[], Config] | None = None,
    ) -> None:
        """Build the shell; the services, QSettings and config loader are injectable for tests."""
        super().__init__()
        self._config_loader = config_loader or load_config
        self._cfg = cfg or self._config_loader()
        self.setWindowTitle("OmniScan")
        self._qsettings = qsettings or QSettings("OmniScan", "gui")

        self.library_view = LibraryView(self._cfg)
        self.reader_view = ReaderView(self._cfg)
        self.run_view = RunView(self._cfg)
        self.models_view = ModelsView(models_service or ModelsService(self._cfg))
        self.settings_view = SettingsView(self._cfg, hardware=hardware_service)
        self.import_view = ImportView(importer_service or ImporterService(self._cfg))

        self.stack = QStackedWidget()
        for view in (
            self.library_view,
            self.reader_view,
            self.run_view,
            self.models_view,
            self.settings_view,
            self.import_view,
        ):
            self.stack.addWidget(view)
        self.sidebar = QListWidget()
        self.sidebar.addItems(PAGES)
        self.sidebar.setFixedWidth(140)

        central = QSplitter()
        central.addWidget(self.sidebar)
        central.addWidget(self.stack)
        central.setSizes([1, 6])
        self.setCentralWidget(central)

        self.device_label = QLabel()
        self.job_label = QLabel("job: idle")
        status = QStatusBar(self)
        status.addWidget(self.device_label)
        status.addPermanentWidget(self.job_label)
        self.setStatusBar(status)

        for index in range(len(PAGES)):
            QShortcut(QKeySequence(f"Ctrl+{index + 1}"), self, lambda i=index: self.show_page(i))

        self.sidebar.currentRowChanged.connect(self._on_sidebar)
        self.library_view.chapter_opened.connect(self._on_chapter_opened)
        self.run_view.busy_changed.connect(self._on_busy_changed)
        self.settings_view.settings_changed.connect(self._reload_config)

        self._show_device()
        self._restore()

    # ------------------------------------------------------------------ state

    def show_page(self, index: int) -> None:
        """Switch to page `index` (0..4); the sidebar selection drives the stack."""
        self.sidebar.setCurrentRow(index)

    # ------------------------------------------------------------------ slots

    def _on_sidebar(self, row: int) -> None:
        """Show the page matching the sidebar row."""
        if row >= 0:
            self.stack.setCurrentIndex(row)

    def _on_chapter_opened(self, series: str, chapter: str) -> None:
        """Open the double-clicked library chapter in the reader."""
        if self.reader_view.open_chapter(series, chapter):
            self.show_page(PAGES.index("Reader"))
        else:
            self.statusBar().showMessage(f"cannot open {series} — {chapter}", 5000)

    def _on_busy_changed(self, running: bool) -> None:
        """Reflect the run page's state in the status bar."""
        self.job_label.setText("job: running" if running else "job: idle")

    def _reload_config(self) -> None:
        """A settings write landed: reload the config and hand it to every page."""
        self._cfg = self._config_loader()
        self.library_view.reconfigure(self._cfg)
        self.reader_view.reconfigure(self._cfg)
        self.run_view.reconfigure(self._cfg)
        self.settings_view.reconfigure(self._cfg)
        self.import_view.reconfigure(ImporterService(self._cfg))
        self._show_device()

    # ------------------------------------------------------------------ internals

    def _show_device(self) -> None:
        """The status bar's device hint (resolved per run; "auto" resolves at run time)."""
        self.device_label.setText(f"device: {self._cfg.gpu.device}")

    def _restore(self) -> None:
        """Restore window geometry and the last page from QSettings."""
        geometry = self._qsettings.value("window/geometry")
        if isinstance(geometry, QByteArray) and geometry:
            self.restoreGeometry(geometry)
        last = cast(int, self._qsettings.value("window/last_page", 0, type=int))
        self.sidebar.setCurrentRow(min(max(last, 0), len(PAGES) - 1))

    def closeEvent(self, event: Any) -> None:
        """Remember the geometry and the open page before closing."""
        self._qsettings.setValue("window/geometry", self.saveGeometry())
        self._qsettings.setValue("window/last_page", self.sidebar.currentRow())
        super().closeEvent(event)

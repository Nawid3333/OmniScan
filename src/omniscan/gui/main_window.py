"""MainWindow: the desktop shell — sidebar navigation, the pages, status bar, remembered geometry.

Pages are built once and shown via a QStackedWidget; the library's double-click jumps to the reader
and a settings write reloads the config into every page. Window geometry and the last page persist
through QSettings.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from PySide6.QtCore import QByteArray, QSettings, QSize, Qt
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
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
from omniscan.gui.studio_view import StudioView
from omniscan.gui.theme import MODE_PAGES, Appearance, set_role

# new pages go last so the first five keep their indices (scripts/gui_screenshots.py hard-codes them).
PAGES = ("Library", "Reader", "Run", "Models", "Settings", "Import", "Studio")
# the platform's own icon set (Segoe Fluent on Windows, SF Symbols on macOS, the icon theme on Linux)
_ICONS = {
    "Library": QIcon.ThemeIcon.FolderOpen,
    "Reader": QIcon.ThemeIcon.DocumentPrintPreview,
    "Run": QIcon.ThemeIcon.MediaPlaybackStart,
    "Models": QIcon.ThemeIcon.Computer,
    "Settings": QIcon.ThemeIcon.DocumentProperties,
    "Import": QIcon.ThemeIcon.DocumentOpen,
    "Studio": QIcon.ThemeIcon.InsertText,
}
_SIDEBAR_WIDTH = 190
_CONTENT_MARGINS = (24, 16, 24, 12)


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
        self.reader_view = ReaderView(self._cfg, qsettings=self._qsettings)
        self.run_view = RunView(self._cfg)
        self.models_view = ModelsView(models_service or ModelsService(self._cfg))
        self.settings_view = SettingsView(self._cfg, hardware=hardware_service, qsettings=self._qsettings)
        self.import_view = ImportView(importer_service or ImporterService(self._cfg))
        self.studio_view = StudioView(self._cfg)

        self.stack = QStackedWidget()
        for view in (
            self.library_view,
            self.reader_view,
            self.run_view,
            self.models_view,
            self.settings_view,
            self.import_view,
            self.studio_view,
        ):
            self.stack.addWidget(view)
        self.sidebar = QListWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setIconSize(QSize(18, 18))
        for name in PAGES:
            self.sidebar.addItem(QListWidgetItem(QIcon.fromTheme(_ICONS[name]), name))
        brand = QLabel("OmniScan")
        brand.setObjectName("brand")
        self.title_label = QLabel()
        set_role(self.title_label, "title")

        nav = QWidget()
        self._nav = nav
        nav.setObjectName("nav")
        nav.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)  # draw the stylesheet's border
        nav.setFixedWidth(_SIDEBAR_WIDTH)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)
        nav_layout.addWidget(brand)
        nav_layout.addWidget(self.sidebar, 1)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        self._content_layout = content_layout
        content_layout.setContentsMargins(*_CONTENT_MARGINS)
        content_layout.addWidget(self.title_label)
        content_layout.addWidget(self.stack, 1)
        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(nav)
        central_layout.addWidget(content, 1)
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
        self.studio_view.busy_changed.connect(self._on_busy_changed)
        self.settings_view.settings_changed.connect(self._reload_config)
        self.settings_view.appearance_changed.connect(self.apply_mode)
        self.reader_view.reading_changed.connect(self._on_reading_changed)
        self._was_maximized = False

        self._show_device()
        self._restore()
        self.apply_mode(self.settings_view.appearance)  # after restore: a hidden last page falls back

    # ------------------------------------------------------------------ state

    def show_page(self, index: int) -> None:
        """Switch to page `index` (an index into PAGES); the sidebar selection drives the stack."""
        self.sidebar.setCurrentRow(index)

    def apply_mode(self, appearance: Appearance) -> None:
        """Show only the pages the quick / standard / pro mode includes (indices stay stable)."""
        visible = MODE_PAGES[appearance.mode]
        for index, name in enumerate(PAGES):
            self.sidebar.setRowHidden(index, name not in visible)
        if PAGES[max(self.sidebar.currentRow(), 0)] not in visible:
            self.show_page(PAGES.index("Library"))

    # ------------------------------------------------------------------ slots

    def _on_sidebar(self, row: int) -> None:
        """Show the page matching the sidebar row."""
        if row >= 0:
            if row != PAGES.index("Reader"):
                self.reader_view.set_reading(False)
            self.stack.setCurrentIndex(row)
            self.title_label.setText(PAGES[row])

    def _on_chapter_opened(self, series: str, chapter: str) -> None:
        """Open the double-clicked library chapter in the reader."""
        if self.reader_view.open_chapter(series, chapter):
            self.show_page(PAGES.index("Reader"))
        else:
            self.statusBar().showMessage(f"cannot open {series} — {chapter}", 5000)

    def _on_reading_changed(self, reading: bool) -> None:
        """Reading mode: only the pages, full screen; leaving it restores the window as it was."""
        for widget in (self._nav, self.title_label, self.statusBar()):
            widget.setVisible(not reading)
        self._content_layout.setContentsMargins(*((0, 0, 0, 0) if reading else _CONTENT_MARGINS))
        if reading:
            self._was_maximized = self.isMaximized()
            self.showFullScreen()
        elif self._was_maximized:
            self.showMaximized()
        else:
            self.showNormal()

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
        self.studio_view.reconfigure(self._cfg)
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

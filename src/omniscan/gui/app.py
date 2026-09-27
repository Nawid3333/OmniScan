"""Application bootstrap for `omniscan gui` (and `python -m omniscan.gui`).

Qt stays an optional extra: without PySide6 `main` prints one line and returns 2 (the CLI command
maps that to exit code 2), so the rest of OmniScan never needs the GUI stack.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Build the QApplication, show the main window and run the event loop."""
    try:
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
    except ImportError as error:
        print(f"the GUI needs PySide6: install it with `uv sync --extra gui` ({error})", file=sys.stderr)
        return 2
    from omniscan.core.config import load_config
    from omniscan.gui.main_window import MainWindow
    from omniscan.gui.theme import apply_theme, load_appearance

    app = QApplication(list(argv) if argv is not None else sys.argv)
    apply_theme(app, load_appearance(QSettings("OmniScan", "gui")))  # OLED black by default
    window = MainWindow(load_config())
    window.show()
    return app.exec()

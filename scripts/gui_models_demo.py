"""Show the models settings view; `--screenshot` renders it headless for inspection.

Usage: `uv run --frozen python scripts/gui_models_demo.py [--screenshot OUT.png]
[--size 1300x800] [--role ROLE]`. Reads the config from the usual sources (env overrides
work, e.g. `OMNISCAN_PATHS__MODELS_DIR`). The view talks to the real Ollama daemon if it is
running; when it is down the ollama models simply show as `unknown`. Exits 1 for a bad
`--size`/`--role`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the optional --screenshot/--size/--role flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path, default=None, help="render offscreen to this PNG and exit")
    parser.add_argument("--size", default="1300x800", help="window size WxH (default: 1300x800)")
    parser.add_argument("--role", default=None, help="preselect this role in the filter combo")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the demo; returns the process exit code."""
    args = _parse_args(argv)
    if args.screenshot is not None:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"  # must be set before importing Qt
        if (
            sys.platform == "win32"
        ):  # the offscreen platform has no font database of its own: labels would be boxes
            os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    try:
        width, height = (int(part) for part in args.size.lower().split("x", 1))
    except ValueError:
        print(f"bad --size {args.size!r} (want WxH, e.g. 1300x800)", file=sys.stderr)
        return 1

    from omniscan.core.config import get_config
    from omniscan.gui.services.models import ModelsService

    service = ModelsService(get_config())

    from PySide6.QtWidgets import QApplication

    from omniscan.gui.models_view import ModelsView

    app = QApplication.instance() or QApplication([])
    view = ModelsView(service)
    view.resize(width, height)
    view.setWindowTitle("Models")
    view.show()
    app.processEvents()

    if args.role is not None:
        index = view.role_combo.findText(args.role)
        if index < 0:
            print(f"unknown role {args.role!r} (the combo has: {', '.join(view.role_combo.itemTexts())})", file=sys.stderr)
            return 1
        view.role_combo.setCurrentIndex(index)
        app.processEvents()

    if args.screenshot is not None:
        app.processEvents()
        if not view.grab().save(str(args.screenshot)):
            print(f"could not write {args.screenshot}", file=sys.stderr)
            return 1
        return 0
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
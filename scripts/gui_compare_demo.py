"""Show one chapter in the reader/compare view; `--screenshot` renders it headless for inspection.

Usage: `uv run --frozen python scripts/gui_compare_demo.py SERIES CHAPTER [--screenshot OUT.png]
[--size 1400x900] [--y STRIP_Y]`. Reads the config from the usual sources (env overrides work,
e.g. `OMNISCAN_PATHS__LIBRARY_ROOT`). Exits 1 with a message on stderr for an unknown
series/chapter.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse SERIES CHAPTER and the optional --screenshot/--size/--y flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", help="series folder name, e.g. 'My Series'")
    parser.add_argument("chapter", help="chapter folder name, e.g. 'Chapter 1'")
    parser.add_argument("--screenshot", type=Path, default=None, help="render offscreen to this PNG and exit")
    parser.add_argument("--size", default="1400x900", help="window size WxH (default: 1400x900)")
    parser.add_argument("--y", type=float, default=0.0, help="scroll both views to this strip y (default: 0)")
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
        print(f"bad --size {args.size!r} (want WxH, e.g. 1400x900)", file=sys.stderr)
        return 1

    from omniscan.core.config import get_config
    from omniscan.gui.services import library

    cfg = get_config()
    if args.series not in library.list_series(cfg):
        print(f"unknown series {args.series!r}", file=sys.stderr)
        return 1
    if args.chapter not in library.list_chapter_names(cfg, args.series):
        print(f"unknown chapter {args.chapter!r} of series {args.series!r}", file=sys.stderr)
        return 1
    view = library.load_chapter_view(cfg, args.series, args.chapter)

    from PySide6.QtWidgets import QApplication

    from omniscan.gui.compare_view import CompareView

    app = QApplication.instance() or QApplication([])
    compare = CompareView()
    compare.set_chapter(view)
    compare.resize(width, height)
    compare.setWindowTitle(f"{view.series} — {view.chapter}")
    compare.show()
    app.processEvents()
    compare.left.set_strip_y(args.y)  # linked mode scrolls the right side along

    if args.screenshot is not None:
        app.processEvents()
        if not compare.grab().save(str(args.screenshot)):
            print(f"could not write {args.screenshot}", file=sys.stderr)
            return 1
        return 0
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

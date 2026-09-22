"""Show the import view; `--screenshot` renders it headless for inspection.

Usage: `uv run --frozen python scripts/gui_import_demo.py [--screenshot OUT.png]
[--size 1300x800] [--source PATH]`. Without `--source` a synthetic .cbz (three chapters,
JPEG/PNG mix) is generated in a temp dir, so the page can be explored end to end without
any real manga. Everything on screen is real: the plan preview, the editable grouping,
the conversion notice, and the commit (which writes into the configured library root).
Exits 1 for a bad `--size` or `--source`.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import zipfile
from io import BytesIO
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the optional --screenshot/--size/--source flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path, default=None, help="render offscreen to this PNG and exit")
    parser.add_argument("--size", default="1300x800", help="window size WxH (default: 1300x800)")
    parser.add_argument("--source", type=Path, default=None, help="a source folder or .zip/.cbz archive")
    return parser.parse_args(argv)


def _make_demo_archive(directory: Path) -> Path:
    """A synthetic .cbz: three chapters, one of them holding convertible PNGs."""
    from PIL import Image

    path = directory / "Demo Series.cbz"
    with zipfile.ZipFile(path, "w") as archive:
        for index in range(1, 4):  # Chapter 1: three JPEG pages (copied as-is)
            buffer = BytesIO()
            Image.new("RGB", (40, 56), (30 + 40 * index, 60, 90)).save(buffer, format="JPEG", quality=95)
            archive.writestr(f"Chapter 1/{index}.jpg", buffer.getvalue())
        for index in range(1, 3):  # Chapter 2: two PNG pages (converted at import)
            buffer = BytesIO()
            Image.new("RGB", (40, 56), (200, 40 * index, 10)).save(buffer, format="PNG")
            archive.writestr(f"Chapter 2/{index}.png", buffer.getvalue())
        buffer = BytesIO()
        Image.new("RGB", (40, 56), (90, 90, 90)).save(buffer, format="JPEG", quality=95)
        archive.writestr("Chapter 2/3.jpg", buffer.getvalue())
        buffer = BytesIO()
        Image.new("RGB", (40, 56), (10, 10, 10)).save(buffer, format="JPEG", quality=95)
        archive.writestr("Chapter 3/1.jpg", buffer.getvalue())
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the demo; returns the process exit code."""
    args = _parse_args(argv)
    if args.screenshot is not None:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"  # must be set before importing Qt
        if sys.platform == "win32":  # the offscreen platform needs a font dir or labels are boxes
            os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    try:
        width, height = (int(part) for part in args.size.lower().split("x", 1))
    except ValueError:
        print(f"bad --size {args.size!r} (want WxH, e.g. 1300x800)", file=sys.stderr)
        return 1

    from omniscan.core.config import get_config
    from omniscan.gui.services.importer import ImporterService

    if args.source is not None:
        source = args.source
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="omniscan-demo-")
        source = _make_demo_archive(Path(temp_dir.name))
        print(f"demo source: {source}")

    service = ImporterService(get_config())

    from PySide6.QtWidgets import QApplication

    from omniscan.gui.import_view import ImportView

    app = QApplication.instance() or QApplication([])
    view = ImportView(service)
    view.resize(width, height)
    view.setWindowTitle("Import")
    view.show()
    view.set_source(source)

    if args.screenshot is None:
        return app.exec()

    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 5.0  # the plan arrives on a worker thread; wait for it
    while view.tree.topLevelItemCount() == 0 and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    if view.tree.topLevelItemCount() == 0:
        print(f"planning failed: {view.status_label.text()}", file=sys.stderr)
        return 1
    view.conversions_button.setChecked(True)  # show the per-file conversion list in the shot
    app.processEvents()
    if not view.grab().save(str(args.screenshot)):
        print(f"could not write {args.screenshot}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

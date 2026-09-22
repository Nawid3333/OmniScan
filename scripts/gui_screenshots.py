"""Render one PNG per GUI page over the synthetic fixture library (offscreen).

Usage: `uv run --frozen python scripts/gui_screenshots.py --out data/screenshots/U3c`
[--size 1300x800] [--series NAME] [--chapter NAME]. Writes `01-library.png` …
`06-settings.png` into `--out` (created); the run page is also captured mid-run with a
step-mode preview waiting (04). Services that would touch Ollama/the GPU are faked, so the
script needs no daemon and never takes GPU time. Exits 1 for a bad `--size`.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse --out/--size/--series/--chapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/screenshots/U3c"), help="PNG output directory")
    parser.add_argument("--size", default="1300x800", help="window size WxH (default: 1300x800)")
    parser.add_argument(
        "--series", default=None, help="chapter series to open in the Reader (default: fixture)"
    )
    parser.add_argument("--chapter", default=None, help="chapter to open in the Reader (default: fixture)")
    return parser.parse_args(argv)


class FakeModels:
    """ModelsService stand-in: four static rows, no Ollama and no torch."""

    def rows(self, **_kw: object) -> tuple[list[Any], Any]:
        """The catalog rows shown on the Models page plus the hardware snapshot."""
        from tests.unit.test_gui_hardware_service import _hw, _row  # the shared catalog-row builder

        return (
            [
                _row("ppocr-det", "ok", "cuda:0", ()),
                _row("ppocr-rec", "ok", "cuda:0", ()),
                _row("lama-fp16", "ok", "cuda:0", ()),
                _row("gemma4-12b", "slow", "cuda:0", ("needs 24 GB",)),
            ],
            _hw(),
        )

    def download_required(
        self, on_progress: Callable[[str, int, int | None], None] | None = None
    ) -> list[tuple[str, str | None]]:
        return []

    def download(self, model_id: str, on_progress: Callable[[int, int | None], None] | None = None) -> str:
        return "already installed"

    def remove(self, model_id: str) -> bool:
        return False


class FakeHardware:
    """HardwareService stand-in: the same static snapshot with one warning."""

    def report(self) -> Any:
        from tests.unit.test_gui_hardware_service import _hw

        from omniscan.gui.services.hardware import HardwareReport, ModelWarning

        return HardwareReport(
            info=_hw(), warnings=(ModelWarning("gemma4-12b", "slow", "cuda:0", ("needs 24 GB",)),)
        )


def main(argv: list[str] | None = None) -> int:
    """Render the screenshots; returns the process exit code."""
    args = _parse_args(argv)
    try:
        width, height = (int(part) for part in args.size.lower().split("x", 1))
    except ValueError:
        print(f"bad --size {args.size!r} (want WxH, e.g. 1300x800)", file=sys.stderr)
        return 1

    os.environ["QT_QPA_PLATFORM"] = "offscreen"  # before any Qt import
    if sys.platform == "win32":
        # the offscreen platform has no font database of its own: labels would be boxes
        os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests.* is not an installed package

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from tests.fixtures.gui_library import CHAPTERS, SERIES, build_library

    from omniscan.gui.main_window import MainWindow
    from omniscan.gui.services.runs import StageUpdate, StepPreview

    series = args.series or SERIES
    chapter = args.chapter or CHAPTERS[0]

    app = QApplication.instance() or QApplication([])
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="omniscan-gui-shots-") as tmp:
        cfg = build_library(Path(tmp) / "lib")
        qsettings = QSettings(str(Path(tmp) / "gui.ini"), QSettings.Format.IniFormat)
        window = MainWindow(  # type: ignore[arg-type]
            cfg, models_service=FakeModels(), hardware_service=FakeHardware(), qsettings=qsettings
        )
        window.resize(width, height)
        window.show()

        def shot(name: str) -> None:
            app.processEvents()
            if not window.grab().save(str(out_dir / name)):
                print(f"could not write {out_dir / name}", file=sys.stderr)
                raise SystemExit(1)

        # 01 Library: series list + chapter/stage table (Episode 01 is selected by default)
        window.show_page(0)
        app.processEvents()
        shot("01-library.png")

        # 02 Reader: the compare view on one chapter
        window.show_page(1)
        window.reader_view.open_chapter(series, chapter)
        app.processEvents()
        shot("02-reader.png")

        # 03 Run: the form, idle
        window.show_page(2)
        app.processEvents()
        shot("03-run.png")

        # 04 Run mid-run, step preview waiting on Continue/Abort (synthetic progress, no pipeline)
        run = window.run_view
        run.mode_combo.setCurrentText("Step")  # the preview below belongs to a step run
        run._set_running(True)
        run._on_stage(
            StageUpdate(
                chapter=CHAPTERS[1],
                stage="ocr",
                status="done",
                seconds=1.2,
                error=None,
                chapter_number=2,
                chapter_total=len(CHAPTERS),
                stage_number=4,
                stage_total=10,
            )
        )
        run._on_preview(
            StepPreview(
                stage="translate",
                status="done",
                summary="18 text lines translated (gemma4-12b)",
                details=(
                    "source 18 / target 18",
                    "tokens in 2 941 · out 1 213",
                    "judge: 17 accepted, 1 revised",
                ),
                position=5,
                total=10,
                error=None,
            )
        )
        run.log.appendPlainText("== step run: press Continue to advance, Abort to stop the run")
        app.processEvents()
        shot("04-run-step.png")
        run._set_running(False)  # the page is synthetic; leave it as if nothing ran

        # 05 Models: catalog rows + hardware header (faked, instant)
        window.show_page(3)
        app.processEvents()
        shot("05-models.png")

        # 06 Settings: the Global tab
        window.show_page(4)
        app.processEvents()
        shot("06-settings.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

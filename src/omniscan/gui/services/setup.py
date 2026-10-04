"""Setup service: what the first-run wizard writes and measures (Qt-free).

One data folder for the library, work, output (and models) folders; the time left of a download at the rate so
far; and the runner that downloads the GPU runtime without a console window, each line of uv's output reported.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

from omniscan.core.config import Config
from omniscan.gui.services import settings

FOLDERS: tuple[str, ...] = ("library_root", "work_root", "output_root")  # what a data folder always holds
SUBFOLDERS = {"library_root": "library", "work_root": "work", "output_root": "output", "models_dir": "models"}


def data_folder(cfg: Config) -> Path:
    """The folder that holds the library, work and output folders today (their common parent), else ~/omniscan."""
    parents = {getattr(cfg.paths, key).parent for key in FOLDERS}
    return parents.pop() if len(parents) == 1 else Path.home() / "omniscan"


def use_data_folder(folder: Path, *, models: bool, path: Path | None = None) -> list[str]:
    """Point the library, work and output folders (and the models folder when `models`) into `folder`; returns
    the keys written. SettingError when a value is refused."""
    keys = [*FOLDERS, "models_dir"] if models else list(FOLDERS)
    for key in keys:
        settings.set_global("paths", key, str(folder / SUBFOLDERS[key]), path=path)
    return keys


def time_left(done: int, total: int, elapsed_s: float) -> float | None:
    """Seconds left of a download of `total` bytes at the rate so far; None before there is a rate to go by."""
    if total <= 0 or done <= 0 or elapsed_s < 1.0:
        return None
    return max(0.0, (total - done) * elapsed_s / done)


def quiet_runner(on_line: Callable[[str], None]) -> Callable[[list[str], Mapping[str, str]], int]:
    """A `runtime.install` runner for the desktop app: no console window, each non-empty output line to
    `on_line`; returns the exit code."""

    def run(command: list[str], env: Mapping[str, str]) -> int:
        with subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # Windows only
        ) as process:
            for line in process.stdout or ():
                if line.strip():
                    on_line(line.strip())
            return process.wait()

    return run

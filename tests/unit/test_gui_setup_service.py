"""Setup service (the first-run wizard's Qt-free part): the data folder, a download's time left, the quiet runner."""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

from omniscan.core.config import Config, PathsConfig
from omniscan.gui.services import setup


def test_the_data_folder_is_the_common_parent_else_the_default(tmp_path: Path) -> None:
    together = PathsConfig(
        library_root=tmp_path / "a" / "library",
        work_root=tmp_path / "a" / "work",
        output_root=tmp_path / "a" / "out",
    )
    assert setup.data_folder(Config(paths=together)) == tmp_path / "a"
    apart = together.model_copy(update={"work_root": tmp_path / "b" / "work"})
    assert setup.data_folder(Config(paths=apart)) == Path.home() / "omniscan"


def test_use_data_folder_writes_the_folders(tmp_path: Path) -> None:
    toml = tmp_path / "config.toml"
    assert setup.use_data_folder(tmp_path / "d", models=True, path=toml) == [
        "library_root",
        "work_root",
        "output_root",
        "models_dir",
    ]
    assert tomllib.loads(toml.read_text(encoding="utf-8"))["paths"]["models_dir"] == str(
        tmp_path / "d" / "models"
    )


def test_time_left_needs_a_rate_first() -> None:
    assert setup.time_left(0, 100, 5.0) is None
    assert setup.time_left(10, 100, 0.5) is None
    assert setup.time_left(10, 0, 5.0) is None
    assert setup.time_left(25, 100, 10.0) == 30.0
    assert setup.time_left(120, 100, 10.0) == 0.0  # the sizes are estimates: never below zero


def test_the_quiet_runner_reports_each_line_and_the_exit_code() -> None:
    lines: list[str] = []
    run = setup.quiet_runner(lines.append)
    script = "import sys; print('Resolved 2 packages', flush=True); print(flush=True); print('oops', file=sys.stderr); sys.exit(3)"
    code = run([sys.executable, "-c", script], os.environ)
    assert code == 3 and lines == ["Resolved 2 packages", "oops"]

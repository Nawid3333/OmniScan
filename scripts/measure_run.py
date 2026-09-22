"""Time full `omniscan run` commands in fresh child processes (G3 wall-clock measurements).

Usage (repo root):
  uv run --no-sync --frozen python scripts/measure_run.py SERIES CHAPTER [--runs N] [--models-dir DIR]

Each run is a fresh python process running the eight GPU stages with --force, exactly the command the
G3 card measures, with the timeline enabled on stderr; the script prints one wall-clock line per run.
`--models-dir` is passed through as OMNISCAN_PATHS__MODELS_DIR (a worktree's own models folder is
usually empty; the weights live in the main checkout's models dir).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_STAGES = ("ingest", "slice", "detect", "ocr", "inpaint", "inpaint_lama", "typeset", "export")


def main() -> int:
    """Run the command `--runs` times in fresh processes and print each wall clock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series")
    parser.add_argument("chapter")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--models-dir", default=os.environ.get("OMNISCAN_PATHS__MODELS_DIR"))
    parser.add_argument("--timeline", default="1", help="OMNISCAN_TIMELINE value (1 = stderr table)")
    parser.add_argument(
        "--stages",
        default=",".join(_STAGES),
        help="comma-separated stage list (default: the eight G3 stages)",
    )
    parser.add_argument(
        "--env", action="append", default=[], metavar="KEY=VALUE", help="extra child env var (repeatable)"
    )
    args = parser.parse_args()

    env = dict(os.environ)
    env["OMNISCAN_TIMELINE"] = args.timeline
    if args.models_dir:
        env["OMNISCAN_PATHS__MODELS_DIR"] = args.models_dir
    for override in args.env:
        key, _, value = override.partition("=")
        if not key:
            raise SystemExit(f"--env expects KEY=VALUE, got {override!r}")
        env[key] = value
    command = [
        sys.executable,
        "-c",
        "import sys; from omniscan.cli import app; sys.exit(app())",
        "run",
        args.series,
        "-c",
        args.chapter,
        "--force",
        *(name for stage in args.stages.split(",") for name in ("--stage", stage)),
    ]
    for run in range(1, args.runs + 1):
        started = time.perf_counter()
        result = subprocess.run(command, env=env, check=False)
        wall = time.perf_counter() - started
        print(f"measure_run {args.series}/{args.chapter} run {run}/{args.runs}: wall {wall:.1f} s")
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

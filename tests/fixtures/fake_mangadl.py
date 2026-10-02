"""A stand-in for manhwa-manga-downloader's `mangadl --json`, run as a real subprocess by the `--from-url` tests.

It follows the documented contract: chapter folders under `--out`, its bookkeeping files beside them, one result
object on stdout, messages on stderr, exit 0 / 2 / 1. The scenario is the URL's last path segment. It also
writes its own argv next to the output folder (`<out>.argv.json`) so a test can check the flags it got.

Usage: python fake_mangadl.py --out=DIR [--chapters=RANGE] --yes --json --no-convert -- URL
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _chapter(out: Path, number: int, pages: int) -> str:
    """Write one `num<N>_Chapter <N>` folder of `pages` small files; returns its name."""
    folder = out / f"num{number}_Chapter {number}"
    folder.mkdir(parents=True, exist_ok=True)
    for page in range(1, pages + 1):
        (folder / f"{page:04d}.jpg").write_bytes(b"page %d of chapter %d" % (page, number))
    return folder.name


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def main(args: list[str]) -> int:
    """Act out the scenario named by the URL; returns the exit code."""
    url = args[args.index("--") + 1]
    out = Path(next(arg for arg in args if arg.startswith("--out=")).removeprefix("--out="))
    _write_json(out.parent / (out.name + ".argv.json"), args)
    scenario = url.rsplit("/", 1)[-1]
    result: dict[str, Any] = {
        "schema": 1,
        "site": "fake",
        "series": "solo-leveling",
        "out_dir": str(out),
        "chapters": 0,
        "images": 0,
        "failed_chapters": 0,
        "complete_chapters": [],
        "incomplete_chapters": [],
    }
    code = 0
    if scenario in ("complete", "id", "legacy"):
        result["complete_chapters"] = [_chapter(out, 1, 2), _chapter(out, 2, 1)]
        _write_json(out / "chapter_manifest.json", {"chapters": {"num1_Chapter 1": 2, "num2_Chapter 2": 1}})
        if scenario == "id":
            result["series"] = "1234"
        if scenario == "legacy":  # what downloader 1.4.x printed
            for key in ("schema", "series", "complete_chapters"):
                del result[key]
    elif scenario == "partial":
        result["complete_chapters"] = [_chapter(out, 1, 2)]
        result["incomplete_chapters"] = [_chapter(out, 2, 1)]
        _write_json(
            out / "incomplete_chapters.json",
            {"chapters": [{"folder": "num2_Chapter 2", "downloaded": 1, "total": 3}]},
        )
        code = 2
    elif scenario == "none-finished":
        result["incomplete_chapters"] = [_chapter(out, 1, 1)]
        code = 2
    elif scenario == "error":
        result["error"] = "Unsupported site: no driver matched the domain"
        code = 1
    elif scenario == "schema9":
        result["schema"] = 9
    elif scenario == "silent":
        sys.stderr.write("Traceback (most recent call last):\nRuntimeError: boom\n")
        return 1
    sys.stderr.write(f"Site: fake ({scenario})\n")
    print(json.dumps(result))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

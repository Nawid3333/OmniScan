"""`omniscan import --from-url`: download with manhwa-manga-downloader (`mangadl --json`), then import its folder.

The downloader is a plug-in, never a dependency: it runs as a subprocess, found on PATH or set with
`[importer] downloader` in config.toml, and OmniScan reads only its documented `--json` result object (one JSON
object on stdout; exit code 0 complete, 2 some chapters incomplete, anything else an error). The folder it
writes is planned like any other downloader series folder (`importer/downloader.py`), restricted to the chapters
this run reports as finished.

Every URL downloads into its own folder, `work_root/_downloads/<hash of the URL>`, so a run that left chapters
incomplete can simply be repeated: the downloader resumes and keeps every page it already has. The folder is
removed (`discard_download`) once a run finished every chapter it was asked for and they were imported.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from omniscan.importer.downloader import downloader_series_name
from omniscan.importer.plan import ImportPlan, ImportPlanError, plan_import

DOWNLOADS_DIR = "_downloads"  # under work_root; the `_` prefix keeps it out of the series listings
# Result-object versions this importer understands. Downloader 1.4.x printed the same object without the
# field, so a missing `schema` reads as 1.
SUPPORTED_SCHEMAS = frozenset({1})
EXIT_OK = 0
EXIT_INCOMPLETE = 2
_STDERR_TAIL = 20  # lines of the downloader's messages kept for an error report

LineSink = Callable[[str], None]


class DownloaderError(ImportPlanError):
    """The downloader is missing, failed, or printed a result this importer does not understand."""


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """What one `mangadl --json` run reported."""

    out_dir: Path  # the folder it downloaded into (the one OmniScan chose)
    series: str | None  # the downloader's series folder name for the URL; None when it could not tell
    complete: (
        list[str] | None
    )  # chapter folders this run finished; None when the downloader does not report them
    incomplete: list[str]  # chapter folders still missing pages
    finished: bool  # every chapter it was asked for is complete (exit code 0)


def downloader_command(setting: str | Sequence[str]) -> list[str]:
    """The command that runs the downloader (`[importer] downloader`), its executable resolved on PATH."""
    parts = [setting] if isinstance(setting, str) else list(setting)
    if not parts or not parts[0].strip():
        raise DownloaderError("[importer] downloader is empty — set it to the mangadl executable")
    executable = shutil.which(parts[0])
    if executable is None:
        raise DownloaderError(
            f"can't find the downloader {parts[0]!r}: install manhwa-manga-downloader "
            "(https://github.com/Nawid3333/manhwa-manga-downloader) so `mangadl` is on PATH, or set "
            '[importer] downloader in config.toml to its command, e.g. ["<its venv python>", "<its main.py>"]'
        )
    return [executable, *parts[1:]]


def download_dir(work_root: Path, url: str) -> Path:
    """The folder `url` downloads into: `work_root/_downloads/<hash>`, the same for every run of that URL."""
    digest = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]
    return work_root / DOWNLOADS_DIR / digest


def download(
    url: str,
    dest: Path,
    *,
    command: Sequence[str],
    chapters: str | None = None,
    on_line: LineSink | None = None,
) -> DownloadResult:
    """Run the downloader for `url` into `dest` and read its result.

    Its messages go to this process's stderr (the CLI shows them live), or line by line to `on_line` when given
    (the desktop app). Pages are kept as the site serves them (`--no-convert`): the import converts them to
    JPEG once, at its own quality.
    """
    argv = [*command, url, "--out", str(dest), "--yes", "--json", "--no-convert"]
    if chapters is not None:
        argv += ["--chapters", chapters]
    dest.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # its messages decode the same way on every OS
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if on_line is not None else 0
    tail: deque[str] = deque(maxlen=_STDERR_TAIL)
    # stdout goes to a file, not a pipe: only stderr is read while the downloader runs, so a full stdout pipe
    # could never stall it.
    with tempfile.TemporaryFile(mode="w+b") as stdout_file:
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=subprocess.PIPE if on_line is not None else None,
                env=env,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise DownloaderError(f"can't start the downloader {argv[0]}: {exc}") from exc
        if proc.stderr is not None and on_line is not None:
            for raw in proc.stderr:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    tail.append(line)
                    on_line(line)
            proc.stderr.close()
        returncode = proc.wait()
        stdout_file.seek(0)
        stdout = stdout_file.read().decode("utf-8", errors="replace")
    return parse_result(stdout, returncode, dest, messages=list(tail))


def parse_result(stdout: str, returncode: int, dest: Path, *, messages: Sequence[str] = ()) -> DownloadResult:
    """The `DownloadResult` of one run from its stdout and exit code; DownloaderError for a failed run."""
    data = _result_object(stdout)
    if data is None:
        raise DownloaderError(
            f"the downloader exited with code {returncode} without printing a result" + _tail(messages)
        )
    schema = data.get("schema", 1)
    if schema not in SUPPORTED_SCHEMAS:
        raise DownloaderError(
            f"the downloader's result has schema {schema!r}, this OmniScan understands "
            f"{', '.join(map(str, sorted(SUPPORTED_SCHEMAS)))} — update OmniScan"
        )
    if returncode not in (EXIT_OK, EXIT_INCOMPLETE):
        error = data.get("error")
        reason = error if isinstance(error, str) and error else f"exit code {returncode}"
        raise DownloaderError(f"the download failed: {reason}" + _tail(messages))
    series = data.get("series")
    complete = data.get("complete_chapters")
    incomplete = _names(data.get("incomplete_chapters"))
    return DownloadResult(
        out_dir=dest,
        series=series if isinstance(series, str) and series else None,
        complete=_names(complete) if isinstance(complete, list) else None,
        incomplete=incomplete,
        finished=returncode == EXIT_OK and not incomplete,
    )


def plan_download(result: DownloadResult, *, series: str | None, chapter: str | None = None) -> ImportPlan:
    """Plan the downloaded folder: the chapters this run finished, under `series` or the downloader's name."""
    if result.complete is not None and not result.complete:
        missing = f" (incomplete: {', '.join(result.incomplete)})" if result.incomplete else ""
        raise ImportPlanError(
            f"the downloader finished no chapter{missing} — run the same command again to retry; "
            f"pages already downloaded are kept in {result.out_dir}"
        )
    if series is None:
        if result.series is None:
            raise ImportPlanError("the downloader could not tell this URL's series name — pass --series")
        series = downloader_series_name(result.series)
    plan = plan_import(result.out_dir, series=series, chapter=chapter)
    if result.complete is not None:
        finished = set(result.complete)
        items = [item for item in plan.items if item.files and item.files[0].parent.name in finished]
        if not items:
            raise ImportPlanError(
                f"none of the chapters this download finished can be imported from {result.out_dir}"
            )
        plan = replace(plan, items=items)
    if result.incomplete:
        plan = replace(
            plan,
            warnings=[
                *plan.warnings,
                f"{len(result.incomplete)} chapter(s) are still incomplete — run the same command again to "
                f"finish them; pages already downloaded are kept in {result.out_dir}",
            ],
        )
    return plan


def discard_download(result: DownloadResult) -> None:
    """Remove the download folder once everything in it has been imported (OSError when that fails)."""
    shutil.rmtree(result.out_dir)


def _result_object(stdout: str) -> dict[str, object] | None:
    """The JSON object on the last non-empty stdout line, or None when there is none."""
    last = next((line for line in reversed(stdout.splitlines()) if line.strip()), "")
    try:
        data = json.loads(last)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _names(value: object) -> list[str]:
    """The strings of a JSON list (anything else: empty)."""
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _tail(messages: Sequence[str]) -> str:
    """The downloader's last messages, for an error report ("" when none were kept)."""
    return ("\n" + "\n".join(messages)) if messages else ""

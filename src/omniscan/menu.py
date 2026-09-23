"""Interactive top-level menu: `omniscan` with no subcommand (card MENU1).

Every prompt takes `read` (default `input`) instead of calling `input` directly, so tests can inject
canned answers. EOFError from `read` is a normal "back"/"cancel" everywhere: `pick_from` returns
None for it; mid-flow prompts (`pick_chapters`, `pick_bool`, `pick_text`, `pick_int`) propagate it
and the submenu catches it, so a flow never half-executes and the process never crashes on EOF.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from pathlib import Path

import typer

from omniscan.cli import (
    PackFormat,
    UpdateChannel,
    cmd_gui,
    cmd_import,
    cmd_pack,
    cmd_run,
    cmd_serve,
    doctor,
    hardware,
    models_download,
    models_list,
    models_remove,
    models_verify,
    update_check,
)
from omniscan.core.config import Config
from omniscan.core.paths import SeriesPaths, natural_key
from omniscan.library.cli import CoverProvider, CoverSize, cover, info
from omniscan.match.cli import chapters as match_chapters
from omniscan.pipeline.stages import STAGE_ORDER
from omniscan.update.github import DEFAULT_REPO

TOP_LEVEL_OPTIONS: tuple[str, ...] = (
    "Run pipeline (a series' chapters through the full pipeline, or one stage at a time)",
    "Library (import chapters, package finished chapters, covers/metadata, match two chapter sets)",
    "Models (list, download, remove, verify)",
    "Diagnostics (doctor, hardware report, check for updates)",
    "Launch (web debug viewer, desktop app)",
)


def pick_from(
    options: Sequence[str], title: str, *, read: Callable[[str], str] = input, allow_back: bool = True
) -> int | None:
    """Print `title` then a 1-based numbered list of `options`; return the chosen 0-based index.

    "0" (shown only when `allow_back`), empty input or EOFError -> None ("back"/"cancel"); an
    out-of-range number or non-numeric input reprints "not a valid choice, try again" and asks again.
    """
    while True:
        typer.echo(title)
        for number, option in enumerate(options, 1):
            typer.echo(f"{number}. {option}")
        if allow_back:
            typer.echo("0. Back")
        try:
            answer = read("> ").strip()
        except EOFError:
            return None
        if not answer or (answer == "0" and allow_back):
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        typer.echo("not a valid choice, try again")


def pick_series(cfg: Config, *, read: Callable[[str], str] = input) -> str | None:
    """Let the user pick one of the library's series folders (natural order); None = back/no series."""
    root = cfg.paths.library_root
    names = (
        sorted(
            (p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))),
            key=natural_key,
        )
        if root.is_dir()
        else []
    )
    if not names:
        typer.echo(f"no series found in {root}")
        return None
    index = pick_from(names, "Series", read=read)
    return None if index is None else names[index]


def pick_chapters(sp: SeriesPaths, *, read: Callable[[str], str] = input) -> list[str] | None:
    """Prompt for which of `sp.chapters()` to act on; None = all, [] = the series has no chapters.

    Accepts "a"/"all" (case-insensitive) or empty input (both mean all), or a comma-separated list of
    1-based numbers returned as chapter folder names in the order given. An empty series prints
    "no chapters found for <series>" and returns []. EOFError propagates: a cancelled pick must
    never read as "all".
    """
    chapters = sp.chapters()
    if not chapters:
        typer.echo(f"no chapters found for {sp.series}")
        return []
    while True:
        typer.echo(f"Chapters of {sp.series}")
        for number, name in enumerate(chapters, 1):
            typer.echo(f"{number}. {name}")
        typer.echo("a. all")
        answer = read("Chapters to act on (numbers, 'a' = all): ").strip()
        if not answer or answer.casefold() in ("a", "all"):
            return None
        tokens = [token.strip() for token in answer.split(",")]
        if all(token.isdigit() and 1 <= int(token) <= len(chapters) for token in tokens):
            return [chapters[int(token) - 1] for token in tokens]
        typer.echo("not a valid choice, try again")


def pick_bool(prompt: str, default: bool, *, read: Callable[[str], str] = input) -> bool:
    """Ask a yes/no question ("{prompt} [Y/n]"/"{prompt} [y/N]"); empty input -> `default`."""
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = read(f"{prompt} {hint} ").strip().casefold()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def pick_text(prompt: str, default: str | None, *, read: Callable[[str], str] = input) -> str | None:
    """Ask for a text answer ("{prompt} [{default}]: " or "{prompt}: "); empty input -> `default`."""
    suffix = f" [{default}]" if default is not None else ""
    return read(f"{prompt}{suffix}: ").strip() or default


def pick_int(prompt: str, default: int, *, read: Callable[[str], str] = input) -> int:
    """Like `pick_text` but parses an int; a non-integer answer reprints the prompt and asks again."""
    while True:
        answer = read(f"{prompt} [{default}]: ").strip()
        if not answer:
            return default
        try:
            return int(answer)
        except ValueError:
            continue


def pause(*, read: Callable[[str], str] = input) -> None:
    """Print "Press Enter to continue..." and read one line, discarding it (EOF is fine too)."""
    typer.echo("Press Enter to continue...")
    with contextlib.suppress(EOFError):
        read("")


def _run_action(label: str, action: Callable[[], object], *, read: Callable[[str], str]) -> None:
    """Run one menu action, then pause before the submenu redraws.

    `typer.Exit` (every command's own "nothing to do"/"bad input" path) and Ctrl+C are caught here so
    one failed action never kills the menu process; a non-zero exit prints "({label}: exit {n})".
    """
    try:
        try:
            action()
        except typer.Exit as exc:
            if exc.exit_code:
                typer.echo(f"({label}: exit {exc.exit_code})")
        except KeyboardInterrupt:
            typer.echo(f"({label}: interrupted)")
    finally:
        pause(read=read)


def run_menu(cfg: Config, *, read: Callable[[str], str] = input) -> int:
    """Interactive top-level menu loop; returns a process exit code (0 normally).

    Loops printing the main menu and dispatching to the picked category's submenu until the user
    picks "0" (Exit), enters "q" or `read` raises EOFError — all three exit cleanly with 0.
    """
    menus: tuple[Callable[..., None], ...] = (
        menu_pipeline,
        menu_library,
        menu_models,
        menu_diagnostics,
        menu_launch,
    )
    while True:
        index = pick_from(TOP_LEVEL_OPTIONS, "OmniScan", read=_read_or_quit(read))
        if index is None:  # "0" (Exit), empty input or EOF: leave the menu cleanly
            return 0
        menus[index](cfg, read=read)


def _read_or_quit(read: Callable[[str], str]) -> Callable[[str], str]:
    """`read` wrapper for the top-level prompt: "q" means exit (same as EOF/Ctrl+D)."""

    def wrapper(prompt: str) -> str:
        answer = read(prompt)
        if answer.strip().casefold() == "q":
            raise EOFError
        return answer

    return wrapper


def menu_pipeline(cfg: Config, *, read: Callable[[str], str] = input) -> None:
    """Run pipeline submenu: every stage, or one stage, for a picked series' chapters."""
    options = ("Run everything (ingest -> export) for a series", "Run one stage for a series")
    while True:
        index = pick_from(options, "Run pipeline", read=read)
        if index is None:
            return
        try:
            if index == 0:
                _run_everything(cfg, read=read)
            else:
                _run_one_stage(cfg, read=read)
        except EOFError:  # input ended mid-flow: cancel the flow, redraw this submenu
            pass


def _run_everything(cfg: Config, *, read: Callable[[str], str]) -> None:
    """Pipeline leaf 1: `cmd_run` with all ten stages (LaMa and force asked as yes/no questions)."""
    series = pick_series(cfg, read=read)
    if series is None:
        return
    chapters = pick_chapters(SeriesPaths.from_config(cfg, series), read=read)
    if chapters is not None and not chapters:
        typer.echo("nothing to do")
        return
    no_lama = pick_bool("Skip the LaMa inpaint stage?", False, read=read)
    force = pick_bool("Force re-run stages that are already up to date?", False, read=read)
    _run_action(
        "run",
        lambda: cmd_run(
            series=series,
            chapter=chapters,
            stage=None,
            no_lama=no_lama,
            force=force,
            step=False,
            preview_chapter=None,
        ),
        read=read,
    )


def _run_one_stage(cfg: Config, *, read: Callable[[str], str]) -> None:
    """Pipeline leaf 2: `cmd_run` with exactly one chosen stage."""
    series = pick_series(cfg, read=read)
    if series is None:
        return
    chapters = pick_chapters(SeriesPaths.from_config(cfg, series), read=read)
    if chapters is not None and not chapters:
        typer.echo("nothing to do")
        return
    index = pick_from(list(STAGE_ORDER), "Stage", read=read)
    if index is None:
        return
    stage = STAGE_ORDER[index]
    force = pick_bool("Force re-run stages that are already up to date?", False, read=read)
    _run_action(
        "run",
        lambda: cmd_run(
            series=series,
            chapter=chapters,
            stage=[stage],
            no_lama=False,
            force=force,
            step=False,
            preview_chapter=None,
        ),
        read=read,
    )


def menu_library(cfg: Config, *, read: Callable[[str], str] = input) -> None:
    """Library submenu: import, pack, covers/metadata and chapter-set matching."""
    options = (
        "Import chapters from a folder or archive",
        "Package finished chapters into CBZ/PDF",
        "Fetch cover and metadata",
        "Show a series' stored metadata",
        "Match two independently-sourced chapter sets by page art",
    )
    while True:
        index = pick_from(options, "Library", read=read)
        if index is None:
            return
        try:
            if index == 0:
                _import_chapters(read=read)
            elif index == 1:
                _pack_chapters(cfg, read=read)
            elif index == 2:
                _fetch_cover(cfg, read=read)
            elif index == 3:
                _show_metadata(cfg, read=read)
            else:
                _match_chapter_sets(read=read)
        except EOFError:  # input ended mid-flow: cancel the flow, redraw this submenu
            pass


def _import_chapters(*, read: Callable[[str], str]) -> None:
    """Library leaf 1: import a folder or .zip/.cbz archive into the library."""
    path = pick_text("Path to a folder or .zip/.cbz archive", None, read=read)
    if not path:
        return
    source = Path(path)
    if not source.exists():
        typer.echo(f"import: {source} does not exist")
        return
    series = pick_text("Series name (blank = guess from the path)", None, read=read)
    chapter = pick_text("Chapter name (blank = guess from the path)", None, read=read)
    move = pick_bool("Move instead of copy (deletes the source)?", False, read=read)
    dry_run = pick_bool("Dry run (show the plan, write nothing)?", False, read=read)
    _run_action(
        "import",
        lambda: cmd_import(
            source=source, series=series or None, chapter=chapter or None, move=move, dry_run=dry_run
        ),
        read=read,
    )


def _pack_chapters(cfg: Config, *, read: Callable[[str], str]) -> None:
    """Library leaf 2: package output chapters into CBZ/PDF (None = every chapter under output_root)."""
    series = pick_series(cfg, read=read)
    if series is None:
        return
    chapters = pick_chapters(SeriesPaths.from_config(cfg, series), read=read)
    index = pick_from(["cbz", "pdf", "both"], "Format", read=read)
    if index is None:
        return
    formats = {0: [PackFormat.CBZ], 1: [PackFormat.PDF], 2: [PackFormat.CBZ, PackFormat.PDF]}[index]
    _run_action("pack", lambda: cmd_pack(series=series, chapter=chapters, fmt=formats, out=None), read=read)


def _fetch_cover(cfg: Config, *, read: Callable[[str], str]) -> None:
    """Library leaf 3: find the series online and save its cover (the common path; no --pick/--file)."""
    series = pick_series(cfg, read=read)
    if series is None:
        return
    title = pick_text("Search title (blank = series name)", None, read=read)
    _run_action(
        "cover",
        lambda: cover(
            series=series,
            title=title or None,
            provider=CoverProvider.ALL,
            pick=None,
            file=None,
            size=CoverSize.LARGE,
            as_json=False,
        ),
        read=read,
    )


def _show_metadata(cfg: Config, *, read: Callable[[str], str]) -> None:
    """Library leaf 4: print the series' stored cover metadata."""
    series = pick_series(cfg, read=read)
    if series is None:
        return
    _run_action("info", lambda: info(series=series, as_json=False), read=read)


def _match_chapter_sets(*, read: Callable[[str], str]) -> None:
    """Library leaf 5: align two independently-sourced chapter sets by page art (threshold defaults)."""
    first = pick_text("First chapter set folder", None, read=read)
    if not first:
        return
    second = pick_text("Second chapter set folder", None, read=read)
    if not second:
        return
    dir_a = Path(first)
    dir_b = Path(second)
    for path in (dir_a, dir_b):
        if not (path.exists() and path.is_dir()):
            typer.echo(f"match: {path} does not exist or is not a folder")
            return
    _run_action(
        "match",
        lambda: match_chapters(
            dir_a=dir_a,
            dir_b=dir_b,
            out=None,
            force=False,
            page_similarity=0.75,
            page_gap=0.25,
            chapter_gap=0.6,
            min_quality=0.3,
            review_quality=0.5,
            as_json=False,
        ),
        read=read,
    )


def menu_models(cfg: Config, *, read: Callable[[str], str] = input) -> None:
    """Models submenu: list, download, remove and verify (the commands read the config themselves)."""
    options = (
        "List all models (install status, hardware fit)",
        "Download models",
        "Remove installed models",
        "Verify installed models",
    )
    while True:
        index = pick_from(options, "Models", read=read)
        if index is None:
            return
        try:
            if index == 0:
                _list_models(read=read)
            elif index == 1:
                _download_models(read=read)
            elif index == 2:
                _remove_models(read=read)
            else:
                _run_action("models", lambda: models_verify(ids=None, deep=False), read=read)
        except EOFError:  # input ended mid-flow: cancel the flow, redraw this submenu
            pass


def _list_models(*, read: Callable[[str], str]) -> None:
    """Models leaf 1: the catalog table with install status and hardware fit."""
    _run_action("models", lambda: models_list(as_json=False, role=None, lang=None), read=read)


def _download_models(*, read: Callable[[str], str]) -> None:
    """Models leaf 2: list first, then download the typed ids (blank: every missing required model)."""
    _list_models(read=read)
    ids = pick_text("Model id(s), comma-separated (blank = every missing required model)", None, read=read)
    if not ids:
        _run_action("models", lambda: models_download(ids=None, required=True, force=False), read=read)
        return
    chosen = [part.strip() for part in ids.split(",") if part.strip()]
    _run_action("models", lambda: models_download(ids=chosen, required=False, force=False), read=read)


def _remove_models(*, read: Callable[[str], str]) -> None:
    """Models leaf 3: list first, then remove the typed ids (blank: nothing removed)."""
    _list_models(read=read)
    ids = pick_text("Model id(s) to remove, comma-separated", None, read=read)
    if not ids:
        return
    chosen = [part.strip() for part in ids.split(",") if part.strip()]
    _run_action("models", lambda: models_remove(ids=chosen), read=read)


def menu_diagnostics(cfg: Config, *, read: Callable[[str], str] = input) -> None:
    """Diagnostics submenu: doctor, hardware report and the update check."""
    options = (
        "Doctor (check this machine is ready)",
        "Hardware report",
        "Check for a newer OmniScan release",
    )
    while True:
        index = pick_from(options, "Diagnostics", read=read)
        if index is None:
            return
        try:
            if index == 0:
                _run_action("doctor", lambda: doctor(as_json=False), read=read)
            elif index == 1:
                _run_action("hardware", lambda: hardware(as_json=False), read=read)
            else:
                _run_action(
                    "update",
                    lambda: update_check(channel=UpdateChannel.STABLE, repo=DEFAULT_REPO, as_json=False),
                    read=read,
                )
        except EOFError:  # input ended mid-flow: cancel the flow, redraw this submenu
            pass


def menu_launch(cfg: Config, *, read: Callable[[str], str] = input) -> None:
    """Launch submenu: the web debug viewer's API and the desktop app."""
    options = (
        "Web debug viewer (starts the API on 127.0.0.1:8000; run `npm run dev` in webui/ for the UI)",
        "Desktop app",
    )
    while True:
        index = pick_from(options, "Launch", read=read)
        if index is None:
            return
        try:
            if index == 0:
                _serve(read=read)
            else:
                _run_action("gui", cmd_gui, read=read)
        except EOFError:  # input ended mid-flow: cancel the flow, redraw this submenu
            pass


def _serve(*, read: Callable[[str], str]) -> None:
    """Launch leaf 1: run the debug API until Ctrl+C (which returns to the Launch submenu)."""
    typer.echo("Starting the debug API on http://127.0.0.1:8000 (Ctrl+C to stop)...")
    with contextlib.suppress(KeyboardInterrupt):
        cmd_serve(host="127.0.0.1", port=8000, reload=False)
    pause(read=read)

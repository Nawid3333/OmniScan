# MENU1 — Interactive menu, part 1 (framework + core flows)

**Owner:** GLM builder · **Branch:** `MENU1` · **Worktree:** `V:\OmniScan-wt\MENU1` (sibling of the repo, created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. It overrides nothing below but applies everywhere.

## Why (context, not instructions)
The owner wants to run OmniScan without memorising subcommands and flags: type `omniscan`, get a
numbered menu, pick things by number, done. This card builds the menu **framework** (pickers, the
menu loop, wiring into the CLI's bare invocation) plus the flows people reach for constantly: running
the pipeline, importing/packing chapters, model/hardware diagnostics, updates, and launching the
GUI/web viewer. A follow-up card (MENU2, not written yet) adds the rest (translation/glossary
workflow, watermark, promo filter, job queue) on top of the same framework.

**This is additive, not a replacement.** Every existing subcommand (`omniscan run SERIES ...`,
`omniscan doctor`, etc.) keeps working exactly as it does today — tests, `scripts/omni_builder.py`,
CI, and anyone scripting the CLI all call those directly and must not see any behaviour change. The
only new thing: running `omniscan` with **no subcommand** now opens the menu instead of printing
`--help`.

## Files you may create / modify
- `src/omniscan/menu.py` (create) — the framework + every leaf this card covers
- `src/omniscan/cli.py` (modify — only the `main()` callback and its `typer.Typer(...)` construction;
  do not touch any command function)
- `tests/unit/test_menu.py` (create)
Anything not listed is off-limits (especially `src/omniscan/core/**`).

## The one wiring change to `cli.py`
Today:
```python
app = typer.Typer(help="OmniScan — manhwa/manga translator", no_args_is_help=True)


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug log output.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Warnings only.")] = False,
) -> None:
    """OmniScan global options."""
    setup_logging("DEBUG" if verbose else "WARNING" if quiet else "INFO")
```
Change to:
```python
app = typer.Typer(help="OmniScan — manhwa/manga translator", invoke_without_command=True)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug log output.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Warnings only.")] = False,
) -> None:
    """OmniScan global options. No subcommand: opens the interactive menu."""
    setup_logging("DEBUG" if verbose else "WARNING" if quiet else "INFO")
    if ctx.invoked_subcommand is None:
        from omniscan.menu import run_menu

        raise typer.Exit(run_menu(get_config()))
```
`omniscan --help` must still print the normal Typer help (Click handles `--help` before the callback
body runs when `invoke_without_command=True` — verify this manually, it's the whole point of the
change: `omniscan --help` unchanged, `omniscan <command> ...` unchanged, `omniscan` alone now opens
the menu). `get_config` is already imported in `cli.py`.

## Interfaces (exact — do not rename or change signatures)
All in the new `src/omniscan/menu.py`.

```python
def run_menu(cfg: Config, *, read: Callable[[str], str] = input) -> int:
    """Interactive top-level menu loop; returns a process exit code (0 normally).

    Every prompt in this module and everything it calls takes `read` (or receives it via closure)
    instead of calling `input` directly, so tests can inject canned answers. Loops printing the main
    menu, calling `pick_from` on the top-level category list, dispatching to that category's
    submenu function, and looping again until the user picks "Exit" (or `read` raises EOFError /
    the user enters "q" — both mean "exit cleanly", return 0)."""
```

```python
def pick_from(
    options: Sequence[str], title: str, *, read: Callable[[str], str] = input, allow_back: bool = True
) -> int | None:
    """Print `title` then a 1-based numbered list of `options`; return the chosen 0-based index.

    "0" (only shown when `allow_back`) or empty input or EOFError -> return None ("back"/"cancel").
    An out-of-range number or non-numeric input reprints "not a valid choice, try again" and asks
    again (does not raise, does not exit the menu)."""


def pick_series(cfg: Config, *, read: Callable[[str], str] = input) -> str | None:
    """List `cfg.paths.library_root`'s subdirectories (sorted by `omniscan.core.paths.natural_key`,
    skipping names starting with "_" or "."), let the user pick one via `pick_from`. An empty or
    missing library_root prints "no series found in <library_root>" and returns None without
    prompting."""


def pick_chapters(sp: SeriesPaths, *, read: Callable[[str], str] = input) -> list[str] | None:
    """Prompt for which of `sp.chapters()` to act on. Shows the numbered chapter list plus an "a"
    option meaning "all". Accepts a comma-separated list of 1-based numbers ("1,3,4") or "a"/"all"
    (case-insensitive) or empty input (also means all). Returns `None` for "all" (matching every
    existing `--chapter` option's own "None = every chapter" default) or the list of chosen chapter
    folder names for a specific subset. Out-of-range or non-numeric entries in the list reprint the
    prompt and ask again. `sp.chapters()` empty -> print "no chapters found for <series>" and
    return `[]` (the caller must treat an empty list as "nothing to do", not as "all")."""


def pick_bool(prompt: str, default: bool, *, read: Callable[[str], str] = input) -> bool:
    """"{prompt} [Y/n]" or "{prompt} [y/N]" depending on `default`; y/yes -> True, n/no -> False,
    empty input -> `default`, anything else reprints and asks again (case-insensitive)."""


def pick_text(prompt: str, default: str | None, *, read: Callable[[str], str] = input) -> str | None:
    """"{prompt} [{default}]: " (or "{prompt}: " when `default` is None); empty input -> `default`,
    otherwise the trimmed input."""


def pick_int(prompt: str, default: int, *, read: Callable[[str], str] = input) -> int:
    """Like `pick_text` but parses an int; a non-integer answer reprints and asks again."""


def pause(*, read: Callable[[str], str] = input) -> None:
    """Print "Press Enter to continue..." and read one line, discarding it (lets the user read a
    command's output before the menu redraws)."""
```

Category submenu functions (each: print its own numbered list via `pick_from`, dispatch, loop until
"back"; same shape as `run_menu`, all take `(cfg: Config, *, read: Callable[[str], str] = input) -> None`):

```python
def menu_pipeline(cfg: Config, *, read: Callable[[str], str] = input) -> None: ...
def menu_library(cfg: Config, *, read: Callable[[str], str] = input) -> None: ...
def menu_models(cfg: Config, *, read: Callable[[str], str] = input) -> None: ...
def menu_diagnostics(cfg: Config, *, read: Callable[[str], str] = input) -> None: ...
def menu_launch(cfg: Config, *, read: Callable[[str], str] = input) -> None: ...
```

## Top-level menu (`run_menu`)
```
OmniScan
1. Run pipeline (a series' chapters through the full pipeline, or one stage at a time)
2. Library (import chapters, package finished chapters, covers/metadata, match two chapter sets)
3. Models (list, download, remove, verify)
4. Diagnostics (doctor, hardware report, check for updates)
5. Launch (web debug viewer, desktop app)
0. Exit
```
Picking 1-5 calls the matching `menu_*` function; that function loops its own submenu until the user
backs out (enters 0), then control returns here. Every command invocation below: run it, print
whatever it prints (these functions already call `typer.echo`/`Console().print` — nothing to
suppress or wrap), catch `typer.Exit` around the call (a failed command exits its own function via
`typer.Exit(n)` for n != 0; catch it, print `"({label}: exit {n})"` when `n != 0`, print nothing extra
when `n == 0`) so one failed action never kills the menu process, then `pause()` before redrawing the
submenu.

## Submenu: `menu_pipeline`
```
Run pipeline
1. Run everything (ingest -> export) for a series
2. Run one stage for a series
0. Back
```
- **1. Run everything:** `pick_series` (back to submenu if None) -> `pick_chapters` (empty list ->
  print "nothing to do" and return to submenu) -> `pick_bool("Skip the LaMa inpaint stage?", False)`
  -> `pick_bool("Force re-run stages that are already up to date?", False)` -> call
  `omniscan.cli.cmd_run(series=series, chapter=chapters, stage=None, no_lama=no_lama, force=force,
  step=False, preview_chapter=None)` (step mode is out of scope for the menu — a step-by-step run
  needs its own preview UI, not a fit for a numbered menu; the CLI flag stays available for anyone
  who wants it directly).
- **2. Run one stage:** `pick_series` -> `pick_chapters` -> `pick_from` over
  `list(omniscan.pipeline.stages.STAGE_ORDER)` (import it) to choose exactly one stage -> `pick_bool`
  for force -> call `cmd_run(series=series, chapter=chapters, stage=[chosen_stage], no_lama=False,
  force=force, step=False, preview_chapter=None)`.

## Submenu: `menu_library`
```
Library
1. Import chapters from a folder or archive
2. Package finished chapters into CBZ/PDF
3. Fetch cover and metadata
4. Show a series' stored metadata
5. Match two independently-sourced chapter sets by page art
0. Back
```
- **1. Import:** `pick_text("Path to a folder or .zip/.cbz archive", None)`; if empty, back to
  submenu. Validate the path exists with `pathlib.Path(...).exists()` before calling (print
  `"import: <path> does not exist"` and return to the submenu if not, matching the existing command's
  own `exists=True` Typer validation) -> `pick_text("Series name (blank = guess from the path)",
  None)` -> `pick_text("Chapter name (blank = guess from the path)", None)` -> `pick_bool("Move
  instead of copy (deletes the source)?", False)` -> `pick_bool("Dry run (show the plan, write
  nothing)?", False)` -> call `omniscan.cli.cmd_import(source=Path(path), series=series or None,
  chapter=chapter or None, move=move, dry_run=dry_run)`.
- **2. Package:** `pick_series` -> `pick_chapters` (`None` here means "every chapter under
  output_root", matching `cmd_pack`'s own default — do not turn it into `[]`) -> `pick_from(["cbz",
  "pdf", "both"], "Format")` -> map to `cmd_pack`'s `fmt` argument (`[PackFormat.CBZ]`,
  `[PackFormat.PDF]`, or both; import `PackFormat` from `omniscan.cli`) -> call
  `omniscan.cli.cmd_pack(series=series, chapter=chapters, fmt=fmt, out=None)`.
- **3. Fetch cover/metadata:** `pick_series` -> `pick_text("Search title (blank = series name)",
  None)` -> call `omniscan.library.cli.cover(series=series, title=title or None,
  provider=CoverProvider.ALL, pick=None, file=None, size=CoverSize.LARGE, as_json=False)` (import
  `CoverProvider`, `CoverSize`, `cover` from `omniscan.library.cli`). If it prints a numbered
  candidate list and exits 1 because nothing was confident enough, that is normal —
  the existing "no confident match" message already tells the user to re-run with `--pick`, which the
  menu does not need to replicate; the menu path is "good enough for the common case", not
  feature-complete with every flag.
- **4. Show metadata:** `pick_series` -> call `omniscan.library.cli.info(series=series,
  as_json=False)`.
- **5. Match two chapter sets:** `pick_text("First chapter set folder", None)` -> `pick_text("Second
  chapter set folder", None)`; validate both exist (`Path.exists() and Path.is_dir()`) before calling,
  same pattern as Import's validation -> call `omniscan.match.cli.chapters(dir_a=Path(a),
  dir_b=Path(b), out=None, force=False, page_similarity=0.75, page_gap=0.25, chapter_gap=0.6,
  min_quality=0.3, review_quality=0.5, as_json=False)`.

## Submenu: `menu_models`
```
Models
1. List all models (install status, hardware fit)
2. Download models
3. Remove installed models
4. Verify installed models
0. Back
```
- **1. List:** call `omniscan.cli.models_list(as_json=False, role=None, lang=None)`.
- **2. Download:** run `models_list` first so the user can see ids, then `pick_text("Model id(s), comma-separated (blank = every missing required model)", None)` -> if blank, call
  `omniscan.cli.models_download(ids=None, required=True, force=False)`; otherwise split on "," and
  strip whitespace, call `models_download(ids=[...], required=False, force=False)`.
- **3. Remove:** run `models_list` first, then `pick_text("Model id(s) to remove, comma-separated",
  None)`; blank -> back to submenu (nothing removed); otherwise call
  `omniscan.cli.models_remove(ids=[...])`.
- **4. Verify:** call `omniscan.cli.models_verify(ids=None, deep=False)`.

## Submenu: `menu_diagnostics`
```
Diagnostics
1. Doctor (check this machine is ready)
2. Hardware report
3. Check for a newer OmniScan release
0. Back
```
- **1.** call `omniscan.cli.doctor(as_json=False)`.
- **2.** call `omniscan.cli.hardware(as_json=False)`.
- **3.** call `omniscan.update.cli` — there is no such module; call
  `omniscan.cli.update_check(channel=omniscan.cli.UpdateChannel.STABLE, repo=omniscan.update.github.DEFAULT_REPO, as_json=False)`.

## Submenu: `menu_launch`
```
Launch
1. Web debug viewer (starts the API on 127.0.0.1:8000 -- run `npm run dev` in webui/ yourself for the UI)
2. Desktop app
0. Back
```
- **1.** Print the line `"Starting the debug API on http://127.0.0.1:8000 (Ctrl+C to stop)..."`, then
  call `omniscan.cli.cmd_serve(host="127.0.0.1", port=8000, reload=False)` (this blocks until Ctrl+C /
  `KeyboardInterrupt`, exactly like running `omniscan serve` directly today — catch
  `KeyboardInterrupt` around the call so Ctrl+C returns to the submenu instead of killing the whole
  menu process).
- **2.** call `omniscan.cli.cmd_gui()`; catch `typer.Exit` as everywhere else.

## Definitions (no interpretation needed)
- "Back to submenu" / "0" always means: stop the current flow (even mid-prompt-sequence, i.e. `None`
  from any `pick_*` call in a flow above aborts that whole flow, not just the current prompt) and
  redraw the submenu the user came from. Nothing partially executes: e.g. in "Run everything", if
  `pick_chapters` is cancelled (EOFError) after `pick_series` already returned a value, the flow still
  aborts without calling `cmd_run`.
- EOFError from `read` anywhere (not just at the top level) is treated exactly like the user entering
  "0"/empty at that specific prompt — propagate it up as a normal "back"/"cancel", never let it
  crash the process. (`run_menu` itself also returns 0 on EOFError at its own top-level prompt.)
- `typer.Exit` raised by a called command function is not a bug — every `cmd_*`/`cover`/`info`/
  `chapters` function already raises it on its own normal "nothing to do" / "bad input" / "partial
  failure" paths (see their existing bodies in `cli.py`/`library/cli.py`/`match/cli.py`). The menu
  must catch it at the point of each call (not with one blanket `try` around the whole submenu loop,
  so the submenu itself keeps running after one failed action).
- `Config`, `SeriesPaths`, `get_config` come from `omniscan.core.config`/`omniscan.core.paths` (already
  used the same way in `cli.py` — copy those imports).

## Acceptance tests (must exist and pass)
1. `test_pick_from_valid_choice`, `test_pick_from_zero_returns_none`, `test_pick_from_out_of_range_reprompts`,
   `test_pick_from_non_numeric_reprompts`, `test_pick_from_eof_returns_none`: drive `pick_from` with a
   `read` that is a list-popping fake (`iter(["not a number", "9", "2"]).__next__` style helper) and
   assert the returned index / reprompt count / `None`.
2. `test_pick_series_lists_library_dirs_sorted_naturally`: a `tmp_path` library_root with dirs
   `["Z", "A", "_meta", ".hidden", "Chapter 10", "Chapter 2"]` created via `tmp_path.mkdir` (files
   too, e.g. a stray `.txt`, to prove only directories are listed) -> only the four real series dirs
   are offered, in natural order, `_meta`/`.hidden` and the file excluded.
3. `test_pick_series_empty_library_returns_none_without_prompting`: `library_root` missing entirely ->
   `pick_series` returns `None` and the passed `read` fake is never called (assert a call counter is 0).
4. `test_pick_chapters_all_via_blank_and_via_a`: a `SeriesPaths` over 3 real chapter folders (build
   with the same fixture helper pattern `tests/unit/test_cli.py` already uses for chapters, or a
   `tmp_path`-based `SeriesPaths.from_config`) -> both `""` and `"a"` return `None`; `"1,3"` returns
   the two matching folder names in the order given; `"5"` (out of range) reprompts once then accepts
   a valid follow-up.
5. `test_pick_chapters_empty_series_returns_empty_list_without_prompting`: zero chapters -> `[]`,
   `read` never called.
6. `test_pick_bool_default_on_blank`, `test_pick_text_default_on_blank`, `test_pick_int_reprompts_on_non_integer`.
7. `test_run_menu_exit_immediately`: `read` fake returns `"0"` once -> `run_menu` returns `0` without
   calling anything else.
8. `test_run_menu_eof_returns_zero`: `read` fake raises `EOFError` on first call -> `run_menu` returns
   `0`.
9. `test_menu_pipeline_run_everything_calls_cmd_run`: monkeypatch `omniscan.menu.cmd_run` (imported
   name inside `menu.py`) with a fake recording its kwargs; drive `menu_pipeline` through "1" (run
   everything), a real series/chapter pick, "n"/"n" for the two bools, then "0","0" to exit both
   submenu and (via the caller) confirm the fake was called once with `stage=None`,
   `no_lama=False`, `force=False` and the picked `series`/`chapter`.
10. `test_menu_pipeline_run_one_stage_calls_cmd_run_with_single_stage_list`: same pattern, picks stage
    3 of `STAGE_ORDER`, asserts `stage=[STAGE_ORDER[2]]`.
11. `test_menu_dispatch_catches_typer_exit_and_continues`: monkeypatch a called function (e.g.
    `omniscan.menu.doctor`) to `raise typer.Exit(1)`; drive `menu_diagnostics` through "1" then "0" ->
    no exception propagates out of `menu_diagnostics`, and a follow-up call in the same submenu
    session (e.g. picking "1" again before "0") still runs (proves the loop survived the exit).
12. `test_menu_launch_serve_keyboard_interrupt_returns_to_submenu`: monkeypatch
    `omniscan.menu.cmd_serve` to `raise KeyboardInterrupt`; drive `menu_launch` through "1" then "0" ->
    no exception propagates.
13. `test_bare_cli_invocation_opens_menu`: `typer.testing.CliRunner().invoke(app, [])` with
    `monkeypatch.setattr("omniscan.menu.run_menu", lambda cfg, **_: 0)` (patch before invoking) ->
    exit code 0, the fake was called exactly once. Add this test to `tests/unit/test_cli.py` instead
    of `test_menu.py` (it belongs with the CLI's own tests) — this is the one addition allowed there;
    do not touch any other existing test in that file.
14. `test_cli_help_still_works`: `CliRunner().invoke(app, ["--help"])` -> exit code 0, output contains
    `"Usage:"` (proves `--help` was not broken by `invoke_without_command=True`). Same file as #13.
15. `test_cli_known_subcommand_unaffected`: `CliRunner().invoke(app, ["version"])` -> exit code 0
    (proves a normal subcommand invocation is unaffected). Same file as #13.

## Out of scope
- MENU2 (translate/judge/story/glossary/reference, watermark, promo filter, job queue) — a follow-up
  card on top of this same framework.
- Step-mode (`--step`/`--preview-chapter`) pipeline runs, `models_download`'s `--force`
  incompatible-hardware override, `library cover --pick`/`--file`, `match chapters`' threshold flags,
  `update download` (staging an update file) — all remain CLI-only for now; the menu covers the
  common path, not every flag of every command.
- A raw config/`series.toml` editor — point people at `omniscan gui`'s Settings page instead (already
  built, per `docs/USER_GUIDE.md`); do not build a second settings UI here.
- Any change to a `cmd_*`/`cover`/`info`/`chapters` function's own behaviour. This card only adds
  callers.

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_menu.py tests/unit/test_cli.py
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
Write `docs/reports/MENU1.md` (sections: Changes, Tests, Deviations, Questions) and commit
`MENU1: <summary>`. If anything above is unclear: stop, write the question under Questions, commit
what you have, and end.

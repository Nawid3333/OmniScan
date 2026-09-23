# MENU1 — Interactive menu, part 1 (framework + core flows)

## Changes

- `src/omniscan/menu.py` (new): the menu framework and every leaf this card covers.
  - Prompts: `pick_from`, `pick_series`, `pick_chapters`, `pick_bool`, `pick_text`, `pick_int`,
    `pause` — all take `read` (default `input`) so tests inject canned answers.
  - Loops: `run_menu` plus `menu_pipeline`, `menu_library`, `menu_models`, `menu_diagnostics`,
    `menu_launch`; leaves are private helpers (`_run_everything`, `_run_one_stage`, `_import_chapters`,
    `_pack_chapters`, `_fetch_cover`, `_show_metadata`, `_match_chapter_sets`, `_download_models`,
    `_remove_models`, `_list_models`, `_serve`).
  - `_run_action` is the single call point for every command invocation: catches `typer.Exit`
    (prints `(<label>: exit <n>)` only when n != 0) and Ctrl+C, then `pause()`s before the submenu
    redraws. Command functions are imported at module level (`from omniscan.cli import cmd_run, …`)
    so tests can monkeypatch `omniscan.menu.<name>`.
- `src/omniscan/cli.py` — the one wiring change from the card: `no_args_is_help=True` ->
  `invoke_without_command=True` (app + callback), `main` gains `ctx: typer.Context`, and a bare
  invocation raises `typer.Exit(run_menu(get_config()))`. No command function touched.
- `tests/unit/test_menu.py` (new), `tests/unit/test_cli.py` (+3 tests: bare invocation, `--help`,
  known subcommand).

EOFError semantics (card Definitions): `pick_from` returns None on EOF ("back"); mid-flow prompts
(`pick_chapters`, `pick_bool`, `pick_text`, `pick_int`) propagate it and each submenu catches it, so
a flow aborts without half-executing (e.g. EOF at the chapter pick never reads as "all" and never
reaches `cmd_run`); `pause` swallows it. `run_menu`'s top level: "0", empty input, "q" and EOF all
return 0.

## Tests

- `uv run --no-sync pytest tests/unit/test_menu.py tests/unit/test_cli.py -q` — 31 passed.
- `uv run --no-sync pytest -m "not gpu"` — 3923 passed, 26 deselected (gpu), 1 xfailed
  (pre-existing), in ~85 s.
- `uv run --no-sync ruff format . && uv run --no-sync ruff check .` — clean (376 files unchanged).
- `uv run --no-sync pyright` — 0 errors, 0 warnings.
- Manual wiring check (the card's three cases): `omniscan --help` prints the normal Typer help;
  `omniscan version` → `0.1.0`; bare `omniscan < /dev/null` prints the main menu once and exits 0
  (EOF at the prompt = clean exit).

Environment note: plain `uv run …` (with sync) currently fails on this machine with
`failed to remove file …\.venv\Scripts\omniscan.exe: … os error 32` — another process is holding
that exe (not this card's changes; the lock predates them). All runs above used
`uv run --no-sync`; the venv's editable install reflects current source (the new `omniscan.menu`
imports and passes). Once the holder exits, normal `uv run` will re-sync fine.

GPU-marked tests were deselected (no GPU code is touched by this card, and running them would stack
GPU work on this machine).

## Deviations

1. Top-level "0" line: `pick_from` prints its fixed back line `0. Back`, so the main menu shows
   "0. Back" where the card's sketch shows "0. Exit". `pick_from`'s signature is fixed by the card,
   so the label is not parameterisable; behaviour is identical (0 = leave the menu, returns 0).
   If the owner wants the exact wording, MENU2 could give `pick_from` an optional `back_label`.
2. EOF at `pick_bool`/`pick_text`/`pick_int` propagates (the flow aborts, submenu redraws) instead
   of returning the default: the card's Definitions say EOF propagates as a normal back/cancel and
   gives the `pick_chapters`-EOF-aborts example, so EOF must be distinguishable from the empty
   answer (which means default/all there). `pick_from` (whose spec fixes EOF -> None) and `pause`
   are the exceptions. Untested paths either way; the safe reading (a dead input stream never
   triggers a full pipeline run) is the one implemented.
3. Ctrl+C is caught for every command action via `_run_action`, not only around `cmd_serve` (the
   card mandates the serve case; the general case keeps one Ctrl+C during a long `cmd_run` from
   killing the whole menu process).
4. Match validation message: `match: <path> does not exist or is not a folder` (the check is
   `exists() and is_dir()` per the card; Import's message wording only covers existence).
5. `test_menu_launch_serve_keyboard_interrupt_returns_to_submenu` drives `"1", "0", "0"` (the card
   sketch says "1" then "0"): the pause after the interrupted serve consumes one line, because the
   card's general rule pauses after every command invocation, serve included.
6. Extra tests beyond the 15 listed (all in the two allowed test files): `q` exits, `allow_back=False`
   makes "0" invalid, `pick_chapters` whitespace/`3,1` order variants, the nothing-to-do branch, the
   EOF-mid-flow abort, and `pick_series` with an existing-but-empty library root.

## Questions

- Is "0. Back" acceptable on the main menu (vs the sketch's "0. Exit"), or should `pick_from` grow
  an optional back-label parameter in MENU2?
- The locked `omniscan.exe` (see Tests): if another session still holds it, plain `uv run` stays
  broken for this worktree until that process exits — worth a check on the machine.
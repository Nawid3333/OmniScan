# MENU2 — Interactive menu, part 2: translate/glossary/watermark/filter/queue

**Owner:** GLM builder · **Branch:** `MENU2` · **Worktree:** `V:\OmniScan-wt\MENU2` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read `src/omniscan/menu.py` **in full** (MENU1, already merged — this card adds
to it, imitating its exact shape: `pick_from`/`pick_series`/`pick_chapters`/`pick_bool`/`pick_text`/`pick_int`/
`pause`/`_run_action`/`run_menu`/`_read_or_quit`, and every existing submenu, especially `menu_library` and its
five leaves as the closest structural template — a submenu with several leaves, each doing a few picks then
one `_run_action`). Then read `tests/unit/test_menu.py` in full for the exact test style (`make_read`,
`counting_read`, `menu_cfg`, `monkeypatch.setattr(menu, "cmd_run", fake_cmd_run)`-style spies). Then read the
exact CLI functions this card wraps (line numbers below are current; if a rebase shifts them, that's fine —
the function names and signatures are what matter): `src/omniscan/cli.py` — `cmd_translate` (565),
`cmd_judge` (646), `story_summarize` (1303), `glossary_list`/`glossary_export`/`glossary_import`/
`glossary_propose` (1168-1296), `cmd_reference` (924), `watermark_add`/`watermark_list`/`watermark_remove`
(1367-1421), `filter_run`/`filter_restore`/`filter_force`/`filter_add` (1020-1130ish), `queue_add`/`queue_list`/
`queue_run`/`queue_pause`/`queue_resume`/`queue_cancel`/`queue_retry`/`queue_clear` (1453-1573), and the
`KNOWN_STAGES`, `STATUSES` constants imported from `omniscan.queue.store` at the top of `cli.py`.

## Goal
Five new top-level menu categories on top of MENU1's exact framework — Translate, Glossary, Watermark, Filter,
Queue — so every remaining CLI area the owner uses regularly (translate/judge/story, glossary maintenance,
fixed-position watermark regions, the promo filter's manual overrides, and the job queue) is reachable from the
bare `omniscan` menu, matching the owner's 2026-09-23 instruction that the menu, not raw subcommands, is the
primary interface. Every existing subcommand keeps working unchanged (additive only, same as MENU1).

## Files you may create / modify
- `src/omniscan/menu.py` (extend — new imports, one new `pick_float` helper, five new submenu functions plus
  their leaves, `TOP_LEVEL_OPTIONS` and `run_menu`'s `menus` tuple both grow from 5 to 10 entries)
- `tests/unit/test_menu.py` (extend)
- `docs/reports/MENU2.md` (create)
Do not touch `src/omniscan/cli.py`, `src/omniscan/core/**`, or any other module — this card only adds a menu
layer on top of functions that already exist and already work from the command line.

## Interfaces (exact — do not rename or change signatures of anything already in menu.py)

Add one new picker, right after `pick_int` (same file, same style):
```python
def pick_float(prompt: str, default: float, *, read: Callable[[str], str] = input) -> float:
    """Like `pick_int` but parses a float; a non-numeric answer reprints the prompt and asks again."""
```

Extend `TOP_LEVEL_OPTIONS` (a 10-tuple now) by inserting five new entries **after** the existing "Library"
entry and **before** the existing "Models" entry, keeping every existing entry's exact text unchanged:
```python
TOP_LEVEL_OPTIONS: tuple[str, ...] = (
    "Run pipeline (...)",                                    # unchanged
    "Library (...)",                                          # unchanged
    "Translate (translate chapters, judge candidates, summarize story)",
    "Glossary (list, export, import, propose terms, bootstrap from reference chapters)",
    "Watermark (fixed-position regions excluded from translation)",
    "Filter (promo-filter overrides: restore or force-filter a file/slice)",
    "Queue (queue pipeline jobs, drain, pause/resume/cancel/retry)",
    "Models (...)",                                           # unchanged
    "Diagnostics (...)",                                      # unchanged
    "Launch (...)",                                           # unchanged
)
```
`run_menu`'s `menus` tuple gains the five new submenu functions in the exact same position:
```python
menus: tuple[Callable[..., None], ...] = (
    menu_pipeline,
    menu_library,
    menu_translate,
    menu_glossary,
    menu_watermark,
    menu_filter,
    menu_queue,
    menu_models,
    menu_diagnostics,
    menu_launch,
)
```

Five new submenu functions, each `(cfg: Config, *, read: Callable[[str], str] = input) -> None`, each the
exact `while True: pick_from(...); try: ... except EOFError: pass` shape every existing submenu already uses
(a picked-but-cancelled leaf mid-flow redraws the same submenu, never crashes, never falls through to the next
category):

```python
def menu_translate(cfg: Config, *, read: Callable[[str], str] = input) -> None:
    """Translate submenu: translate chapters, judge candidate runs, summarize into story memory."""

def menu_glossary(cfg: Config, *, read: Callable[[str], str] = input) -> None:
    """Glossary submenu: list, export, import, propose terms, bootstrap from reference chapters (D7)."""

def menu_watermark(cfg: Config, *, read: Callable[[str], str] = input) -> None:
    """Watermark submenu: add, list and remove a series' fixed-position watermark regions."""

def menu_filter(cfg: Config, *, read: Callable[[str], str] = input) -> None:
    """Filter submenu: run the promo filter over a series, restore or force-filter one file/slice."""

def menu_queue(cfg: Config, *, read: Callable[[str], str] = input) -> None:
    """Queue submenu: queue pipeline stages, list/drain the queue, pause/resume/cancel/retry a job."""
```

## Leaves (exact — implement each of these private functions, one `_run_action` call per leaf)

**`menu_translate`** — options `("Translate chapters", "Judge candidate runs", "Summarize into story memory")`:
- `_translate_run(cfg, *, read)`: `pick_series`; `pick_chapters(sp, read=read)`, and if it returns `[]`
  (empty series) print `"nothing to do"` and return (same guard `_run_everything` already uses); `pick_text(
  "Profile name(s), comma-separated (blank = every enabled profile)", None, read=read)` split into a list on
  `","` (strip each part, drop empties — the exact pattern `_download_models` already uses) or `None` if blank;
  `pick_bool("Force re-run even if the run file already exists?", False, read=read)`; then `_run_action(
  "translate", lambda: cmd_translate(series=series, chapter=chapters, profile=profiles, force=force),
  read=read)`.
- `_judge_run(cfg, *, read)`: same shape as above but for `cmd_judge` — `pick_text("Run id(s) to judge,
  comma-separated (blank = every run)", None, read=read)` (comma-split -> `run=...`), `pick_bool("Force
  re-run even if final.json already exists?", False, read=read)` -> `force`.
- `_story_summarize(cfg, *, read)`: `pick_series`, `pick_chapters`, `pick_text("Chat model", "gemma4:31b-cloud",
  read=read)`, `pick_bool("Re-summarize chapters that already have a summary?", False, read=read)` ->
  `story_summarize(series=series, chapter=chapters, model=model, force=force, as_json=False)`.

**`menu_glossary`** — options `("List", "Export to YAML", "Import from YAML", "Propose terms from OCR text",
"Bootstrap from reference chapters (D7)")`:
- `_glossary_list(cfg, *, read)`: `pick_series`; `pick_from(["proposed", "locked", "rejected", "all"],
  "Status", read=read)` (`None` from `pick_from` = cancelled, return; index 3 = "all" -> `status=None`,
  otherwise `status=` the matching literal string) -> `_run_action("glossary", lambda:
  glossary_list(series=series, status=status), read=read)`.
- `_glossary_export(cfg, *, read)`: `pick_series` -> `_run_action("glossary", lambda:
  glossary_export(series=series), read=read)`.
- `_glossary_import(cfg, *, read)`: `pick_series`; `pick_from(["merge", "replace"], "Mode", read=read)` ->
  `_run_action("glossary", lambda: glossary_import(series=series, mode=mode), read=read)`.
- `_glossary_propose(cfg, *, read)`: `pick_series`, `pick_chapters`, `pick_text("Chat model",
  "gemma4:31b-cloud", read=read)`, `pick_int("Minimum chapters a term must recur in", 2, read=read)`,
  `pick_bool("Dry run (scan and extract, write nothing)?", False, read=read)` -> `glossary_propose(series=
  series, chapter=chapters, model=model, min_chapters=min_chapters, dry_run=dry_run, as_json=False)`.
- `_reference_bootstrap(cfg, *, read)`: `pick_series`, `pick_int("Distinct reference chapters a pair must
  recur in to lock", 3, read=read)`, `pick_text("Chat model", "gemma4:31b-cloud", read=read)`, `pick_bool(
  "Dry run (match/OCR/extract, write nothing)?", False, read=read)`, `pick_bool("Force re-run OCR stages even
  if up to date?", False, read=read)` -> `cmd_reference(series=series, min_locks=min_locks, model=model,
  dry_run=dry_run, force=force)`. Do **not** pre-check `sp.reference_dir` yourself — `cmd_reference` already
  prints a clear `typer.Exit(2)` error when it's missing, and `_run_action` already surfaces that (same
  "let the wrapped command's own error speak" pattern every other leaf in this file uses; duplicating the
  check here would just be dead code that can drift out of sync with the real one).

**`menu_watermark`** — options `("Add region", "List regions", "Remove region")`:
- `_watermark_add(cfg, *, read)`: `pick_series`; four `pick_float` prompts in order `"Left edge (0-1, fraction
  of page width)"`/`"Top edge (0-1, fraction of page height)"`/`"Right edge (0-1)"`/`"Bottom edge (0-1)"`, each
  with default `0.0` for x0/y0 and `1.0` for x1/y1; `pick_text("Note (optional)", None, read=read)` -> `note`
  -> `_run_action("watermark", lambda: watermark_add(series=series, x0=x0, y0=y0, x1=x1, y1=y1, note=note),
  read=read)`.
- `_watermark_list(cfg, *, read)`: `pick_series` -> `_run_action("watermark", lambda:
  watermark_list(series=series), read=read)`.
- `_watermark_remove(cfg, *, read)`: `pick_series`, `pick_int("Region index to remove", 0, read=read)` ->
  `_run_action("watermark", lambda: watermark_remove(series=series, index=index), read=read)`.

**`menu_filter`** — options `("Run filter over a series", "Add a promo example image", "Restore a filtered
file/slice", "Force-filter a file/slice")`:
- `_filter_run(cfg, *, read)`: `pick_series`, `pick_chapters` -> `_run_action("filter", lambda:
  filter_run(series=series, chapter=chapters, as_json=False), read=read)`.
- `_filter_add(cfg, *, read)`: `pick_series`; `pick_text("Path to the example image", None, read=read)`; if
  blank, return; build `Path(path)` and check `.exists()` yourself first (same explicit check
  `_import_chapters` already does for its own path argument — `filter_add`'s own `exists=True` constraint is
  a Typer-CLI-layer check that does not fire when the plain Python function is called directly), printing
  `f"filter: {path} does not exist"` and returning if missing; `pick_bool("Add to the shared global examples
  instead of this series'?", False, read=read)` -> `global_`; `pick_text("Example name (blank = the file's own
  name)", None, read=read)` -> `name` -> `_run_action("filter", lambda: filter_add(series=series, path=path,
  global_=global_, name=name), read=read)`.
- `_filter_restore(cfg, *, read)` / `_filter_force(cfg, *, read)`: `pick_series`; then — **not**
  `pick_chapters`, which returns `None`/a list for a multi-chapter run — pick exactly **one** chapter the same
  way `_pack_chapters`'s format picker works: `index = pick_from(sp.chapters(), "Chapter", read=read)`, return
  if `None`, `chapter = sp.chapters()[index]`; `pick_from(["file", "slice"], "Target", read=read)` -> `target`;
  `pick_int("Index", 0, read=read)` -> `index` (a different `index` variable than the chapter picker's — name
  them distinctly, e.g. `chapter_index`/`override_index`, to avoid shadowing) -> `_run_action("filter", lambda:
  filter_restore(series=series, chapter=chapter, target=target, index=override_index), read=read)` (or
  `filter_force` for the other leaf). `sp` here is `SeriesPaths.from_config(cfg, series)`, same as every other
  leaf that needs `sp.chapters()`.

**`menu_queue`** — options `("Add a job", "List jobs", "Run the queue (drain)", "Pause/resume/cancel/retry a
job", "Clear finished jobs")`:
- `_queue_add(cfg, *, read)`: `pick_series`; `pick_text("Stage(s), comma-separated (blank = ingest, slice)",
  None, read=read)` comma-split into a list or `None` if blank -> `stage`; `pick_chapters` -> `chapters`;
  `pick_int("Priority (higher runs first)", 0, read=read)` -> `priority`; `pick_int("Max attempts", 2,
  read=read)` -> `max_attempts`; `pick_bool("Re-run stages even if up to date?", False, read=read)` -> `force`
  -> `_run_action("queue", lambda: queue_add(series=series, stage=stage, chapter=chapters, priority=priority,
  max_attempts=max_attempts, force=force), read=read)`. Do not pre-validate the stage names yourself —
  `queue_add` already does and echoes a clear message (same pattern as `_reference_bootstrap` above).
- `_queue_list(cfg, *, read)`: `pick_from([*STATUSES, "all"], "Status", read=read)` (last option -> `status=
  None`, otherwise the matching status string) -> `_run_action("queue", lambda: queue_list(status=status),
  read=read)`. Import `STATUSES` from `omniscan.queue.store` at the top of `menu.py`, alongside the existing
  `omniscan.cli` import block.
- `_queue_run(cfg, *, read)`: `pick_text("Webhook URL (blank = none)", None, read=read)` -> `webhook`;
  `pick_int("Stop after this many jobs (0 = until empty)", 0, read=read)` -> raw int, pass `max_jobs=None if
  raw == 0 else raw` (mirrors `queue_run`'s own `int | None` field: 0 has no real meaning as "stop after zero
  jobs", so it means "no limit" here) -> `_run_action("queue", lambda: queue_run(webhook=webhook,
  max_jobs=max_jobs), read=read)`. This call can block for a while draining a real queue — that is correct,
  expected behaviour (same blocking-until-Ctrl+C shape `_serve` already has in `menu_launch`; `_run_action`
  already catches `KeyboardInterrupt`).
- `_queue_job_action(cfg, *, read)`: `pick_from(["pause", "resume", "cancel", "retry"], "Action", read=read)`;
  `pick_int("Job id", 0, read=read)` -> `job_id`; dispatch to the matching one of `queue_pause`/`queue_resume`/
  `queue_cancel`/`queue_retry` (each takes only `job_id`) via `_run_action("queue", lambda:
  {queue_pause, queue_resume, queue_cancel, queue_retry}[action_index](job_id=job_id), read=read)` — pick
  whichever exact-code shape (an if/elif chain or a tuple indexed by `action_index`) reads most like the rest
  of this file's existing dispatch style (e.g. `menu_models`'s `if index == 0: ... elif ... else:` chain).
- `_queue_clear(cfg, *, read)`: no prompts at all -> `_run_action("queue", queue_clear, read=read)` (same
  no-argument shape as `menu_diagnostics`'s doctor/hardware leaves).

## Imports to add at the top of `menu.py`
Extend the existing `from omniscan.cli import (...)` block (keep it one alphabetically-sorted import, ruff
will re-sort it) with: `cmd_judge`, `cmd_reference`, `cmd_translate`, `filter_add`, `filter_force`,
`filter_restore`, `filter_run`, `glossary_export`, `glossary_import`, `glossary_list`, `glossary_propose`,
`queue_add`, `queue_cancel`, `queue_clear`, `queue_list`, `queue_pause`, `queue_resume`, `queue_retry`,
`queue_run`, `story_summarize`, `watermark_add`, `watermark_list`, `watermark_remove`. Add a new import line
`from omniscan.queue.store import STATUSES`.

## Definitions (no interpretation needed)
- Every new leaf follows the exact `_run_action(label, lambda: wrapped_fn(...), read=read)` shape already used
  everywhere in this file — never call a wrapped CLI function directly without going through `_run_action`
  (it's what makes `typer.Exit`/`KeyboardInterrupt` safe and pauses before redrawing).
- A leaf that needs a series but the library has none (`pick_series` returns `None`) returns immediately,
  same as every existing leaf.
- Comma-separated multi-value text fields (`profile`, `run`, `stage`) use exactly the pattern already in
  `_download_models`/`_remove_models`: `[part.strip() for part in text.split(",") if part.strip()]` on a
  non-blank answer, `None` (meaning "the wrapped function's own default") on a blank one.
- No new validation logic duplicates what the wrapped CLI function already checks and echoes on failure —
  every existing leaf in MENU1 already follows this "let the real command's own error speak" rule (e.g.
  `_import_chapters` doesn't pre-validate the series name either); new leaves must match it, not add their own
  parallel checks that could drift out of sync.

## Acceptance tests (CPU only, no GPU/models — mirror `test_menu.py`'s existing style exactly)
For every one of the ~19 new leaf functions: at least one test using `make_read(...)` to script a full
answer sequence, `monkeypatch.setattr(menu, "<wrapped_fn>", fake)` to capture the call, and an assertion on
the exact kwargs the fake was called with (mirroring how existing MENU1 tests check `cmd_run`/`cmd_import`
call args). In particular:
1. `pick_float`: valid input, blank -> default, non-numeric reprompts then accepts a valid one (same 3-case
   shape as the existing `pick_int` tests).
2. `_glossary_list`/`_queue_list`: picking "all" passes `status=None`; picking a specific option passes that
   exact string.
3. `_translate_run`/`_judge_run`/`_queue_add`: a blank comma-list answer passes `None` for that field; a
   `"a, b ,c"` answer passes `["a", "b", "c"]` (whitespace stripped, matching `_download_models`'s own test
   coverage style for the identical parsing rule).
4. `_filter_restore`/`_filter_force`: confirms exactly one chapter name (not a list) reaches the wrapped
   function, and that a distinct `chapter_index`/`override_index` naming (or equivalent) didn't cross-wire the
   chapter pick with the file/slice index pick — assert both values independently in the fake's captured kwargs.
5. `_queue_run`: a `"0"` answer for "stop after this many jobs" results in `max_jobs=None` reaching
   `queue_run`, not `max_jobs=0`.
6. `_reference_bootstrap`: confirms there is **no** call to `sp.reference_dir` or any pre-check — only
   `cmd_reference` is invoked (a grep-style test is fine: assert the fake was called exactly once, with the
   full kwarg set, and nothing else touched the filesystem before it).
7. `TOP_LEVEL_OPTIONS`/`run_menu`'s `menus` tuple: length is now 10; a full smoke walk (`make_read` picking
   each of the 5 new top-level indices then immediately "0" to back out) reaches each new submenu without
   raising, exercising `run_menu`'s dispatch itself (not just calling each submenu function directly).
8. `uv run pytest -q -m "not gpu"` and any doc-consistency test (grep for one, e.g. `tests/unit/test_docs.py`)
   stay green.

## Out of scope
Any change to `src/omniscan/cli.py` or the wrapped functions themselves (if one of them turns out to need a
different signature to be menu-friendly, stop and write that under Questions rather than editing `cli.py` —
this card is additive-only, same rule as MENU1), a multi-select picker (every multi-value field uses the
existing comma-text convention, not a new UI widget), re-ordering or renaming any of MENU1's existing options
or functions, `docs/USER_GUIDE.md` updates (a follow-up, not required for this card).

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_menu.py -q
uv run pytest -q -m "not gpu"
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
`docs/reports/MENU2.md`: Changes, Tests (commands + results), Deviations, Questions. Commit early and often
(this is a big card — commit after each submenu is done and tested, e.g. `MENU2: WIP menu_translate`,
`MENU2: WIP menu_glossary`, ...), final commit `MENU2: translate/glossary/watermark/filter/queue submenus`.
You have 150 tool calls in total — budget carefully across five submenus; if you run low, finish and commit
whichever submenus are done and complete rather than leaving all five half-done. If anything is unclear: stop,
write the question under Questions, commit what you have, and end.

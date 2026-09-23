# MENU2 — Interactive menu, part 2 (translate/glossary/watermark/filter/queue)

## Changes

- `src/omniscan/menu.py` (extend):
  - New picker `pick_float` (same shape as `pick_int`; non-numeric answers reprompt, blank → default).
  - `TOP_LEVEL_OPTIONS` grows 5 → 10: Translate, Glossary, Watermark, Filter, Queue inserted between
    "Library" and "Models"; every MENU1 entry's text unchanged. `run_menu`'s `menus` tuple grows in the
    same positions, so top-level dispatch is `menu_pipeline, menu_library, menu_translate, menu_glossary,
    menu_watermark, menu_filter, menu_queue, menu_models, menu_diagnostics, menu_launch`.
  - Five new submenus in MENU1's exact `while True: pick_from(...) / try: ... except EOFError: pass`
    shape, each with its leaves as private helpers going through `_run_action`:
    - `menu_translate` → `_translate_run` (cmd_translate), `_judge_run` (cmd_judge),
      `_story_summarize` (story_summarize). Translate/judge carry the same "nothing to do" guard for an
      empty series that `_run_everything` uses; profile/run/stage comma-lists use the
      `_download_models` split pattern (blank → None meaning "the command's own default").
    - `menu_glossary` → `_glossary_list` ("all" → `status=None`), `_glossary_export`,
      `_glossary_import` (merge/replace), `_glossary_propose`, `_reference_bootstrap` (no
      `sp.reference_dir` pre-check — cmd_reference's own `typer.Exit(2)` error speaks through
      `_run_action`).
    - `menu_watermark` → `_watermark_add` (four `pick_float` edges, optional note), `_watermark_list`,
      `_watermark_remove`.
    - `menu_filter` → `_filter_run`, `_filter_add` (explicit `.exists()` check before the call, same as
      `_import_chapters`; `filter_add`'s `exists=True` is a Typer-layer constraint that does not fire on
      a direct Python call), `_filter_restore`/`_filter_force` (exactly one chapter picked the
      `_pack_chapters` way; `chapter_index` and `override_index` named distinctly so the two picks
      cannot cross-wire).
    - `menu_queue` → `_queue_add` (no stage pre-validation — queue_add echoes its own error),
      `_queue_list` (`[*STATUSES, "all"]`, last → `status=None`), `_queue_run` ("0" → `max_jobs=None`,
      mirroring the command's `int | None`), `_queue_job_action` (tuple-indexed dispatch to
      queue_pause/resume/cancel/retry, like `_pack_chapters`'s format map), `_queue_clear` (no prompts).
  - New imports: the 23 wrapped CLI functions the card lists, plus `STATUSES` from
    `omniscan.queue.store`.
- `tests/unit/test_menu.py` (extend): 22 new tests (48 total in the file) covering every new leaf, the
  comma-list split rule, the "all" → None status mapping, the single-chapter restore/force picks,
  `max_jobs=None` on "0", the no-pre-check bootstrap call, `pick_float`'s 3 cases, the exact
  10-entry `TOP_LEVEL_OPTIONS`, and a `run_menu` smoke walk that reaches all five new submenus through
  the real dispatch. New `spy()` helper = `recorder()` for any wrapped function name.

## Tests

- `uv run --no-sync pytest tests/unit/test_menu.py -q` → 48 passed.
- `uv run --no-sync pytest -q -m "not gpu"` → all pass except one pre-existing, unrelated failure:
  `tests/unit/test_gui_library.py::test_demo_script_screenshot_and_unknown_series` fails with
  `ModuleNotFoundError: No module named 'PySide6'` (PySide6 is not installed in this worktree's venv —
  the same reason the other GUI tests are skipped). Not touched by this card.
- `uv run --no-sync ruff format . && uv run --no-sync ruff check .` → clean.
- `uv run --no-sync pyright` → 0 errors in `menu.py` / `test_menu.py`; the 47 remaining errors are all
  pre-existing `PySide6` import-resolution errors in `gui/**` and `tests/gui/**`, unrelated to this card.

## Deviations

- The three "Chat model" prompts need `pick_text(...) or "gemma4:31b-cloud"`: `story_summarize`,
  `glossary_propose` and `cmd_reference` take a concrete `str` (the CLI default), while `pick_text` is
  typed `str | None` — pyright rejects passing `None`. The fallback is the same literal the CLI uses, so
  behaviour is identical (blank → that model). `pick_text`'s signature was not changed.
- WIP commits: `MENU2: WIP menu_translate + menu_glossary` after those two were done and tested;
  watermark/filter/queue landed in the final commit (call-budget choice, the card's "commit early and
  often" is otherwise honoured).

## Questions

1. The "nothing to do" empty-series guard is specified for `_translate_run` and (by "same shape")
   `_judge_run`, but `_story_summarize`, `_glossary_propose`, `_filter_run` and `_queue_add` currently
   pass `[]` straight through, so the wrapped command's own "no chapters found" error speaks (surfaced
   as `(story: exit 2)` etc.). Should the guard be extended to those four leaves for a consistent UX?
   Implemented exactly as the card words it for now.
2. The four queue leaves that don't need a series (`_queue_list`, `_queue_run`, `_queue_job_action`,
   `_queue_clear`) take `cfg` per the card and never use it — kept for signature uniformity; fine?
# C7b — Typeset stage: `omniscan typeset`

## Changes

New files:

- `src/omniscan/typeset/plan.py` — `luminance`, `ellipse_polygon`, `target_box`, `plan_layout`
  (target box / font role / colour / stroke decisions on top of the C7a `layout_region` fitter).
- `src/omniscan/typeset/stage.py` — `TypesetStage` (`ocr.json` + `final.json` + `inpaint.json` →
  `layout.json`; no GPU group; metrics `items` / `overflow` / `skipped`).
- `tests/unit/test_typeset_plan.py`, `tests/unit/test_typeset_stage.py`.
- `docs/reports/C7b.md` (this file).

Modified:

- `src/omniscan/cli.py` — the `typeset` stub is replaced by the real command (`series`,
  `--chapter/-c`, `--force`; runs only `[TypesetStage()]` through `_run_stages`); `typeset` removed
  from `_STUB_COMMANDS`.
- `tests/unit/test_cli.py` — `typeset` left the stub list; added `test_cli_typeset` (runs only the
  typeset stage per manifest, `--chapter`/`--force`, unknown series → exit 2) and
  `test_cli_slice_unaffected`.
- `README.md` — typeset status row → `working`; dropped from the "Not implemented yet" table.
- `docs/USER_GUIDE.md` — `### omniscan typeset` subsection, the `layout.json` row of the folder
  table, the seven `[typeset]` keys in the config table, `typeset` removed from the stubs paragraph.

## Tests

All commands run in the C7b worktree (Python 3.14, `uv run --no-sync`; the venv was locked by another
session so plain `uv run` could not re-sync — no dependency changes were needed, so this is equivalent).

- `uv run pytest tests/unit/test_typeset_plan.py tests/unit/test_typeset_stage.py tests/unit/test_cli.py tests/unit/test_docs.py`
  → 42 passed.
- `uv run pytest` → **2359 passed**, 3 warnings (the pre-existing harmless NumPy read-only warnings).
- `uv run ruff format . && uv run ruff check .` → clean.
- `uv run pyright` → 0 errors, 0 warnings.

Card acceptance points 1–8 covered: luminance/ellipse-polygon math, `target_box` (ellipse box
`BBox(66,36,334,164)` verified against the real `inscribed_box` before hard-coding; strictly-larger
text box wins; equal area → ellipse box; `free_grow` growth with origin clamping), the
role/colour/stroke table (fake font), skip/order/font-name/overflow behaviour, a real-font bubble
layout inside the inscribed ellipse box, stage resumability + invalidation (config `max_px` or any of
the three inputs) + missing-input failure messages + metrics, and the CLI wiring.

## Deviations

- One test detail: the card's overflow example ("a text too long for its box yields `overflow=True`
  at `size_px == cfg.min_px`") needed a small bubble (100×60) rather than the card's 400×200 bubble —
  with the fake font the card's long sentence actually fits the 400×200 ellipse box at size 22
  (no overflow). The assertion is unchanged; only the fixture size differs.
- `skipped` for a chapter where every translatable region has no final line is 3 (not 0): the metric
  counts translatable regions without a usable text, which the card's definition implies.

## Questions

None — the card was unambiguous. (For a later card, note the queue's `STAGE_TABLE` in
`src/omniscan/queue/executor.py` still lacks `typeset`, so a queued `typeset` job fails permanently
with "not implemented yet"; adding it there looked like scope beyond this card's file list.)
## Review addendum (director)
Rebase conflicts (README status/stub tables, USER_GUIDE tables and command sections, stub tuple in `cli.py`/`test_cli.py`) resolved: `detect`, `judge`, `inpaint` and `typeset` are real commands now.
Mutation check, 16 mutants over `plan.py` (luminance weights, ellipse point count and sin/cos, `>`→`>=`, inverted comparison, margin ignored, x0/y0 clamp, wrong axis for the grow, colour threshold boundary, no-fill default, `text_color`/`stroke_color`
overrides ignored, sfx stroke, whitespace-only text, free-text role): **all 16 killed**. The queue's `STAGE_TABLE` gets every stage with the `omniscan run` card (R1) — nothing to do here.

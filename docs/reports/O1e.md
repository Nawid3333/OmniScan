# O1e — report

## STOPPED: the card's fix is incomplete — a second, by-design merge in `core/**` defeats it

Per CLAUDE.md ("If the spec is ambiguous or seems wrong: STOP, write the question, commit, and end"),
I implemented the runner-side half of the card, then proved — empirically, with a stage spy — that the
**crash persists** because the series' `series.toml` reaches the stages through a second merge the card
did not account for, and that second merge lives in `src/omniscan/core/stage.py`, which builders must
not edit. Nothing below claims the bug is fixed.

### The full clobber chain (root cause corrected)

The card identified one of **two** merge sites. Both are by design — `cli.py:250`, `cli.py:788` and
`queue/executor.py:24` say so explicitly ("models to load, and make_context() below would otherwise
merge it too late for that choice"):

1. **`run_pipeline`'s merge** (`pipeline/runner.py:277`) — exists so the *model-loading* decision
   (`build_vram_manager`) sees the series' overrides. The card's `merge_series_config` flag targets this one.
2. **`make_context`'s merge** (`core/stage.py:100`) — added 2026-09-20 in 9fa69ce together with the whole
   per-series-override feature. `run_series` (`core/stage.py:227`) calls `make_context` **per chapter**, and
   it unconditionally does `cfg=series_config(cfg, sp.library_dir)` when building every `ChapterContext`.
   Its docstring states the intent: "the series' `series.toml` overrides are already applied to `ctx.cfg`".

The OCR dispatch reads the second one, not the first: `OcrStage.run` does `cfg = ctx.cfg`
(`ocr/stage.py:63`) and dispatches on `cfg.ocr.engine` (`ocr/stage.py:64`), taking the `models["reader"]`
branch for anything that is not `"ppocr"`. The models dict, however, comes from the VRAM manager built
from the *candidate* config (`gpu/groups.py:40-55`: `ppocr` → `line_detector`+`recognizer`, others → `reader`).

So even with `run_pipeline(..., merge_series_config=False)`, for `PepperCarrotJA`:

- `candidate_config` produces the correct `pair_cfg` (`ocr.engine == "ppocr"`) → manager loads ppocr models;
- `run_pipeline` skips **its** merge;
- `run_series` → `make_context` re-merges `PepperCarrotJA/series.toml` into `ctx.cfg` anyway
  → `ctx.cfg.ocr.engine == "paddleocr_vl"` at stage time;
- `OcrStage.run` takes the else branch → `models["reader"]` → **`KeyError: 'reader'`, unchanged**.

Empirical proof (not just code reading): `tests/unit/test_pipeline_runner.py::test_a_stage_still_sees_the_series_toml_when_run_pipeline_skips_its_merge`
runs a real `run_pipeline(..., merge_series_config=False)` over a temp series with `series.toml`
`[detect] threshold = 0.9` and a spy stage records what the stage actually received: **0.9**, i.e. the
series' value, merged by `make_context` — the exact class of clobber the card set out to remove.

Corollary: with today's core code the new flag is behaviourally a **no-op for stages** (and for the
qualification run as a whole); it only changes `run_pipeline`-local details (`build_stage`'s config, the
prefetch pass), none of which affect the OCR dispatch. The `manga_ocr` candidates on JA are likewise still
measured dishonestly: `ctx.cfg.ocr.engine` is the series' `"paddleocr_vl"` while the loaded reader is
MangaOcr (`engine_rec_model` labels the metrics with the wrong engine).

### The needed core-side change (director-owned, `core/**`)

Thread the flag through so the qualification path can skip **both** merges:

- `core/stage.py`: `make_context(cfg, series, chapter, gpu, *, merge_series_config: bool = True)` —
  `cfg=series_config(cfg, sp.library_dir) if merge_series_config else cfg`; same keyword (default `True`)
  on `run_series`, passed through to `make_context` (`run_series` is the only production caller besides
  `cli.py:352`, which keeps the default).
- `pipeline/runner.py` (already done in the WIP commit): `run_pipeline` forwards its
  `merge_series_config` to every `run_series` call (auto loop, step-mode preview loop).
- `eval/qualify.py` (already done): `candidate_config` merges the series' settings then applies the
  candidate on top; `default_run_candidate` passes `merge_series_config=False`.

Callers that keep the default are byte-for-byte unchanged: merging is idempotent, so `cli.py`, the queue
executor and `glossary/reference.py` (which merge once themselves before calling) see identical `ctx.cfg`
either way — but verify that when the core card runs (e.g. `test_reference_cli.py` already asserts the
reference-side merge trap).

## Changes

- `src/omniscan/pipeline/runner.py` — `run_pipeline` gains keyword-only `merge_series_config: bool = True`
  (docstring explains the contract); when `False` it skips its internal `series_config` line. Nothing else changed.
- `src/omniscan/eval/qualify.py` — `candidate_config` now merges the series' own `series.toml` first
  (deep-copied; `SeriesConfigError` propagates), then applies the candidate's
  `ocr.engine`/`det_model`/`rec_model`/`lang` last, so the candidate is the final word on the config the
  qualification intends to measure; `default_run_candidate` passes `merge_series_config=False`. Docstrings
  state both, plus the open core-side gap.
- `tests/unit/test_pipeline_runner.py` — `ConfigSpyStage` double + 3 tests (default merge end-to-end,
  run_pipeline's own merge contract, the make_context gap pin).
- `tests/unit/test_eval_qualify.py` — `write_series_toml` helper; series-then-candidate pin for
  `candidate_config` (input config untouched); `SeriesConfigError` propagation; `merge_series_config=False`
  assertion added to `test_default_run_candidate_measures_one_chapter`; the two-layer regression pin
  `test_o1e_regression_the_stage_still_sees_the_series_engine_until_the_core_fix`.
- `docs/reports/O1e.md` — this file.
- Not touched (per card): `core/**`, `ocr/stage.py`, `ocr/crop_readers.py`, `config/qualification.toml`, `data/**`.
- Grep check: the other `run_pipeline` call sites — `cli.py:853`, `queue/executor.py:52`,
  `glossary/reference.py:466` — pass no `merge_series_config` and needed no changes (default `True`).

## Tests

Commands and results (all in the O1e worktree):

```
uv run pytest tests/unit/test_pipeline_runner.py tests/unit/test_eval_qualify.py -q
  → 65 passed
uv run pytest tests/unit/test_pipeline_runner.py tests/unit/test_eval_qualify.py tests/unit/test_docs.py -q
  → 74 passed
uv run pytest -m "not gpu"
  → 3744 passed, 25 deselected, 1 xfailed (pre-existing), 3 warnings in 77.84s
uv run ruff format . && uv run ruff check .
  → 2 files reformatted (the two test files); all checks passed
uv run pyright
  → 0 errors, 0 warnings, 0 informations
```

### What each new/changed test proves — and what it does not

1. `test_run_pipeline_merges_the_series_toml_by_default` (runner) — acceptance 1: with the default (no
   argument), a temp `library/S/series.toml` `[detect] threshold = 0.9` reaches a stage spy as `0.9`.
   End-to-end through the real `run_series`/`make_context`. Proves the default path is unchanged.
2. `test_run_pipeline_merge_series_config_false_skips_its_own_merge` (runner) — acceptance 2, **adapted**:
   pins that `run_pipeline` itself calls `series_config` exactly once by default and not at all with
   `merge_series_config=False` (via a spy wrapped around `runner_module.series_config`; `make_context`
   imports it from `core.config` directly, so only `run_pipeline`'s call is counted). It deliberately does
   NOT assert stage-visible config here — that would be false (see 3).
3. `test_a_stage_still_sees_the_series_toml_when_run_pipeline_skips_its_merge` (runner) — the gap pin:
   with `merge_series_config=False` a stage still receives the series' `0.9`, because `make_context`
   re-merges. **This test pins current (broken-for-qualification) behaviour on purpose**; the core-side fix
   must flip its assertion to `[0.3]`.
4. `test_candidate_config_merges_the_series_toml_then_applies_the_candidate` (qualify) — acceptance 3,
   both pins in one test: series.toml `[ocr] engine = "paddleocr_vl"` + `[detect] threshold = 0.9`, a
   `ppocr` candidate → `out.ocr.engine == "ppocr"` (candidate wins) **and** `out.detect.threshold == 0.9`
   (series' other sections apply); also `det_model`/`rec_model`/`lang`, `work_root`, and that the input
   config is untouched (`manga_ocr`/`0.3` preserved). A series with no `series.toml` is covered unmodified
   by the existing `test_candidate_config_applies_the_candidate_onto_a_copy`.
5. `test_candidate_config_propagates_a_broken_series_toml` (qualify) — invalid TOML raises
   `SeriesConfigError` out of `candidate_config` (per card: a real config error, not swallowed).
6. `test_default_run_candidate_measures_one_chapter` (qualify, extended) — acceptance 4: the mocked
   `run_pipeline` now requires and records `merge_series_config` (no default, so a missing kwarg fails the
   test loudly) and asserts it is `False`. The rest of the test is unchanged.
7. `test_o1e_regression_the_stage_still_sees_the_series_engine_until_the_core_fix` (qualify) — acceptance 5,
   **honest form**. It exercises the real chain end-to-end with real `Config`/`SeriesPaths`/`run_pipeline`:
   a temp `library/PepperCarrotJA/series.toml` with `[ocr] engine = "paddleocr_vl"`, a `ppocr` candidate,
   `candidate_config` → `run_pipeline(..., merge_series_config=False)` with a spy `ocr` stage wired through
   the real `run_series`/`make_context`. It asserts both layers: `pair_cfg.ocr.engine == "ppocr"` (holds
   already) and — currently — `seen == ["paddleocr_vl"]` at stage level.
   - **Proves**: `candidate_config` orders the merges correctly (series first, candidate last), the flag
     reaches `run_pipeline`, and the full config plumbing up to the stage boundary works.
   - **Does not prove**: that the bug is fixed. It cannot, because it isn't — the stage-level assertion
     documents the residual `make_context` clobber. When the core-side fix lands, flipping that one
     assertion to `["ppocr"]` turns this into the regression pin the card asked for; the test then fails
     again if any layer ever re-introduces the clobber (which is the property acceptance 5 wanted).

## Deviations

1. **Stopped short of the card's goal; the fix is only half-applied.** The runner-side and eval-side halves
   are implemented and green, but the live `KeyError: 'reader'` on `PepperCarrotJA` will persist until the
   core-side change described above lands. Re-running the qualification suite now would still show 100 %
   failures for the ppocr candidates on `ja` — do not re-run it expecting different numbers yet.
2. **Acceptance test 2 as written is unachievable** without editing `core/**`: "the stages see the config
   exactly as passed in" is false under the real `run_series`/`make_context`. I split it into the two tests
   described above (run_pipeline's own contract + the gap pin) rather than fake a pass by stubbing out
   `run_series` itself, which would have tested nothing.
3. **Acceptance test 5's stage-level expectation inverted**, honestly: it pins the residual gap and names
   the flip. Rationale in the test docstring and above.
4. **Final commit message differs** from the card's prescribed `O1e: fix qualification suite candidate
   config clobbered by series.toml` — that message would be false. The final commit is
   `O1e: STOPPED — make_context (core) re-merges series.toml per chapter; runner-side half only`.

## Questions

1. **Q1 (blocks the actual fix): approve the core-side change?** Thread `merge_series_config: bool = True`
   through `run_series` → `make_context` (`core/stage.py`), forwarded from `run_pipeline`. Default `True`
   keeps every existing caller identical (merging is idempotent for them). Want this as a new card (it
   touches `core/**`, so it is yours), as an amended O1e, or shall the runner-side half be reverted until
   then? The two gap-pinning tests and `docs/reports/O1e.md` mark exactly which assertion flips when it lands.
2. **Q2: keep the runner-side half in the meantime?** The flag is a no-op for stage-visible behaviour until
   Q1 lands (harmless, contract already pinned by tests); `candidate_config`'s series-then-candidate order
   is correct today and required by the eventual fix. I recommend keeping both; say the word and I revert.
3. **Q3: `manga_ocr` candidates on JA are mis-measured even after the ppocr crash is fixed** — if a series
   forces engine X, any candidate of engine Y≠X cannot be measured honestly on that series at all; the
   plan-level alternative is excluding such (candidate, dataset) pairs (or per-language truth series without
   `[ocr]` overrides). Not needed for this fix; flagging for the qualification-plan design.

## Review addendum (director)

Applied the core-side change from Q1 directly (approved) rather than spinning up a separate card — it is
exactly the shape this report sketched, small, and this fix has already blocked two live qualification runs
tonight:

- `core/stage.py`: `make_context` gains `merge_series_config: bool = True` (keyword-only, after `gpu`);
  `cfg=series_config(cfg, sp.library_dir) if merge_series_config else cfg`. `run_series` gains the same
  keyword, forwarded into its per-chapter `make_context` call. Every other field/behaviour unchanged.
- `pipeline/runner.py`: both `run_series` call sites (the step-mode preview loop and the main pass loop)
  and `_run_preview` itself now thread `merge_series_config` through from `run_pipeline`'s own parameter
  (already added in the WIP commit). `cli.py:352`'s direct `make_context(cfg, series, chapter)` call needs
  no change — the new keyword defaults to `True`.
- Flipped the two gap-pinning tests to their fixed-behaviour assertions, exactly where each one's own
  docstring said to: `test_a_stage_still_sees_the_series_toml_when_run_pipeline_skips_its_merge` →
  renamed `test_run_pipeline_merge_series_config_false_is_honored_by_stages_too`, now asserts
  `spy.seen == [cfg.detect.threshold]` (not the series' `0.9`); `test_o1e_regression_the_stage_still_sees_the_series_engine_until_the_core_fix`
  → renamed `..._the_stage_sees_the_candidate_engine_not_the_series_default`, now asserts
  `seen == ["ppocr"]` (not `["paddleocr_vl"]`) — this is the exact live `KeyError: 'reader'` reproduction
  from the real qualification run, now passing.
- **Re-verified** (worktree, after the core change): targeted files (`test_pipeline_runner.py` +
  `test_eval_qualify.py` + `test_docs.py`) all pass; full `pytest -q -m "not gpu"` all pass (no new
  failures, same pre-existing xfail); `ruff format`/`ruff check` clean; `pyright` 0 errors.
- **Answers**: Q1 — approved and applied as above. Q2 — moot, both halves now do real work together. Q3 —
  actually resolved as a side effect: since the candidate's engine now survives all the way to the stage
  regardless of the series' own `series.toml`, a `manga_ocr`/`paddleocr_vl` candidate on a series that
  forces a different engine is now measured with *its own* engine too, not the series' default. No
  remaining known gap for this bug.
- Grep-audited every `run_pipeline` call site in the repo for the same clobber risk (`cli.py`,
  `queue/executor.py`, `glossary/reference.py`, `eval/qualify.py`): only `eval/qualify.py` deliberately
  overrides a `SERIES_SECTIONS` field the way this bug needs; the other three just want the series' normal
  config and are unaffected by (and don't need) the new flag.
- Final commit message corrected to the card's original: `O1e: fix qualification suite candidate config
  clobbered by series.toml` (the STOPPED commit's message no longer describes the state of this branch).
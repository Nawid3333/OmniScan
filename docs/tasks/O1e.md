# O1e — Fix: a series' own `series.toml` silently overrides the OCR qualification suite's candidate

**Owner:** GLM builder · **Branch:** `O1e` · **Worktree:** `V:\OmniScan-wt\O1e` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read `src/omniscan/eval/qualify.py` (`candidate_config`, `default_run_candidate`,
`run_qualification`), `src/omniscan/pipeline/runner.py` (`run_pipeline`, and how it calls `series_config`),
`src/omniscan/core/config.py` (`series_config`, `SERIES_SECTIONS` — read-only, do not modify), `src/omniscan/ocr/stage.py`
(`OcrStage.run` — the `if cfg.ocr.engine == "ppocr":` branch that crashes), and
`src/omniscan/glossary/reference.py`'s `run_reference` (the code comment above its `series_config` call: a previous
card, GL1, hit the same family of bug once already — read it for context, it is not the same fix but the same trap).

## Why (root-caused by the director, reproduced live on real hardware)
`config/qualification.toml` lists `paddleocr-vl-1.6` etc. as candidates for the `ja` dataset (`PepperCarrotJA`).
`data/raws/PepperCarrotJA/series.toml` (a real, legitimate production file) contains:
```toml
[ocr]
engine = "paddleocr_vl"
```
`candidate_config()` builds a `Config` copy with `ocr.engine` set to the **candidate under test** (e.g. `"ppocr"` for
`ppocr-v5-server-multi`). `default_run_candidate()` then calls `build_vram_manager(cfg)` — which correctly loads a
`line_detector`/`recognizer` pair for the `"ppocr"` engine — and passes that same `cfg` into `run_pipeline(cfg, ...)`.
But `run_pipeline` (`pipeline/runner.py`) unconditionally does `cfg = series_config(cfg, <library_dir>)` as its very
first line, which re-merges `PepperCarrotJA/series.toml` **on top of** the candidate's config — silently flipping
`ctx.cfg.ocr.engine` back to `"paddleocr_vl"` for the actual pipeline run, while the models that were already loaded
into VRAM (via the untouched candidate cfg) are still the `"ppocr"` ones. `ocr/stage.py`'s dispatch
(`if cfg.ocr.engine == "ppocr": ... else: ... models["reader"]`) then takes the wrong branch and crashes:
```
KeyError: 'reader'
```
Reproduced live: every `ppocr`-engine candidate (`ppocr-v5-server-multi`, `ppocr-v6-tiny/small/medium`) fails on
100% of `PepperCarrotJA` chapters this way, while the same candidates succeed fine on `PepperCarrotKR`/`PepperCarrotCN`
(whose `series.toml` files don't touch `[ocr]`). `manga-ocr-2025`/`manga-ocr-base`/`paddleocr-vl-1.6` candidates on
`ja` don't crash — but only by coincidence: `"manga_ocr"`/`"paddleocr_vl"` and the series' forced `"paddleocr_vl"`
all take the *same* `else`/`models["reader"]` branch, so the crash doesn't happen, though the qualification suite is
still not honestly testing what it thinks it's testing whenever a series carries its own `[ocr]` override.
**Out of scope, do not touch:** `PaddleOcrVlReader`'s per-crop sequential `generate()` calls (`ocr/crop_readers.py`) —
that is a separate, already-documented, deliberately-deferred performance limitation ("batched generation is a
later card"), not a bug, and has nothing to do with this fix.

## The fix (design decided by the director — implement this shape, not a different one)
1. **`src/omniscan/pipeline/runner.py`** — add one new keyword-only parameter to `run_pipeline`:
   `merge_series_config: bool = True`. When `True` (the default — every existing call site keeps this, so their
   behaviour is byte-for-byte unchanged), `run_pipeline` does exactly what it does today. When `False`, it **skips**
   its internal `cfg = series_config(cfg, SeriesPaths.from_config(cfg, series).library_dir)` line entirely and uses
   the `cfg` it was given as-is (the caller is asserting it already prepared the config correctly). Nothing else in
   `run_pipeline` changes.
2. **`src/omniscan/eval/qualify.py::candidate_config`** — currently builds its `Config` copy straight from the `cfg`
   it's given. Change it to **first** apply the series' own `series.toml` (so the candidate still gets that series'
   legitimate non-OCR settings — e.g. a custom `slicer`/`detect` section — exactly like a normal `omniscan run` on
   that series would), and **then** apply the candidate's `ocr.engine`/`det_model`/`rec_model`/`lang` overrides on
   top, so they are the last word: `out = series_config(cfg, SeriesPaths.from_config(cfg, dataset.series).library_dir)`
   (deep-copied, as today), then set `out.paths.work_root`, `out.ocr.engine`, `out.ocr.det_model`, `out.ocr.rec_model`,
   `out.ocr.lang` exactly as it does now. (If `series_config` raises `SeriesConfigError` for a genuinely broken
   `series.toml`, let it propagate — that's a real config error, not something to swallow here.)
3. **`src/omniscan/eval/qualify.py::default_run_candidate`** — its `run_pipeline(...)` call gains
   `merge_series_config=False` (the config was already fully prepared by `candidate_config` in step 2, including any
   legitimate series-level settings — `run_pipeline` must not re-merge and re-clobber the OCR override).
4. Result: every candidate is now tested with **exactly the OCR settings the qualification plan says it should be**,
   on top of that series' other real settings, regardless of what that series' own `series.toml` happens to say about
   `[ocr]`. `ppocr-v5-server-multi`/`ppocr-v6-*` should now succeed on `PepperCarrotJA` the same way they already do
   on the other two series (same style of real numbers, not an error row).

## Files you may create / modify
- `src/omniscan/pipeline/runner.py` (the one new parameter, and the one `if` around the existing `series_config` line
  — nothing else in this file changes)
- `src/omniscan/eval/qualify.py` (`candidate_config`, `default_run_candidate` — as described; nothing else)
- `tests/unit/test_pipeline_runner.py` (extend — the new parameter)
- `tests/unit/test_eval_qualify.py` (extend — `candidate_config` picks up series.toml then applies the candidate on
  top; `default_run_candidate` passes `merge_series_config=False`)
- `docs/reports/O1e.md` (create)
Do not modify `src/omniscan/core/**` (in particular `series_config`/`SERIES_SECTIONS` themselves stay untouched —
this fix works entirely by choosing *when* that existing function runs, not by changing what it does), `ocr/stage.py`,
`ocr/crop_readers.py`, or `config/qualification.toml`/any `data/**` file.

## Acceptance tests (CPU only; temp dirs; no GPU, no real models, no `data/`)
1. **`run_pipeline(..., merge_series_config=True)` (the default, and calling with no argument at all)**: unchanged —
   an existing test (or a new one if none currently covers a `series.toml` override end-to-end) shows a temp
   `library_root/<series>/series.toml` with e.g. `[detect]\nthreshold = 0.9` actually takes effect on the config the
   stages see.
2. **`run_pipeline(..., merge_series_config=False)`**: the *same* `series.toml` from case 1 does **not** take effect
   — the stages see the config exactly as passed in, unmodified. Assert this by inspecting what a stage actually
   received (e.g. via a stage double/spy the way existing `pipeline/runner` tests already do, or by asserting on
   `ctx.cfg` if a test hook already exposes it — follow this test file's existing pattern, don't invent a new one).
3. **`candidate_config`**: given a temp series with a `series.toml` containing `[ocr]\nengine = "paddleocr_vl"` (and,
   to prove non-OCR sections still apply, also `[detect]\nthreshold = 0.9` or similar), `candidate_config(cfg, cand,
   dataset, work_root)` for a candidate whose `engine == "ppocr"` returns a config whose `ocr.engine == "ppocr"`
   (the candidate wins) **and** whose `detect.threshold == 0.9` (the series' other settings still apply) — pin both
   in one test. A series with no `series.toml` at all behaves exactly as today (existing test must still pass
   unmodified).
4. **`default_run_candidate`**: mock `run_pipeline` and assert it is called with `merge_series_config=False` (add
   this assertion to the existing `test_default_run_candidate_measures_one_chapter`-style test rather than writing a
   whole new one, unless that makes the existing test unwieldy — your call, note it in Deviations either way).
5. **Regression pin for the exact bug**: a small end-to-end-ish test (real `Config`/`SeriesPaths`, a temp
   `library_root/<series>/series.toml` with `[ocr]\nengine = "paddleocr_vl"`, a candidate with `engine="ppocr"`) that
   builds the config the real call chain would produce (`candidate_config` then `run_pipeline(...,
   merge_series_config=False)` with a stubbed/fake stage runner, OR — simpler and just as convincing — directly
   assert `candidate_config(...).ocr.engine == "ppocr"` where before this fix it would have been silently correct at
   the `candidate_config` level too (the bug only manifested one layer deeper, inside `run_pipeline`'s *own* second
   merge) — make sure the test actually exercises the `run_pipeline(merge_series_config=False)` path end-to-end
   enough that it would have caught the original `KeyError: 'reader'` class of bug, not just `candidate_config` in
   isolation. Explain in the report exactly what this test does and does not prove.
6. `uv run pytest -q -m "not gpu"` and `tests/unit/test_docs.py` stay green; no change to any other `run_pipeline`
   call site's behavior (`cli.py`, `queue/executor.py`, `glossary/reference.py` all keep the default `True` and are
   therefore untouched — a quick grep confirming none of them needed changes is worth a line in the report).

## Out of scope
`PaddleOcrVlReader`'s sequential-per-crop speed (a separate, already-known, deliberately-deferred limitation — do
not touch `ocr/crop_readers.py`), actually re-running the qualification suite for real (the director does that after
this merges), changing `config/qualification.toml` or any `series.toml`, changing `series_config`/`SERIES_SECTIONS`
themselves, any other `run_pipeline` call site's default behaviour.

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_pipeline_runner.py tests/unit/test_eval_qualify.py tests/unit/test_docs.py -q
uv run pytest -q -m "not gpu"
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
`docs/reports/O1e.md`: Changes, Tests (commands + results), Deviations, Questions. **Commit early**
(`O1e: WIP run_pipeline merge_series_config flag` once step 1 and its test pass); the final commit is
`O1e: fix qualification suite candidate config clobbered by series.toml`. You have 150 tool calls in total. If
anything is unclear: stop, write the question under Questions, commit, and end.

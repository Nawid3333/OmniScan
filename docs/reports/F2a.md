# F2a — Promo filter wired into the pipeline (tier 2: file-level in `ingest`, slice-level in `slice`)

Branch `F2a`, worktree `V:\OmniScan-wt\F2a`. Three commits: `F2a: WIP overrides + ingest filter`,
`F2a: stage-level filter tests, CLI --json progress flag, web index-gap regression`,
`F2a: promo filter wired into ingest and slice`.

## Changes

New files

- `src/omniscan/filter/apply.py` — the pipeline side of the filter:
  `Overrides` (frozen dataclass: `files_restored/files_forced/slices_restored/slices_forced`
  as `frozenset[int]`), `EMPTY_OVERRIDES`, `load_overrides` (manual `filter.json` entries only,
  last entry per `(target, index)` wins, missing/corrupt/invalid file → empty),
  `file_verdict` / `slice_verdict` (forced → filtered, restored → keep, else
  `score >= threshold and matched is not None`), `examples_fingerprint` (sha256 of
  `"name:hash"` lines), `example_files` (sorted jpg/jpeg/png directly under `global/` and
  `<series>/`), `record_override` (appends a manual decision, **creating `filter.json` when
  absent** — the CLI's create-if-missing behaviour, since `decide.restore` must keep raising
  `FileNotFoundError` for its existing test), and `apply_slice_filter` (dHash per non-blank
  slice via `dhash_tensor`, `Slice.model_copy(update={"filtered": True})`, each newly filtered
  slice's crop saved as a quality-95 JPEG to `filtered_dir/{chapter}_slice_{index:04d}.jpg`).
- `dhash_tensor(image, hash_size=8)` in `src/omniscan/filter/hashing.py` — ITU-R 601 luma
  (0.299/0.587/0.114), bilinear **antialiased** resize to `(hash_size, hash_size + 1)`, the same
  MSB-first left>right packing as `dhash`, one host transfer of `hash_size*(hash_size+1)` values.
  Within Hamming ≤ 6 of `dhash(PIL)` on structured content (see Tests).
- `src/omniscan/filter/__init__.py` re-exports the apply-side names next to decide/hashing.
- Tests: `tests/unit/test_filter_apply.py`, `test_ingest_filter.py`,
  `test_slicer_filter.py`, `test_filter_cli.py` (new); `test_filter_hashing.py` (extended).

Modified files

- `src/omniscan/ingest/__init__.py` — `ingest_chapter(..., *, examples=(), threshold=0.90,
  overrides=None, filtered_dir=None)`: each raw file's dHash (of the raw original via
  `Image.open`) is matched with `best_match`, then `file_verdict` decides. Filtered files are not
  converted/laid out/listed; names go to `IngestArtifact.filtered_files` (raw order) and each is
  copied byte-for-byte to `filtered_dir/<name>` (existing identical copy left alone, source never
  moved). Kept files keep their **original raw index** (`SourceFile.index`, gaps allowed; cache
  names use it). No survivor → `ValueError("every image of <raw_dir> matched a promo example")`.
  Without examples and without overrides the artifact is identical to the pre-change output except
  the new empty `filtered_files` (regression test).
- `src/omniscan/ingest/stage.py` — `version = 2  # 2: file-level promo filter`. `inputs()` adds
  `filter.json` (when it exists, regardless of `enabled`) and the example files of
  `promo_examples/{global,<series>}` (when enabled); the examples fingerprint is computed in
  `inputs()` and stored on the stage because `config_subset` has no `ctx` (the runner always calls
  `inputs()` before `config_subset()`). `config_subset = {"quality": 95, "filter": cfg.filter.model_dump(),
  "examples": fingerprint}`. `run()` loads the examples once (skipped when disabled), passes
  `threshold`/`overrides`/`filtered_dir`, metrics gain `"filtered_files"`.
- `src/omniscan/slicer/stage.py` — `version = 4  # 4: slice-level promo filter`. Same
  inputs/fingerprint pattern. `run()` applies `apply_slice_filter` when the filter is enabled and
  there are examples **or** slice-forced overrides; metrics gain `"filtered"`.
- `src/omniscan/cli.py` — `filter` group reworked: `run SERIES [-c NAME]... [--json]` runs
  ingest+slice through `_run_stages` (up-to-date stages stay skipped) and prints per-chapter
  `filtered files: N, filtered slices: M` plus the total line, or the documented JSON payload read
  from `ingest.json`/`slices.json`; `restore`/`force` go through `record_override` (create
  `filter.json` when absent) and print `will apply on the next run`; new `add SERIES PATH
  [--global] [--name NAME]` (plain-name check, JPEG/PNG suffix check, refusal to overwrite with
  exit 2, Pillow decodability check, `shutil.copy2`). `_assemble_strip` deleted (only the old
  `filter run` used it). `_run_stages` gained a keyword-only `progress` flag (see Deviations).
- `src/omniscan/web/app.py` — `get_page` resolves `pages/{index}` by `SourceFile.index`, not list
  position (ingest indices can now have gaps); 404 with a clear detail when the index is absent.
  `gui/services/library.py`, eval, preview, watermark resolution and the strip/slicer/export
  internals were audited (grep of `ingest.files` / `.index` users): they already use `f.index`
  directly, so only `get_page` needed the fix.
- `tests/unit/test_web_app.py` — regression test for the `get_page` gap fix; `tests/unit/
  test_slicer_strategies.py` — `SliceStage.version` assertion 3 → 4.
- `docs/USER_GUIDE.md` — the promo-filter section rewritten for the real commands
  (`run`/`restore`/`force`/`add`), artifact-table rows for `filter.json` and `_filtered/` updated.

`decide.py` was **not** modified: `load_examples`, `best_match`, `decide_files`, `decide_slices`,
`effective_decision` and `restore` keep their behaviour and tests.

## Tests

All CPU, synthetic images, `uv run --frozen --no-sync pytest -p no:cacheprovider` (note: this
worktree's `omniscan.exe` shim was locked by another process during the session, so every uv
invocation needed `--no-sync`; command results are identical otherwise).

1. Card command 1 — `pytest tests/unit/test_filter_apply.py tests/unit/test_filter_hashing.py
   tests/unit/test_ingest_filter.py tests/unit/test_slicer_filter.py tests/unit/test_filter_cli.py
   tests/unit/test_filter_decide.py tests/unit/test_docs.py -q` → **75 passed**.
2. Card command 2 — `pytest -m "not gpu"` (whole CPU suite) → **3472 passed, 22 deselected**
   in ~76 s on the pre-rebase tree; after the director's rebase onto main and the follow-up fix
   **3562 passed, 23 deselected, 1 xfailed** in ~84 s (the growth is the merged cards' tests plus
   the two new full-page-crop tests).
3. `ruff format .` → clean; `ruff check .` → **All checks passed!**; `pyright` →
   **0 errors, 0 warnings, 0 informations** (re-run after the follow-up fix).
4. GPU tests (`pytest -m gpu`, RX 9070 XT / ROCm 10) → **22 passed, 1 skipped** in ~106 s: the two
   `dhash_tensor` GPU parity/regression tests ran on the discrete GPU (the 2934×800 crash
   reproduction is now the regression test), and the golden E2E passed (6 tests, 40.37 s on its
   pre-fix run; no examples are configured there, so the filter is inert — see Follow-up fix for
   how the crash was found anyway). The one skip is `test_ocr_manga_gpu` (manga-ocr weights not
   installed in this worktree; unrelated to this card).
4. Golden E2E (`tests/unit/test_e2e_synthetic.py`, GPU, cached weights) → **6 passed in 40.37 s**
   on the RX 9070 XT; no examples are configured, so the filter is inert there (as expected).

Test inventory against the card's acceptance list:

1. **dhash_tensor** (`test_filter_hashing.py`): 24 seeded synthetic images (upscaled smooth noise,
   stepped ramps, stripes, solids) with `hamming(dhash, dhash_tensor) <= 6`; golden pair — rising
   horizontal gradient → both `0`, falling (the `gradient_jpeg` formula) → both `2**64 - 1`;
   determinism; 1-row/1-column; wrong shape → `ValueError`; CUDA-vs-CPU parity as a `gpu`-marked
   test (deselected on CPU). The fixture docstring records that pure per-pixel noise makes the two
   resize filters diverge beyond 6 bits (measured 8) — hence smooth/structured content only.
2. **Overrides/verdicts/fingerprint** (`test_filter_apply.py`): last-write-wins for both targets,
   automatic entries ignored, manual `keep` counts as neither, missing/corrupt/invalid → empty;
   boundary (`score == threshold` filters, just below keeps; `matched is None` → keep); forced
   beats a non-match; restored beats a match; fingerprint stability/sensitivity; `example_files`
   ordering; `apply_slice_filter` marking/blank-skipping/non-mutation/JPEG-saving/overrides/short
   slices.
3. **ingest_chapter** (`test_ingest_filter.py`): 5 pages with 2 near-copies (similarity ≥ 0.90
   asserted first so the fixture is honest) → `filtered_files == ["002.jpg", "004.jpg"]`, kept
   indices `[0, 2, 4]`, byte-identical `_filtered/` copies, correct strip pixels via `build_strip`;
   cache names use raw indices; all filtered → `ValueError("every image…")`; `files_restored={1}`
   keeps page 1; `files_forced={0}` drops page 0 with no examples; no examples → pre-change
   artifact (round-trip save/load + strip shape); noise-page guard.
4. **IngestStage** (`test_ingest_filter.py`, stage section): filters + metrics + `_filtered/`
   copies, second run skipped; adding an example file re-runs (invalidation) and re-skips;
   `record_override` on `filter.json` re-runs (invalidation) and keeps the restored file;
   `filter.enabled = False` ignores examples (metrics 0, sequential indices).
5. **SliceStage** (`test_slicer_filter.py`): hand-built ingest artifact + `load_strip` monkeypatch
   + `slicer.strategy = "page"` (deterministic slices); the falling-ramp banner (dHash pinned to
   2**64-1 on both implementations by the golden tests) matches the identical promo example →
   middle slice `filtered=True`, JPEG in `_filtered/`, others untouched; blank slices skipped
   (`apply_slice_filter` unit test); restored/forced overrides by slice index; adding an example
   invalidates (done → filtered, then skipped); no examples/no overrides and
   `filter.enabled = False` → `dhash_tensor` spy never called.
6. **Downstream**: `test_detect_postprocess.py::test_build_regions_drops_filtered_slices_and_clamps_boxes`
   (filtered slices produce no regions) and `test_export_stage.py::test_export_skips_filtered_slices`
   (filtered slice skipped, numbering compact) already covered this; `test_detect_stage.py` builds
   its active list from `not blank and not filtered`. Web `source_names` uses `f.index` directly
   (gap-safe as-is); the GUI library service is covered by the existing `test_gui_library.py`
   gapped/filtered cases; the one position-based user (`get_page`) got the fix + regression test
   (`test_page_resolves_by_source_file_index_with_gaps`).
7. **CLI** (`test_filter_cli.py`): `add` copy into series/global dirs, `--name` rename, refusal to
   overwrite (exit 2), non-plain `--name` (exit 2), non-image (exit 2), non-JPEG/PNG (exit 2);
   `run` prints per-chapter counts + total, second run prints `skipped` with the same counts,
   `--json` parses to the documented shape, unknown series exits 2; `restore` then `run` keeps the
   restored file (artifact asserts `["004.jpg"]`, indices `[0, 1, 2, 4]`); `force` without
   examples filters the file; bad target rejected. Docs test green (`test_docs.py` 9 passed).

## Follow-up fix (director review, 2026-09-22)

**Found on the real GPU:** `omniscan filter run` crashed in `slice` → `apply_slice_filter` →
`dhash_tensor` with ROCm's `RuntimeError: … antialias … Too much shared memory required: 90268 vs
65536`. `apply_slice_filter` hashes full-page crops (realistically 800–1600 px wide, 2000–5000 px
tall) and the ROCm antialiased `interpolate` overflows its box-filter buffer at those downscale
factors; the card's own tests only covered small images and the golden E2E has no promo examples,
so the filtered path had never run on the real GPU.

**Fix (`filter/hashing.py`):** `dhash_tensor` now routes through `_antialiased_resize` — one
antialiased `interpolate` for inputs with max(h, w) ≤ 512 (bit-identical to the previous
implementation, so every pre-existing hash is unchanged), and for larger inputs a coarse
`avg_pool2d` first, with integer per-axis factors that keep the intermediate within 512 px per
axis, then the same small antialiased resize. At 512 the final resize needs at most 57×64 box taps
(~15 KB), far below ROCm's 64 KB limit, and everything stays fp32 on the device with the same one
host transfer. Probed against PIL's dhash on full-page sizes: the staged path's Hamming distances
are *identical* to what the single-pass path produces on CPU (measured for 2934×800 and
5000×1600 on smooth, striped and stepped content — e.g. stripes 9 bits in both), i.e. the staging
adds no divergence of its own; the remaining large-size divergence is inherent to PIL-LANCZOS vs
antialiased-bilinear at extreme factors. Golden ramps are unchanged at full-page size (rising → 0,
falling → 2**64-1).

**Tests added** (`test_filter_hashing.py`):
`test_dhash_tensor_matches_dhash_on_full_page_crops` (CPU, 2934×800 smooth + golden falling ramp
through the staged path) and `test_dhash_tensor_full_page_crop_cuda_matches_cpu` (`@pytest.mark.gpu`,
the director's 3×2934×800 repro via `resolve_device("auto")` = the discrete GPU — pre-fix this is
exactly the RuntimeError; post-fix it runs and equals the CPU hash). Both ran on the RX 9070 XT.

**No stage-version bump:** sizes that already worked (max side ≤ 512) are bit-identical; larger
sizes previously *crashed* rather than produced a hash, and enabling the filter already changes the
stages' `config_subset` (invalidating those chapters), so stale slices.json from a pre-fix run
cannot silently keep wrong verdicts.

## Deviations

- **`--threshold` removed from `filter run`.** The threshold now lives in `cfg.filter.threshold`
  (the stages hash it in `config_subset`, so editing it invalidates correctly); keeping a CLI flag
  would have meant a second source of truth the stage cannot see. The card's CLI section for the
  new `run` did not list the option.
- **`decide.py` untouched; `record_override` added in `apply.py`.** `decide.restore` must keep
  raising `FileNotFoundError` (its existing test requires it), so the CLI's create-if-missing
  behaviour lives in the new helper.
- **`decide_files` / `decide_slices` kept** in `decide.py` (their tests keep passing); `_assemble_strip`
  in `cli.py` deleted — nothing else used it (per the card's "say so in the report").
- **`_run_stages` gained a keyword-only `progress` flag** (one line outside the filter group):
  without it, `filter run --json` mixed the ingest/slice progress lines into the JSON stream.
  With `progress=False` successes print nothing and failures go to stderr; every other command
  (no flag) behaves exactly as before. Documented in the CLI test `test_filter_run_json_shape`.
- **Test files beyond the card's list**: `test_web_app.py` (regression for the mandated
  `get_page` index-gap fix) and `test_slicer_strategies.py` (the mandated version bump 3 → 4
  invalidated its `version == 3` assertion).
- **CLI-level restore tested at file level only**; the slice-level restore/force overrides are
  covered at stage level (`test_slicer_filter.py`) because the smart slicer's cut positions are not
  deterministic enough to name a slice index in an end-to-end CLI test.
- **SliceStage test slices deterministically**: `slicer.strategy = "page"` with a hand-built
  ingest artifact and a `load_strip` monkeypatch (as the card suggests), banner = the golden
  falling luminance ramp, so the match is exact on every run.

## Questions

- None blocking. One judgement call worth a look from the director: `filter run --json` now prints
  **only** JSON on stdout (progress lines suppressed, stage failures on stderr, exit 1 unchanged).
  If the director prefers progress lines on stdout in `--json` mode too, `_run_stages(progress=…)`
  is the single switch to flip.
- Note for the director: the worktree's `.venv\Scripts\omniscan.exe` was held by another process
  for most of this session, so `uv run` needed `--no-sync` (it kept trying to reinstall the
  editable package and replace the shim). Worth checking whether a stray `omniscan web`/GUI process
  from an earlier session is still running from this worktree.
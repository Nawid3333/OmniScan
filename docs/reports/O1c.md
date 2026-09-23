# O1c — OCR qualification suite: which model is the default for which language

Owner: director (Nawid) · Builder: GLM (Claude Code) · Branch `O1c`, worktree `V:\OmniScan-wt\O1c`
Date: 2026-09-23

## Changes

| File | What |
|---|---|
| `src/omniscan/eval/qualify.py` (new) | The library: `Candidate`/`Dataset`/`Measurement`/`Summary`/`Recommendation` dataclasses; `load_plan` (TOML → candidates+datasets, `ValueError` naming the entry on bad input); `select` (candidate×dataset pairs, `lang`/`only` filters); `candidate_config` (deep config copy → qual work root + candidate OCR settings); `missing_models` (catalog status check); `run_qualification` (per-pair, per-chapter loop, exceptions and missing models become error Measurements, one progress line per event); `default_run_candidate` (real run: GPU lock held by caller, `resolve_device`, VRAM manager + peak-VRAM tracking, pipeline stages `ingest→slice→detect→ocr`, scores `ocr.json` against the truth with `eval.score.score_chapter`); `summarize` (per (lang, candidate) means, errors excluded but counted); `recommend` (KEEP/PROMOTE rule); `render_markdown` (deterministic table + verdicts). No CLI here. |
| `scripts/qualify_ocr.py` (new) | Thin runner: `--plan --lang --only --chapters --default LANG=ID --json --markdown --dry-run --download --yes`; points `paths.work_root` at `<work_root>_qual` (the normal work root is never written); takes the GPU lock when the device is not `cpu`; exit codes 0 done / 1 every measurement failed / 2 bad arguments; markdown to stdout, progress to stderr. |
| `config/qualification.toml` (new) | The shipped plan: 8 candidates (`ppocr-v5-ko`, `ppocr-v5-server-multi`, `ppocr-v6-tiny/small/medium`, `manga-ocr-2025`, `manga-ocr-base`, `paddleocr-vl-1.6`) and 3 datasets (ko→PepperCarrotKR E06/09 truth `kr`; ja→PepperCarrotJA E06/09/12/22 truth `ja`; zh→PepperCarrotCN E06/09/12/22 truth `cn`). |
| `tests/unit/test_eval_qualify.py` (new) | 34 CPU tests (no GPU, no models, no `data/`): see Tests. |
| `docs/benchmarks/ocr-qualification.md` (new) | Method, decision rule, datasets, prior evidence, run commands, empty Results section for the director run. |

Not touched (read-only, as required): `src/omniscan/core/**`, `config/models.toml`, `eval/score.py`, `eval/truth.py`.

## Tests

Commands and results (all run in the worktree):

- `uv run --frozen pytest tests/unit/test_eval_qualify.py tests/unit/test_docs.py -q` → **42 passed**
- `uv run --frozen pytest -m "not gpu"` → **3623 passed, 25 deselected, 1 xfailed** (the xfail is pre-existing: `test_hw_hf_mutation_gaps.py`)
- `uv run --frozen ruff format .` → 3 files reformatted (this card's files); `uv run --frozen ruff check .` → **All checks passed**
- `uv run --frozen pyright` → **0 errors, 0 warnings**

Acceptance tests from the card → test functions:

1. `load_plan` on the shipped TOML + bad files → `test_load_plan_shipped_plan_matches_the_catalog` (also checks every det/rec id exists in `config/models.toml` with the matching role), `test_load_plan_good_file`, `test_load_plan_bad_entries_name_the_entry` (duplicate id, missing engine, unknown engine, langs/chapters wrong types, non-string id, unknown field, empty chapters, missing truth, invalid TOML), `test_load_plan_toml_syntax_errors_become_value_errors`.
2. `select` → `test_select_pairs_candidates_with_their_datasets_in_order`, `test_select_filters_by_language_and_only` (language filter, `only`, empty `only`, unknown id, candidate with no matching dataset).
3. `candidate_config` → `test_candidate_config_applies_the_candidate_onto_a_copy` (deep copy, work root, engine/det/rec, dataset lang overrides config lang — `cn` truth stays `zh` in `ocr.lang` — input untouched), `test_candidate_config_keeps_its_lang_when_the_dataset_language_is_unknown`.
4. `run_qualification` with a fake `RunFn` → `test_run_qualification_runs_every_chapter_in_pair_order`, `test_run_qualification_isolates_failures` (exception → error Measurement, later pairs still run), `test_run_qualification_skips_pairs_with_missing_models` (one error Measurement per skipped pair, `chapter = dataset.chapters[0]`, log line names the download command).
5. `summarize`/`recommend` → `test_summarize_averages_ok_chapters_and_counts_errors`, `test_summarize_all_errors_has_no_means`, `test_recommend_keeps_when_the_default_is_best`, `test_recommend_promotes_a_clear_winner` (exactly at `min_gain`), `test_recommend_keeps_on_small_gain_or_recall_loss`, `test_recommend_tie_breaks_on_recall_then_seconds`, `test_recommend_never_replaces_a_default_without_data` (missing default / all-errors default / other-language-only data).
6. `render_markdown` golden text → `test_render_markdown_golden` (two languages, PROMOTE + KEEP, `★` on the defaults, `n/a` rows for the all-error candidate, skipped-models note), `test_render_markdown_omits_the_skip_note_without_installed`.
7. `default_run_candidate` with monkeypatches → `test_default_run_candidate_measures_one_chapter` (tiny real artifacts in a tmp work dir; fake `run_pipeline`/`load_truth`/`load_english_pages`/`score_chapter`/`build_vram_manager`; asserts stages, `force=False`, gpu handle passed, truth check dir + folder, regions count, no VRAM tracking on CPU, manager released), `test_default_run_candidate_reports_errors_and_releases`, `test_default_run_candidate_reports_a_failed_stage` (`result.failed` → error Measurement).
8. Script `main(argv)` → `test_main_runs_reports_and_writes_outputs` (runs into `<work_root>_qual`, stdout markdown, `--json`/`--markdown` files, PROMOTE verdict from `--default ko=cand-a`), `test_main_dry_run_prints_pairs_without_running` (pairs + missing models printed, pipeline never called, exit 0), `test_main_only_and_chapters_restrict_the_run`, `test_main_bad_arguments_exit_2` (bad `--default` shape, unknown default id, `--chapters 0`, unreadable plan, no pairs selected), `test_main_exit_1_when_every_measurement_failed`.

## Deviations from the card

1. **VRAM tracking scope**: peak VRAM is tracked (`reset_peak_memory_stats` / `max_memory_allocated`) only when `resolve_device(cfg.gpu.device).type == "cuda"`; CPU runs report `peak_vram_gib = None`. This avoids touching `cuda:0` (the iGPU on this PC) and keeps CPU runs CUDA-free. `torch.cuda.is_available()` is deliberately not consulted.
2. **`Summary` has two extra fields** (`series`, `chapters`) beyond the card's list: `render_markdown` needs them for the `## <lang> (<series>, <n> chapter(s))` heading and to report per-language chapter counts.
3. **`Measurement.recall_chars` is `float | None`** (like `chrf`/`cer_micro`): `score_chapter` can leave it unset, so means skip `None`s and an all-`None` summary reports `n/a`.
4. **Skipped pairs emit exactly one error Measurement** (card: "one measurement per pair") with `chapter = dataset.chapters[0]` as a stand-in; the stderr line carries the `omniscan models download <ids>` command.
5. **`default_run_candidate` treats a failed pipeline chapter as an error** (`RuntimeError: pipeline failed: <stage>: <error>` from `result.failed`) instead of scoring whatever `ocr.json` a previous candidate left behind.
6. **`installed` rendering**: the card specifies the parameter but not what it renders; chosen rendering is a final note `Skipped, models not installed: <ids> — install with \`omniscan models download <model id>\``.
7. **The script also exits 2 on an unreadable `--plan`** (`OSError`), not only on `ValueError` from `load_plan`.
8. **PROMOTE reason string**: `gains <X> chrF with <Y> char recall loss`, where `Y` is negative when recall actually improves. The KEEP reasons are `the default is the best candidate`, `best candidate <id> gains only <X> chrF`, `best candidate <id> loses <Y> char recall`, `no data for the default`.

## Questions

1. **en has no dataset.** The plan ships ko/ja/zh datasets only (English Pepper&Carrot pages are the source images, there is no en truth). en candidates (`ppocr-v5-server-multi`, `ppocr-v6-*`, `paddleocr-vl-1.6`) will report "no data for the default". Fine as-is, or should a later card produce an en truth set (e.g. scored against the original English text)?
2. **Cleanup of `<work_root>_qual`.** After a real run each candidate leaves ingest/slice/detect/ocr artifacts per chapter (needed for reuse across candidates — the vision stages skip when up to date). Keep everything for inspection, or delete the qual work root at the end of a successful run? Current default: keep.
3. **manga_ocr candidates carry no `det_model`**, so `candidate_config` sets `det_model = None` → the pipeline's default detector (from `DetectConfig`) is used and the candidate measures "current detector + candidate recognizer". OK, or should ja candidates also pin an explicit detector (a possible follow-up card)? Default: current behaviour.

None of these block the director run.

## Out of scope (not done, as instructed)

Running the suite for real (director), changes to `models.toml` `recommended_for`/defaults, translation/detection quality, eval truth-assignment fixes, card W1.
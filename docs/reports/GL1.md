# GL1 — reference-mode glossary bootstrap (D7)

## Changes

- `src/omniscan/glossary/reference.py` (new, ~530 lines) — the whole reference pass, one module:
  - `run_reference(cfg, series, *, client, model, min_locks, dry_run, force, gpu) -> ReferenceSummary`:
    merges `series_config(cfg, <real series>.library_dir)` **once, first** (the config trap — see
    Tests #2), builds the pseudo series `SeriesPaths.from_config(cfg, f"{series}/{REFERENCE_DIR}")`
    (verified to land exactly on `SeriesPaths.reference_dir` / `work_root/<series>/_reference_en/` —
    see Tests #3), runs CM1 `match_chapters(library_dir, reference_dir)`, OCRs both sides through
    `run_pipeline` stages `("ingest", "slice", "detect", "ocr")` — no second OCR path — pairs the
    OCR'd regions per matched chapter, extracts terms via the `ChatClient` Protocol (one request per
    matched chapter), aggregates, merges into the store, re-exports `glossary.yaml`.
  - `pair_chapter(regions_a, regions_b, ingest_a, ingest_b, pages) -> PairingResult`: regions are
    mapped to pages via the `IngestArtifact.files` y-ranges (max vertical overlap), then paired by
    `(reading_order, natural_key(id))` rank within each aligned page pair from `ChapterMatch.pages`.
    A page pair whose region counts differ by more than `MAX_REGION_COUNT_DIFF = 1` is skipped and
    counted in `pages_skipped` (±1 tolerates one stray/missed detection; more than that means the
    layouts diverged and rank-pairing would silently mismatch lines). Watermark and empty-text
    regions are unpairable on either side; whitespace is collapsed.
  - `extract_candidates(client, model, chapter, lines)`: builds `terms_messages` from the paired
    `(source, target)` lines, calls `client.chat(model, …, format=TERMS_SCHEMA,
    options={"temperature": 0.0})`; no request at all for a chapter with no paired lines.
    `parse_terms_reply` tolerates prose around the JSON, unknown types → `"other"`.
  - `aggregate(candidates) -> list[AggregatedTerm]`: groups by collapsed source; the winner target
    is the casefolded-most-frequent (ties → most chapters → most occurrences → alphabetical);
    runner-up spellings are kept as `alternatives` (they reach the entry's `notes`, so the human
    reviewer sees the disagreement). Over-length junk (source > 30, target > 40 chars — sentence-
    sized OCR debris) is dropped from candidate targets, not silently merged.
  - `merge_into_store(store, terms, *, min_locks=3, write=True) -> MergeReport`: a `(source, target)`
    agreeing across ≥ `min_locks` distinct chapters is written `status="locked"`,
    `origin="reference"` (D7); everything else `status="proposed"` — never dropped, never auto-locked
    below the threshold. Existing entries are updated via `find_by_source`, never duplicated:
    locked+agreeing → `count` bumped (idempotent, never lowered), locked+disagreeing → flagged as a
    `Conflict` and left untouched, agreeing with a user-proposed entry → bump, disagreeing with one →
    conflict, a machine-proposed reference entry is updated in place with a `"replaced target"`
    note. `write=False` computes the identical report without touching the store (dry run).
  - `format_summary(summary) -> list[str]` — the printed summary (counts, raw-only/reference-only
    chapters, OCR failures, locked/proposed, conflicts, rejected leftovers, dry-run notice).
- `src/omniscan/glossary/reference_prompts.py` (new) — `EXTRACT_SYSTEM`, `TERMS_SCHEMA` (JSON
  schema with the `TermType` enum), `PairedLine`, `terms_messages`, `parse_terms_reply`; plain
  prompt-builder functions in the `translate/prompts.py` style, no prompt strings elsewhere.
- `src/omniscan/cli.py` — `cmd_reference` is real (the `_STUB_COMMANDS`/`_stub`/`_register_stubs`
  stub machinery is gone; `test_docs.py`'s stub-doc test now exercises `_check_stub` directly, the
  C4a pattern). `omniscan reference SERIES [--min-locks N] [--model M] [--dry-run] [--force]`:
  validates the series has chapters and `_reference_en` has reference chapters (else exit 2 with a
  pointer at the user guide), merges the series config, and on a non-CPU device acquires the GPU
  lock **before** `build_vram_manager` and releases both in `finally` (standing 2026-09-22 rule);
  `OllamaRateLimitError` → exit 3, `ValueError` → exit 2, any OCR failure → exit 1 after printing
  the summary. `--dry-run` still runs the (cached) OCR — the card forbids a second OCR path — but
  writes no store/yaml; with an existing db it computes the conflict report read-only.
- `tests/unit/test_glossary_reference.py` (new, 26 tests), `tests/unit/test_reference_cli.py`
  (new, 18 tests), `tests/unit/test_reference_gpu.py` (new, the card's one `@pytest.mark.gpu` e2e).
- `docs/USER_GUIDE.md` — new `### omniscan reference` section (import convention, options, merge
  semantics, example summary, exit codes); folder table rows for `_reference_en` / `series.db` /
  `glossary.yaml` updated. `README.md` — glossary status row now describes `omniscan reference`;
  the "Not implemented yet" section is gone (nothing is).

## Tests

Commands (final state):

```
uv run --frozen pytest tests/unit/test_glossary_reference.py -q -m "not gpu"   # 26 passed
uv run --frozen pytest tests/unit/test_reference_cli.py -q -m "not gpu"        # 18 passed
uv run --frozen pytest -m "not gpu"                                            # 3759 passed, 25 deselected, 1 xfailed
uv run --frozen ruff format . && uv run --frozen ruff check .                  # 1 file reformatted, then All checks passed
uv run --frozen pyright                                                        # 0 errors, 0 warnings, 0 informations
uv run --frozen pytest tests/unit/test_reference_gpu.py -q                     # 1 passed (real GPU, 18.75s)
```

(the 1 xfail is pre-existing: `test_hw_hf_mutation_gaps.py::test_hf_marker_that_is_a_json_list_is_corrupt`)

### Acceptance evidence

1. **Extraction / aggregation / locking** (`test_glossary_reference.py`, synthetic `Region` +
   ingest artifacts built in code, fake `ChatClient`):
   - 민준→Minjun in 3 chapters → `locked`, `origin="reference"`, `count=3`,
     `first_seen_chapter=1.0`; the same pair in only 2 chapters → `proposed` (never dropped);
   - disagreeing targets across chapters → stays `proposed`, the runner-up spelling is recorded in
     the entry's `alternatives`/notes, majority wins by (chapters, occurrences, alphabetical);
   - a term conflicting with an already-`locked` entry → reported as a `Conflict`
     (`reference:   conflict: 민준: store has 'Min-jun' (locked), reference extraction says 'Minjun' — entry left untouched`)
     and the row is untouched;
   - a term agreeing with a locked entry bumps `count` (never lowers it, never duplicates the row —
     asserted via `find_by_source` and re-run idempotency);
   - plus: reading-order-rank pairing (ids that oppose `reading_order` still pair correctly),
     y-range page mapping, ±1 stray-region tolerance, divergent page pairs skipped, watermark/empty
     regions skipped, no chat request without paired lines, schema/temperature/chapter tagging,
     malformed-reply tolerance, case folding with most-frequent spelling kept, sentence-length
     targets rejected, custom `--min-locks`, `write=False` report-only.
2. **Config trap** — `test_both_sides_ocr_with_the_series_merged_config` (parametrized cpu/cuda):
   with `series.toml` setting `[ocr] engine = "manga_ocr"` on the real series, both `run_pipeline`
   calls (raw side *and* the pseudo reference side, whose directory has no series.toml of its own)
   receive a cfg whose `ocr.engine == "manga_ocr"` — the explicit pre-merge in `run_reference`
   mirrors the bug class fixed in `cli.py`/`queue/executor.py`.
3. **Pseudo-series paths** — `test_pseudo_series_paths_land_reference_artifacts_in_the_reference_dir`
   asserts `ref_sp.library_dir == sp.reference_dir`, `ref_sp.chapters()` lists the reference
   chapters, `work_dir` is `work_root/S/_reference_en/ch_01`, and `list_chapters(sp.library_dir)`
   never sees `_reference_en` (the `_` prefix skip). `test_reference_side_reads_english_and_raw_side_reads_korean`
   proves end-to-end reading direction: the extraction prompt contains both 민준이 and Minjun, the
   raw-side artifacts land in `work_root/S/Chapter 001/ocr.json` and the reference-side ones in
   `work_root/S/_reference_en/ch_01/ocr.json`. The GPU e2e proves the same through the real pipeline.
4. **CLI** (`test_reference_cli.py`, real CM1 matcher over synthetic chapter sets, fake pipeline
   that writes the exact artifacts `run_reference` reads, fake client):
   - unmatched raw chapters are OCR-neither-nor-merged: reported as `raw chapter without reference
     match: …` and absent from both pipeline calls;
   - `--dry-run` writes no `series.db` and no `glossary.yaml`;
   - the full summary is printed and the client is closed;
   - GPU lock ordering: `test_reference_cli_acquires_the_gpu_lock_before_the_vram_manager`
     (device="cuda") records `["lock", "manager", "run", "release", "unlock"]` — the lock is held
     before `build_vram_manager` and released after the run;
   - exit codes: 2 unknown series / no reference chapters (message points at the user guide),
     1 when any OCR failed, 3 on `OllamaRateLimitError` (lock and manager still released);
   - `--min-locks/--model/--force` pass through; `--force` re-runs up-to-date OCR stages.
5. **The real-GPU e2e** (`tests/unit/test_reference_gpu.py`, actually run on this machine — RX 9070 XT,
   ROCm 10): three synthetic Korean chapters (real `make_korean_page` art → real detect+OCR) against
   three "official" reference chapters — the same pages with English lettering drawn into the same
   bubbles — and a scripted chat client. Real CM1 matched all three pairs; real pipeline OCR'd both
   sides through the pseudo series paths. Output:

   ```
   reference: 3 matched chapter pair(s), 0 raw-only, 0 reference-only, 9 paired line(s), 0 page pair(s) skipped
   reference: 2 term(s) locked, 1 proposed
   ```

   Assertions: 민준→Minjun `locked` / `origin="reference"` / `count=3` / `first_seen_chapter=1.0`;
   던전→dungeon (agreed in only 2 of 3 chapters) `proposed`; 헌터→hunter `locked`; exactly 3 rows;
   `glossary.yaml` exported next to the store; exactly 3 extraction prompts (one per matched pair).
   Skips (never downloads) when detector/OCR weights are not cached. 1 passed in 18.75s (16.3s of
   that is the 6-chapter pipeline run).
6. Full-suite, lint, types, docs: the command block at the top of this section; `test_docs.py`
   9 passed (stub-doc test updated, see Deviations).

## Deviations

- **No importer convenience for `_reference_en`**: the owner imports the official chapters with the
  existing command — `omniscan import <folder> --series "<Series>/_reference_en"` — which lands
  exactly on `SeriesPaths.reference_dir` because the pseudo name is a plain path join. Documented in
  the user guide with an example; a dedicated flag would have meant touching `importer/**`, which
  the card's file list doesn't include.
- **`--dry-run` still runs OCR** (writes the normal cached artifacts, nothing else). Building a
  no-write OCR variant would be a second OCR path, which the card forbids; pipeline caching makes
  the re-run free, and the store/yaml/report writes are all gated.
- **`tests/unit/test_docs.py`** — the stub-documentation test was rewritten to exercise `_check_stub`
  directly with a fake stub command, because GL1 removes the last live stub (same move as C4a).
  **`tests/unit/test_acquire_cli.py`** — `test_reference_is_still_a_stub` deleted: its subject no
  longer exists. **`README.md`** — status row + removed "Not implemented yet" section (B33d
  precedent; keeping the section with zero entries would contradict the table).
- **Default extraction model `gemma4:31b-cloud`** (the cloud tag keeps it off the local VRAM budget
  during a GPU-heavy OCR pass); `--model` overrides. One request per matched chapter (chapter-level,
  per the card), so `count` on an entry = the number of distinct reference chapters that pair
  occurred in — occurrences beyond one per chapter are not separately counted.
- **`MAX_REGION_COUNT_DIFF = 1`**: rank-pairing within a page pair is only trustworthy when the two
  sides found nearly the same bubbles; ±1 absorbs a single stray/missed detection, larger deltas
  mean the layouts genuinely diverged (inserted splash page, art revision) and guessing would pair
  the wrong lines — those page pairs are counted in the summary instead.
- **GPU-test page seeds are measured, not arbitrary**: raw-vs-reference dHash similarity for seeds
  303/304/306 is 0.91/0.92/0.98, safely above CM1's 0.75 `page_similarity` gate (earlier seeds
  measured 0.72 and fell out of the page map — the comment in the test records this).
- `OllamaRateLimitError` → exit 3 was not in the card's sketch; added for consistency with
  `cmd_run`'s rate-limit handling (the weekly cap is a real, hit-before condition — see
  `docs/OPEN_QUESTIONS.md` Ollama limits).

## Questions

- None blocking. Two design notes for the director:
  - **The reference side is OCR'd with the series' Korean OCR config** — the Korean recognizer reads
    the English reference lettering. This worked on the synthetic e2e (PP-OCR's Korean models carry
    Latin glyphs), but a per-side OCR override would be the robust answer for real releases. That
    needs a `core/config.py` change (per-side overrides), which this card could not make; note that
    putting a `series.toml` inside `_reference_en/` is NOT a workaround — `build_vram_manager` only
    ever receives the real series' merged config, so the pseudo side's overrides would hit the
    R1-class KeyError('reader') trap.
  - `count` semantics: because extraction is chapter-level, `count` = distinct reference chapters
    agreeing, not total occurrences across pages. That is the number D7's ≥3 rule actually gates on.
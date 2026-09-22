# GL1 — reference-mode glossary bootstrap (D7)

**Owner:** GLM builder · **Branch:** `GL1` · **Worktree:** `V:\OmniScan-wt\GL1` · run with `--max-turns 300`
Read `CLAUDE.md` first. Then `docs/OPEN_QUESTIONS.md` D7 (the decided default this card implements: auto-lock
terms consistent across **≥ 3** reference chapters; official release counts as authoritative). Then read, in
this order: `src/omniscan/core/paths.py` (`REFERENCE_DIR = "_reference_en"`, `SeriesPaths.reference_dir`,
`SeriesPaths.chapters()`, `ChapterPaths`), `src/omniscan/core/schemas.py` (`GlossaryEntry` — note
`origin: Literal["llm", "reference", "user"]` already exists, and `Region` — `text`, `reading_order`,
`slice_index`), `src/omniscan/glossary/store.py` (`GlossaryStore`), `src/omniscan/glossary/yaml_io.py`
(`export_yaml`), `src/omniscan/glossary/match.py` (`find_terms`, existing particle-aware matching — reuse,
don't reimplement), `src/omniscan/translate/prompts.py` (`glossary_subset`, `chat_json_messages`,
`substitute_binding` — how locked entries already flow into translation; you are not changing this side),
`src/omniscan/translate/run.py` (`ChatClient` Protocol — reuse for your own LLM extraction calls, do not
invent a second client), `src/omniscan/match/chapters.py` + `align.py` + `pages.py` (CM1, already merged —
`match_chapters(dir_a, dir_b, thresholds) -> ChapterMapping` for chapter correspondence, and the page-level
`align`/`similarity_blocks` machinery you should reuse for page-inside-chapter alignment, see Method),
`src/omniscan/core/stage.py` (`ChapterContext`, `Stage`, `run_series`, `StageOutcome`), `src/omniscan/pipeline/
runner.py` (`run_pipeline`, `plan_passes` — note it already does `cfg = series_config(cfg,
SeriesPaths.from_config(cfg, series).library_dir)` internally), `src/omniscan/core/config.py`
(`series_config` — returns `cfg` unchanged if no `series.toml` is found at the given dir, see Method for why
that matters here), `src/omniscan/cli.py` lines 217–275 (`_run_stages` — the existing GPU-lock +
`build_vram_manager` pattern every real-GPU command follows) and lines ~900–913 (`_STUB_COMMANDS`,
`cmd_reference`, `_register_stubs` — the stub you are replacing), `src/omniscan/gpu/lock.py` (`gpu_lock`,
`acquire_gpu_lock`, `release_gpu_lock` — cross-process GPU mutual exclusion; **mandatory**, see Constraints).
This card is a goal, not a spec: you design the classes; the acceptance evidence at the bottom is what counts.

## Why
The owner will often have more raw chapters than officially-translated ones for a series (e.g. 40 raw
chapters, an official English translation of only the first 20). Those 20 translated chapters are free,
high-quality glossary training data: character names, place names, titles and other recurring terms already
translated consistently by a professional. Bootstrapping the glossary from them — instead of letting the LLM
invent romanisations chapter-by-chapter — makes translation of the remaining, untranslated chapters (21–40)
more consistent from the first one. This is exactly card D7 ("Reference mode") in `docs/OPEN_QUESTIONS.md`,
already decided (3-chapter consistency threshold, official release authoritative), with a schema field
(`GlossaryEntry.origin == "reference"`), a reserved library layout (`SeriesPaths.reference_dir`,
`_reference_en/`) and a stubbed CLI command (`omniscan reference`) already waiting for this implementation.

## Goal
Implement `omniscan reference SERIES` (replacing the stub): given a series whose library already has raw
chapters imported (`SeriesPaths.library_dir/<chapter>/`, via `omniscan import`) and an official English
reference translation for some of those chapters imported under `SeriesPaths.reference_dir`
(`library_dir/_reference_en/<chapter>/`, same import mechanism, same page-image shape), the command:
1. Finds which raw chapters have a corresponding reference chapter (some raw chapters will have none — that
   is the normal case, not an error).
2. OCRs both sides (reusing the existing `ingest`/`slice`/`detect`/`ocr` pipeline stages — do not build a
   second OCR path) to get per-region text in reading order for every matched chapter pair.
3. Extracts source→target term candidates (character names, places, titles, organisations, recurring
   items/skills — the existing `TermType` literal) from each matched chapter pair's paired text.
4. Aggregates candidates across all matched chapters; a `(source, target)` pair that recurs identically in
   **≥ 3** distinct reference chapters is written to the glossary as `status="locked"`, `origin="reference"`;
   everything else is written `status="proposed"` for human review (never silently dropped, never auto-locked
   below the threshold).
5. Merges the result into the series' `GlossaryStore` (existing entries with the same `source` are updated,
   not duplicated — `GlossaryStore.find_by_source`) and re-exports `glossary.yaml`
   (`glossary.yaml_io.export_yaml`) so the owner can hand-review/correct it.
6. Prints a summary (chapters matched / raw-only / reference-only, terms locked, terms proposed, terms that
   conflicted with an already-`locked` entry — flag conflicts, never silently overwrite a human-locked term).

The existing `translate` stage already consumes `GlossaryStore` entries via `translate/prompts.py` — this
card only has to get good entries into the store before the owner runs the normal `omniscan run` on the
untranslated chapters. **Do not touch `translate/prompts.py`, `translate/chapter.py` or anything that
consumes the glossary during translation** — out of scope, see below.

## Method (a strong starting point, not a fixed spec — investigate, measure, decide)
- **Chapter correspondence** (raw chapter ↔ reference chapter): reuse `omniscan.match.chapters.match_chapters`
  (CM1, merged) over `SeriesPaths.library_dir` vs `SeriesPaths.reference_dir` — it already handles "only some
  raw chapters have a match," differing folder names, and flags weak matches for review, with no new
  alignment code needed. A raw chapter with no reference match is normal (`unmatched_a`), not an error.
- **OCR both sides**: the reference chapters live in a sibling folder, not as top-level series chapters, so
  `SeriesPaths.from_config(cfg, series).chapters()` won't see them directly. `SeriesPaths.from_config` builds
  paths as `cfg.paths.library_root / series` — passing the pseudo series name
  `f"{series}/{REFERENCE_DIR}"` (a `Path` join treats the embedded `/` as a normal path separator) lands
  exactly on `SeriesPaths.reference_dir`, `work_root/<series>/_reference_en/`, etc. Verify this actually does
  what you expect (write a small test) before relying on it; it is a strong lead, not guaranteed.
  **Config trap** (the same class of bug fixed twice this session in `cli.py`/`queue/executor.py`): merge
  `cfg = series_config(cfg, SeriesPaths.from_config(cfg, series).library_dir)` **once**, using the *real*
  series name (so the series' own `series.toml` OCR/detect overrides apply), before running OCR on either
  side. `series_config` silently returns `cfg` unchanged when no `series.toml` exists at the given path (see
  `core/config.py:233-250`) — so `run_pipeline`'s own internal re-merge against the pseudo series path is a
  harmless no-op *only if* you already passed it the pre-merged cfg; passing it the raw, unmerged `cfg` would
  silently run reference-side OCR with the wrong engine. Write a test that catches this class of bug (assert
  the reference-side OCR call receives the series-merged config), the same way `test_queue_executor.py` /
  `test_pipeline_cli.py` do for the equivalent `cli.py`/`queue/executor.py` bug.
  Use `pipeline.runner.run_pipeline(cfg, pseudo_or_real_series, chapters=[...], stages=["ingest", "slice",
  "detect", "ocr"], gpu=gpu, force=force)` (or `core.stage.run_series` directly, `_run_stages`'s own
  approach) — either is fine, your call; don't build a third way to run stages.
- **Pairing OCR'd regions within a matched chapter pair**: reuse CM1's page-level alignment
  (`match.pages.chapter_hashes`/`similarity_blocks`, `match.align.align`) to align raw pages to reference
  pages inside the pair (tolerant of an inserted page on either side — the same tool CM1 built for exactly
  this kind of problem, no reason to build a second one). Within an aligned page pair, pair OCR regions by
  `reading_order` rank (simplest starting point — the same panel layout, just re-lettered, should keep the
  same bubble order and usually the same bubble count). Skip/flag a page pair whose region counts differ too
  much to pair confidently rather than guessing; decide and justify the threshold.
- **Term extraction**: an LLM call (via `ChatClient`, cloud or local per existing config) over the paired
  (source line, target line) text of one chapter, asking for a structured JSON list of recurring
  proper-noun-like terms and their consistent translation (source, target, type, optionally gender/notes).
  Design the prompt; keep it in a new `prompts.py`-style module next to your other new code, following the
  existing `translate/prompts.py`/`translate/judge_prompts.py` style (plain functions building message lists,
  no LLM string formatting outside a dedicated module). Chapter-level extraction, not per-region — a
  short chapter transcript in, a JSON term list out, is both cheaper and gives the model context to recognise
  which terms actually recur.
- **Consistency lock**: aggregate `(source, target)` occurrence counts across matched chapters (case handling,
  normalisation — decide and justify); `≥ 3` distinct chapters agreeing on the same `target` for a `source`
  locks it (`origin="reference"`, `status="locked"`, `count` = occurrences, `first_seen_chapter` = the
  earliest matched raw chapter's number). A `source` seen fewer times, or with disagreeing targets, is written
  `proposed` (each disagreement is real signal for the human reviewer — don't just pick the majority silently
  and hide the runner-up; record it in the report / summary at least, a `notes` field on the entry is fine).
  A term that already exists in the store as `status="locked"` (e.g. hand-locked by the owner, or from a
  previous run) must never be silently overwritten by a new reference pass — flag the conflict, don't touch
  the entry, unless the extraction agrees with it (then just bump `count`).

## Files you may create / modify
- `src/omniscan/glossary/reference.py` (or your own module layout under `src/omniscan/glossary/` — extraction
  prompt building, the OCR-both-sides orchestration, aggregation/locking logic, the CLI entry point's actual
  work function).
- `src/omniscan/cli.py`: replace `cmd_reference`'s stub body with the real command (`SERIES` argument, decide
  your own flags — e.g. `--min-locks` overriding the default 3, `--dry-run` printing the summary without
  writing, `--force` re-processing chapters already reflected in the store); remove `"reference"` from
  `_STUB_COMMANDS` and un-stub `cmd_reference` accordingly (keep `_register_stubs`/`_STUB_COMMANDS` working
  for whatever other stub commands remain there). Must acquire `omniscan.gpu.lock.gpu_lock()` (or
  `acquire_gpu_lock`/`release_gpu_lock`) around the OCR work exactly like `_run_stages`/`cmd_run` already do
  (cli.py:236-261, 776-863) — this card adds a new real-GPU entry point and the standing rule is that every
  one of those goes through the lock, no exceptions.
- `tests/unit/test_glossary_reference.py` (or your own filenames, one per new module) — extraction/aggregation/
  locking logic tested with synthetic OCR `Region` data built directly in code (no real OCR/GPU needed for
  this layer) and a fake `ChatClient` test double (same pattern `tests/unit/test_judge.py` /
  `test_pipeline_runner.py` already use).
- `tests/unit/test_reference_cli.py` (or similar) for the CLI plumbing (fake client + fake/small pipeline, no
  GPU) plus **one** `@pytest.mark.gpu` end-to-end test that actually runs OCR on tiny synthetic images through
  the real pipeline for both a raw and reference chapter and checks a term makes it into the store — mirroring
  how other GPU-adjacent cards keep exactly one or two real integration tests instead of mocking the GPU away
  entirely.
- `docs/USER_GUIDE.md`: new section on `omniscan reference` — the `_reference_en` import convention (how the
  owner gets translated chapters there: `omniscan import <folder> --series SERIES --chapter "Chapter 5"`
  pointed at `SeriesPaths.reference_dir` — decide whether that needs a small importer convenience or the
  owner just imports to that path directly with `--series "SERIES/_reference_en"`; say which and why in the
  report), what the command does, how to read/correct `glossary.yaml` afterwards.
- `docs/reports/GL1.md`.
Do not edit `src/omniscan/core/**` (director-owned) — if the pseudo-series-name trick for reaching
`reference_dir` turns out not to work cleanly, stop and describe the `core.paths`/`core.stage` change that
would help instead of hacking around it. Do not edit `translate/**` (out of scope, see Why).

## Constraints
- GPU code follows every rule in `CLAUDE.md`: `omniscan.gpu.device.resolve_device`, no hard-coded `cuda:0`,
  tensors stay on GPU between steps in the reused stages (you are not writing new tensor code, just invoking
  existing stages — but the CLI wiring must not break that). Never import `paddlepaddle`/`paddleocr` directly.
- **Every real-GPU code path this card adds must go through `omniscan.gpu.lock`** — this is the standing rule
  from the owner's 2026-09-22 GPU driver crash; do not add a second GPU entry point that bypasses it.
- No new dependency for the LLM extraction call (`ChatClient`/Ollama is already wired); no new dependency for
  YAML (already used).
- Tests use synthetic fixtures generated in code — small solid-colour/simple-shape images (style of
  `tests/fixtures/chapter_sets.py` from CM1, or `tests/fixtures/images`), synthetic OCR `Region` objects built
  directly, a fake `ChatClient` returning canned JSON. Never real manga/manhwa content, per the repo's fixture
  rule. The one `@pytest.mark.gpu` end-to-end test may use tiny synthetic page images through the real models.

## Acceptance evidence (put it in `docs/reports/GL1.md`)
1. Unit tests for the extraction/aggregation/locking logic: a term consistent across exactly 3 chapters locks;
   across 2 does not (stays proposed); a term with disagreeing targets across chapters stays proposed with
   both candidates recorded; a term that would conflict with an already-`locked` entry is flagged and the
   existing entry is untouched; a term agreeing with an already-`locked` entry bumps its `count` without
   duplicating the row (`find_by_source`).
2. A test proving the config trap described in Method is actually avoided (reference-side OCR runs with the
   real series' merged `series.toml` overrides, not the base config) — same shape as
   `test_gpu_jobs_build_the_vram_manager_from_series_merged_config` in `tests/unit/test_queue_executor.py`.
3. A test proving the pseudo-series-path trick lands OCR artifacts and reads reference chapters from
   `SeriesPaths.reference_dir`, not from the raw library dir.
4. CLI tests: chapters with no reference match are left alone (not an error, reported as such); `--dry-run`
   writes nothing; a summary is printed with the counts described in Goal step 6; the GPU lock is acquired
   around the OCR work (assert via a fake lock / call-order check, matching how `test_pipeline_cli.py`
   verifies the equivalent for `cmd_run`).
5. The one `@pytest.mark.gpu` end-to-end test's actual output (what it asserts, and that it passed on this
   machine — run it for real, don't just claim it).
6. `uv run --frozen pytest -q -m "not gpu"`, `ruff format . && ruff check .`, `pyright` clean.
   `docs/USER_GUIDE.md` updated.

## Out of scope
Changing how `translate`/`judge` consume the glossary (`translate/prompts.py` already does the right thing
with `locked`/`proposed` entries — this card only populates the store better). A GUI page for this (there is
no assembled main window review flow for glossary yet). Any cross-language chapter matching beyond what CM1
already provides — reuse it, don't extend it. Auto-translating the remaining (non-reference) chapters — that
is the existing `omniscan run` / GUI run flow, unchanged by this card. Downloading or fetching anything from a
URL — this card only ever reads two local directories the owner has already imported (raws and reference
translation) via the existing `omniscan import` / upload path.

## Commands to run before finishing
```bash
uv run --frozen pytest tests/unit/test_glossary_reference*.py tests/unit/test_reference_cli.py -q -m "not gpu"
uv run --frozen pytest -q -m "not gpu"
uv run --frozen ruff format . && uv run --frozen ruff check .
uv run --frozen pyright
```

## Report
`docs/reports/GL1.md`: Changes, Tests (commands + results), Deviations, Questions. Commit early
(`GL1: WIP reference-mode bootstrap`), final commit `GL1: reference-mode glossary bootstrap`. You have 300
tool calls. If anything is unclear, stop, write the question in the report, commit, and end.

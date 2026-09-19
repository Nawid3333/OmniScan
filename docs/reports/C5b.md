# C5b — The judge: choose / merge / rewrite between candidate translations → `final.json`

## Changes
- `src/omniscan/translate/judge_config.py` (new): `JudgeConfig` pydantic model with `extra="forbid"` and the
  card's defaults (`model="gemma4:31b-cloud"`, `endpoint="local"`, `think=False`, `temperature=0.2`,
  `chunk_regions=20`, `max_repair_rounds=1`, `agree_threshold=0.9`, `always_judge=False`, `prefer=[]`).
  `default_judge_paths()` = `[config/judge.toml, ~/.config/omniscan/judge.toml]`. `load_judge_config` skips
  missing files, expects a `[judge]` table, validates each file separately (`ValueError` naming the file for
  bad TOML, a non-dict `[judge]`, or any field error), and merges field-level: only the keys a file sets are
  overridden (`model_dump(exclude_unset=True)`), so unlike profiles a later file does not replace whole keys.
- `src/omniscan/translate/judge_prompts.py` (new): `JUDGE_SYSTEM` verbatim card text; `JUDGE_SCHEMA`
  (`judgements` array, items require `id`+`decision`, decision enum pick/merge/rewrite); `JudgeItem` frozen
  dataclass (region, candidates, previous=None, problems=()); `label_for` (A..Z, `ValueError` at 26);
  `judge_messages(items, entries)` → [system, user] where the user message has `Glossary (binding)` /
  `Glossary (suggested)` sections only for locked/proposed entries with a hit in the chunk's sources
  (rejected never shown), then `Regions (reading order):` JSON with keys id/kind/source/candidates per item;
  `previous`/`problems` appear only on repair items.
- `src/omniscan/translate/judge.py` (new): `judge_regions(client, cfg, regions, runs, entries, *, clock)`.
  Deterministic side: `translatable` regions; per-region candidates in priority order (`prefer`∩runs in
  prefer order, then remaining runs alphabetically; whitespace-only texts skipped); unique candidates deduped
  by `normalize_line(text)` and labelled A..Z. A region is auto-picked when its candidates agree (or there
  is a single candidate) and at least one candidate violates no locked term — the first clean candidate in
  priority order wins (rationale "candidates agree" / "single candidate"). Always judged otherwise: candidates
  disagreeing, `always_judge` with ≥2 unique candidates, or every candidate violating a locked term.
  Requests: round 1 is chunked by `cfg.chunk_regions`; unresolved or still-violating items get exactly
  `cfg.max_repair_rounds` repair rounds, each item carrying `previous` ("" when the answer was invalid or
  missing) and `problems` (`missing binding term: X -> Y`). Reply parsing is the translation runner's
  tolerant path via `extract_list(content, "judgements")`; non-dict entries, unknown and duplicate ids are
  ignored (first wins); missing/invalid ids → unresolved. Resolutions: `pick` → the label's candidate text
  with that run id as source; `merge` → whitespace-collapsed text with every candidate's run id as sources;
  `rewrite` → collapsed text with `sources=[]`; rationale whitespace-collapsed and cut to 300 chars
  (non-string → ""). Unresolved regions fall back to the first violation-free candidate (rationale
  "judge failed", flag `judge_failed`, plus `glossary_violation` when it still breaks a term); resolved but
  still-violating answers are kept with flag `glossary_violation`. `JudgeStats` counters include token sums
  (`None` counts as 0) and `seconds` from the injected clock; client errors (including rate limit)
  propagate unchanged. Nothing else touches the network; no images are written.
- `src/omniscan/translate/judge_chapter.py` (new): `judge_chapter(client, paths, cfg, entries, *, run_ids,
  force)` → (`"done"|"skipped"`, FinalArtifact|None, JudgeStats|None). Existing `final.json` without `force`
  → skipped without calls; missing `ocr.json` → `FileNotFoundError` (checked before runs); every non-
  dot-prefixed `translations/*.json` is loaded as a `CandidateRun` into `{run_id: {region_id: text}}` (empty
  texts dropped); `run_ids` selects a subset, unknown ids raise `ValueError` naming the known runs; saves
  `FinalArtifact(judge_model=cfg.model, lines)` atomically and returns the stats.
- `src/omniscan/translate/parse.py`: added generic `extract_list(content, key)` (plain JSON → fenced blocks →
  `raw_decode` at every `{`; first dict whose `key` is a list wins; JSON-encoded strings decoded one level).
  `extract_translations` now delegates to `extract_list(content, "translations")` — behaviour unchanged, all
  pre-existing tests green.
- `src/omniscan/cli.py`: real `omniscan judge SERIES [--chapter/-c]… [--run/-r]… [--force]` replaces the
  stub; `"judge"` removed from `_STUB_COMMANDS`. No chapters → exit 2; invalid judge config → exit 2;
  per chapter: missing `ocr.json`/no runs → that chapter fails and the rest still run (exit 1 at the end);
  unknown `--run` → exit 2 immediately; `OllamaRateLimitError` → exit 3 immediately; other `OllamaError` →
  that chapter fails and the rest still run. Done line reports regions/judged/auto/untranslated/violations/
  seconds; an existing `final.json` prints `skipped`. Glossary entries come from the series db when it
  exists; without a db the judge runs with no glossary and the db is not created.
- `config/judge.toml` (new): the `[judge]` table with the documented defaults and
  `prefer = ["gemma4-31b-cloud", "gemma4-12b-local", "translategemma-12b-local"]`.
- Tests: `tests/unit/test_judge_config.py` (13), `test_judge_prompts.py` (9), `test_judge.py` (26),
  `test_judge_chapter.py` (9), `test_judge_cli.py` (13); appended 7 `extract_list` tests to
  `test_translate_parse.py`; removed `"judge"` from `STUB_COMMANDS` in `test_cli.py`.
- Docs: README status row → `working — omniscan judge (final.json)` and judge dropped from the stub table;
  USER_GUIDE gained a `### omniscan judge` subsection after the translate section and judge was removed from
  the stub list.
- No changes to `src/omniscan/core/**` or `src/omniscan/glossary/**`.

## Tests
```
uv run pytest tests/unit/test_judge_config.py tests/unit/test_judge_prompts.py tests/unit/test_judge.py \
  tests/unit/test_judge_chapter.py tests/unit/test_judge_cli.py tests/unit/test_translate_parse.py \
  tests/unit/test_cli.py tests/unit/test_docs.py -q
#   97 passed
uv run pytest -q            # 2272 passed, 3 warnings (pre-existing codec/anyio warnings)
uv run ruff format .        # 210 files unchanged
uv run ruff check .         # All checks passed
uv run pyright              # 0 errors, 0 warnings, 0 informations
```
All tests are CPU-only with hand-built artifacts and fake chat clients — no network, no real Ollama calls,
no raw manga in the repo.

## Deviations
- None against the card's file contract. Two notes:
  - The judge never asks the model about regions that are untranslated (no candidates) — they are marked
    `manual`/`untranslated` deterministically, per the card's "regions with no candidates are never judged".
  - The judge model `gemma4:31b-cloud` is already in `doctor.py`'s `REQUIRED_OLLAMA_MODELS`, so `omniscan
    doctor` covers it without changes.

## Questions
1. Queue wiring: the job store accepts `judge` as a stage name (`queue/store.py` `KNOWN_STAGES`, line 24),
   but `queue/executor.py`'s `STAGE_TABLE` has only `ingest` and `slice` (translate is likewise absent), so a
   queued judge job fails with "stage 'judge' is not implemented yet". I assumed a later card wires pipeline
   stages into the queue in one go — confirm, or point me at a card that wants `judge` wired now.
## Review addendum (director)
Answer to Question 1: correct — the queue's `STAGE_TABLE` gets all pipeline stages in one go with the `omniscan run` orchestrator card (R1); nothing to do here.
Rebased over C3 (stub-list conflict in README/USER_GUIDE resolved: `detect` and `judge` are both real now). Mutation check, 23 mutants over `judge.py` (always_judge threshold/ignored, agree check inverted, clean check inverted, auto-pick ignoring violations,
rationale boundary and cut length, alphabetical order reversed, `prefer` ignored, dedupe by raw text, repair round bound, repair counter, merge sources swapped, `judge_failed` flag, `violations_left` source, last-id-wins, empty merge text accepted,
no whitespace collapse, `cloud`/`think` ignored, empty repair problems, problem text format, fallback ignoring violations): **all 23 killed**.
Live check against the real Ollama daemon (3 profiles' candidate runs → `omniscan judge`) is still to do once builder slots are free (the Pro plan allows 3 concurrent requests).

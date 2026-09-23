# C5c — Story memory + glossary proposals from OCR

## What changed

### Part 1-2: story store + prompts
- **`src/omniscan/story/__init__.py`** (new) — module docstring only.
- **`src/omniscan/story/store.py`** (new) — `ChapterSummary` (frozen dataclass: `chapter`, `summary`, `model`, `updated_at`) and `SummaryStore` over a sqlite `chapter_summary` table (`chapter TEXT PRIMARY KEY, summary/model/updated_at NOT NULL`) living in the same `series.db` as `GlossaryStore` (WAL + `foreign_keys=ON`, `CREATE TABLE IF NOT EXISTS`). `set` upserts with `ON CONFLICT (chapter) DO UPDATE` and generates `updated_at` (UTC `isoformat()`) inside the call; `get`/`list` (ordered by chapter ascending)/`delete` (rowcount-based bool)/`close`/context manager.
- **`src/omniscan/story/prompts.py`** (new) — `MAX_SUMMARY_CHARS=800`, `SUMMARY_SCHEMA` (object requiring string `summary`), `SUMMARY_SYSTEM` (continuity writer: 3-5 sentences, third person, present tense, this chapter only, JSON only), `summary_messages(lines)` → system + one user message of newline-joined lines, and `parse_summary_reply` (fence-strip → JSON `summary` key → raw-prose fallback; every return truncated to `MAX_SUMMARY_CHARS`; whitespace-only JSON value falls back to the raw text; empty reply → `None`).

### Part 3: story summarize
- **`src/omniscan/story/summarize.py`** (new) — `DEFAULT_SUMMARY_MODEL="gemma4:31b-cloud"`; `chapter_final_lines(paths)` loads `final.json` + `ocr.json` and returns the judged English lines ordered by `(slice_index, reading_order, natural_key(region.id))` from ocr.json (drops unknown region ids and blank lines; `[]` if either artifact is missing); `summarize_chapter(client, model, lines)` (`None` for no lines, else a single chat call with `format=SUMMARY_SCHEMA`, `temperature=0.0`, `cloud=False`); `run_summarize(cfg, series, *, client, chapters=None, model=..., force=False)` → `SummarizeSummary(done, skipped_no_final, skipped_existing)` — creates the work dir, one `SummaryStore` for the whole call, skips already-summarized chapters to `skipped_existing` unless `force`, unusable replies count as `skipped_no_final`.

### Part 4: prompt threading
- **`src/omniscan/translate/prompts.py`** — `chat_json_messages(..., *, story_summary=None)`; when truthy, `"Story so far:\n{summary}"` is the **first** part (before glossary sections).
- **`src/omniscan/translate/judge_prompts.py`** — `judge_messages(..., *, story_summary=None)`, same insertion position.
- **`src/omniscan/translate/run.py`** — `run_profile(..., story_summary=None)` → `_run_chat_json` → `_request_translations` → `chat_json_messages(story_summary=...)`. The translategemma path is untouched.
- **`src/omniscan/translate/judge.py`** — `judge_regions(..., *, story_summary=None)` passed into the `judge_messages` closure.
- **`src/omniscan/translate/chapter.py`** — `translate_chapter(..., *, force=False, story_summary=None)`.
- **`src/omniscan/translate/judge_chapter.py`** — `judge_chapter(..., *, run_ids=None, force=False, story_summary=None)`.
- **`src/omniscan/pipeline/stages.py`** — new `_series_story_context(series, chapter, *, max_chapters=3)`: `None` if no db or the chapter isn't in `series.chapters()`; otherwise the last `max_chapters` prior chapters' summaries (gaps skipped) oldest-first, joined as `- <chapter>: <summary>`; never creates the db. `TranslateStage.run` and `JudgeStage.run` pass it through. `inputs()` unchanged.

### Part 5-6: glossary proposals
- **`src/omniscan/glossary/reference.py`** — `merge_into_store` gained a keyword-only `origin: Literal["llm", "reference"] = "reference"`; both `"reference"` literals replaced with `origin=origin`. Nothing else changed, so existing callers behave identically (existing tests pass unmodified).
- **`src/omniscan/glossary/proposal_prompts.py`** (new) — `PROPOSAL_SYSTEM` (single-language extraction: find Korean terms, strip particles, invent the English rendering, same type enum, same JSON shape) and `proposal_messages(lines)` (lines as one JSON list).
- **`src/omniscan/glossary/proposals.py`** (new) — `chapter_lines(sp, chapter)` (`source_text` of `translatable` regions, `[]` for missing ocr.json); `extract_proposals` (no lines → no request; one chat call with `format=TERMS_SCHEMA`, `temperature=0.0`) → `TermCandidate`s tagged with the chapter; `run_proposals` aggregates and keeps terms seen in ≥ `min_chapters` (`DEFAULT_MIN_CHAPTERS=2`) chapters, then `_merge_proposals` mirrors `reference.py::_merge`'s two-branch (real store / dry-run-on-missing-db) pattern with a local `_NullSink`, always `min_locks=1_000_000` and `origin="llm"` so proposals are written `status="proposed"` and never auto-locked; glossary.yaml re-exported only after a real write.

### Part 7: CLI
- **`src/omniscan/cli.py`** — `glossary_app.command("propose")` (`series`, repeatable `-c/--chapter`, `--model`, `--min-chapters`, `--dry-run`, `--json`; no-chapters → stderr + exit 2 before any client is built; `OllamaRateLimitError` → exit 3; `client.close()` in `finally`; text and JSON output shapes as tested) and a new `story_app` with `story summarize` (`-c`, `--model`, `--force`, `--json`; same no-chapters/rate-limit/close handling). Neither command touches the GPU lock or the VRAM manager — only `OllamaClient(cfg.ollama, get_secrets())`.

### Docs
- **`docs/USER_GUIDE.md`** — new `### `omniscan glossary propose`` section and `### `omniscan story summarize`` section (argument tables, skip/merge semantics, CPU-only notes, examples). `test_docs.py` stays green.

## Tests

All new/extended test files (all CPU-only; fake chat clients; tmp_path fixtures; no manga fixtures):

| File | Status |
|---|---|
| `tests/unit/test_story_store.py` (new, 6) | pass |
| `tests/unit/test_story_prompts.py` (new, 10) | pass |
| `tests/unit/test_story_summarize.py` (new, 10) | pass |
| `tests/unit/test_glossary_proposals.py` (new, 12) | pass |
| `tests/unit/test_story_cli.py` (new, 7) | pass |
| `tests/unit/test_glossary_cli.py` (new, 6) | pass |
| `tests/unit/test_translate_prompts.py` (+2), `test_judge_prompts.py` (+2) | pass |
| `tests/unit/test_translate_chapter.py` (+1), `test_judge_chapter.py` (+1), `test_pipeline_stages.py` (+4), `test_glossary_reference.py` (+1) | pass |

Acceptance commands and results:
- Card's targeted set (all 13 files above + `test_docs.py`): **146 passed**.
- Full suite `uv run pytest -q`: see final run below (includes pre-existing gpu/network-marked tests).
- `uv run ruff format . && uv run ruff check .`: clean (formatter reformatted the new files; 2 autofixable lint findings fixed).
- `uv run pyright`: no new errors (summary-only additions are typed).

GPU-freedom of both new commands is asserted directly: with a `device="cuda"` config and `omniscan.gpu.lock.acquire_gpu_lock` + `omniscan.gpu.groups.build_vram_manager` patched to raise `AssertionError`, both commands still exit 0.

## Deviations

1. **Collapsed WIP commits**: the card suggested three WIP commits; Parts 1-2 and 3-4 were completed together, so they landed as one commit (`e77252f C5c: WIP story memory (store, prompts, summarize, prompt threading)`), with the proposals work as the second WIP commit.
2. **Rate-limit handling** (`OllamaRateLimitError` → exit 3 in both new commands) is not spelled out in the card but mirrors the existing `cmd_reference` pattern and this project's Ollama weekly-limit reality.
3. **`tests/unit/test_glossary_cli.py` is a new file** — the card allowed this ("OR create a new test_glossary_cli.py if no existing file covers glossary_app"); no existing file covers `glossary_app`'s commands.
4. **Doc heading**: the card asked for "a new `### omniscan story` section"; the section is titled `### `omniscan story summarize`` (the only command on the sub-app) to match USER_GUIDE's per-command heading style.

## Questions

1. `DEFAULT_SUMMARY_MODEL` / proposal model default is `"gemma4:31b-cloud"` per the card. Should story summarize also support a local (non-cloud) model alias in config later, or is the `--model` flag enough?
2. `glossary propose` merges only at the end of the whole pass (after all chapters are scanned). If a run dies mid-way (e.g. rate limit at chapter 20/40), everything scanned so far is lost. A per-chapter or per-batch incremental merge would trade atomicity for progress retention — left as-is per the card, but worth a follow-up if large series hit rate limits.
3. Story summaries are fed to translate/judge automatically but only the last 3 prior chapters. For very continuity-heavy series a rolling "story so far" digest (summarize the summaries) might beat a fixed window — out of scope here, noting for a future card.
# C5c — Story memory (chapter summaries fed to translate/judge) + glossary proposals from OCR text

**Owner:** GLM builder · **Branch:** `C5c` · **Worktree:** `V:\OmniScan-wt\C5c` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read exactly these existing files (the card below reuses their patterns and, in three
places, their code directly):
- `src/omniscan/glossary/store.py` (`GlossaryStore` — the sqlite connection pattern every new store here copies)
- `src/omniscan/glossary/reference.py` and `reference_prompts.py` (extraction model call, `TermCandidate`,
  `aggregate`, `merge_into_store`, `Conflict`/`MergeReport` — Part 3 reuses `TermCandidate`, `aggregate` and
  `merge_into_store` unchanged except one new keyword argument)
- `src/omniscan/translate/prompts.py` (`chat_json_messages`, `translatable`, `source_text`) and
  `src/omniscan/translate/judge_prompts.py` (`judge_messages`) — both get one new optional parameter
- `src/omniscan/translate/run.py` (`run_profile`, `_run_chat_json`, `_request_translations`) and
  `src/omniscan/translate/judge.py` (`judge_regions`) — the call chain the new parameter is threaded through
- `src/omniscan/translate/chapter.py` (`translate_chapter`) and `translate/judge_chapter.py` (`judge_chapter`)
- `src/omniscan/pipeline/stages.py` (`_series_entries`, `TranslateStage`, `JudgeStage`) — mirror `_series_entries`
  with a new `_series_story_context` helper
- `src/omniscan/core/paths.py` (`SeriesPaths.chapters()`, `SeriesPaths.db`, `ChapterPaths.artifact`) and
  `src/omniscan/core/schemas.py` (`FinalArtifact`, `FinalLine`, `RegionsArtifact`, `Region`, `GlossaryEntry`, `utcnow`)
- `src/omniscan/cli.py` around `cmd_reference` (lines ~915-990, for CLI style — but this card's commands need **no
  GPU**: do not copy the GPU-lock/VRAM-manager part) and around `glossary_app` (lines ~1142-1206, the sub-app to add
  a command to)
- Test style to imitate: `tests/unit/test_glossary_store.py`, `test_glossary_reference.py`, `test_translate_prompts.py`,
  `test_judge_prompts.py`, `test_translate_chapter.py`, `test_judge_chapter.py`, `test_pipeline_stages.py`,
  `test_reference_cli.py`

## Why
Two related gaps (queue item C5c): (1) translate/judge only ever see the current chapter — nothing tells the model
what already happened in the story, so names/relationships/plot state can drift chapter to chapter; (2) the glossary
only grows through reference mode (needs an official English translation) — most series have none, so proper nouns
never reach the glossary until a human adds them by hand. This card adds an opt-in, human-reviewed pipe for both:
a per-chapter English summary stored in `series.db` and threaded into the translate/judge prompts of *later*
chapters, and an LLM pass over a chapter's own OCR text that proposes glossary terms (always `status="proposed"`,
never auto-locked — there is no official translation to agree with, so confidence is lower than reference mode).
Both are **manual, explicit commands** (`omniscan story summarize`, `omniscan glossary propose`), not extra stages
wired into `omniscan run` — an unconditional extra LLM call per chapter would violate this project's cost-conscious
default (see D5, "judge only on disagreement"). Reading an existing summary back into a later chapter's prompt is
free (no LLM call) and *is* automatic once a summary exists.

## Files you may create / modify
- Create: `src/omniscan/story/__init__.py` (docstring only), `story/store.py`, `story/prompts.py`, `story/summarize.py`
- Create: `src/omniscan/glossary/proposal_prompts.py`, `glossary/proposals.py`
- Modify: `src/omniscan/glossary/reference.py` — **only** `merge_into_store`: add one new keyword parameter (Part 3)
- Modify: `src/omniscan/translate/prompts.py` (`chat_json_messages`), `translate/judge_prompts.py` (`judge_messages`),
  `translate/run.py` (`run_profile`, `_run_chat_json`, `_request_translations`), `translate/judge.py` (`judge_regions`),
  `translate/chapter.py` (`translate_chapter`), `translate/judge_chapter.py` (`judge_chapter`) — each gets one new
  keyword-only parameter that defaults to `None`/unchanged behaviour; no other change to these files
- Modify: `src/omniscan/pipeline/stages.py` — add `_series_story_context`; `TranslateStage.run`/`JudgeStage.run` pass
  it through. Do **not** change `inputs()` on either stage: both already add `ctx.series.db` as an input file when it
  exists, which already covers the new table living in that same db.
- Modify: `src/omniscan/cli.py` — add a new `story_app` typer sub-app (`omniscan story summarize`) registered with
  `app.add_typer(story_app, name="story")`, and one new command `propose` on the existing `glossary_app`. Nothing
  else in `cli.py` changes.
- Tests (create): `tests/unit/test_story_store.py`, `test_story_prompts.py`, `test_story_summarize.py`,
  `test_glossary_proposals.py`, `test_story_cli.py`
- Tests (extend, append only): `test_translate_prompts.py`, `test_judge_prompts.py`, `test_translate_chapter.py`,
  `test_judge_chapter.py`, `test_pipeline_stages.py`, `test_glossary_reference.py` (the new `origin` parameter),
  `test_reference_cli.py` or a new `test_glossary_cli.py` if `propose` needs its own file — check which file already
  covers `glossary_app`'s commands and add there
- Docs: `docs/USER_GUIDE.md` (new `### omniscan story` section; a short addition to the glossary section for
  `propose`) — `tests/unit/test_docs.py` must stay green. `docs/reports/C5c.md` (create).
- Do not touch `src/omniscan/core/**`, `config/*.toml`, or any file not listed above.

---

## Part 1 — `story/store.py`: the chapter-summary table

```python
@dataclass(frozen=True, slots=True)
class ChapterSummary:
    chapter: str
    summary: str
    model: str
    updated_at: str  # ISO 8601 UTC, e.g. "2026-09-23T10:00:00+00:00" (from omniscan.core.schemas.utcnow().isoformat())

class SummaryStore:
    def __init__(self, db_path: Path) -> None: ...       # same connection pattern as GlossaryStore: sqlite3.connect,
                                                            # PRAGMA journal_mode=WAL, PRAGMA foreign_keys=ON,
                                                            # CREATE TABLE IF NOT EXISTS chapter_summary (
                                                            #   chapter TEXT PRIMARY KEY, summary TEXT NOT NULL,
                                                            #   model TEXT NOT NULL, updated_at TEXT NOT NULL), commit
    def set(self, chapter: str, summary: str, model: str) -> ChapterSummary: ...   # upsert (INSERT ... ON CONFLICT
                                                            # (chapter) DO UPDATE SET summary=excluded.summary,
                                                            # model=excluded.model, updated_at=excluded.updated_at);
                                                            # updated_at is generated inside `set`, not passed in
    def get(self, chapter: str) -> ChapterSummary | None: ...
    def list(self) -> list[ChapterSummary]: ...           # all rows, ORDER BY chapter (lexicographic — callers that
                                                            # need reading order sort themselves via SeriesPaths)
    def delete(self, chapter: str) -> bool: ...            # True iff a row was removed
    def close(self) -> None: ...                            # safe to call twice
    def __enter__(self) -> SummaryStore: ...
    def __exit__(self, *exc: object) -> None: ...
```
This is a **separate table in the same `series.db` file** `GlossaryStore` already uses (`SeriesPaths.db`) — do not
add anything to `GlossaryStore` itself (it is intentionally one table, one purpose).

## Part 2 — `story/prompts.py`: the summarization prompt

```python
MAX_SUMMARY_CHARS = 800

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

def summary_messages(lines: Sequence[str]) -> list[dict[str, str]]: ...
def parse_summary_reply(content: str) -> str | None: ...
```
- `summary_messages`: system message (write it — a chapter-summariser instruction: 3-5 sentences, third person,
  present tense, naming the characters/places involved and what changed in *this* chapter only, no meta-commentary,
  no chapter numbers, JSON-only reply shaped `{"summary":"..."}`, in the style of `EXTRACT_SYSTEM` in
  `reference_prompts.py`) plus one user message `f"Chapter lines (reading order):\n" + "\n".join(lines)`.
- `parse_summary_reply` must be tolerant the way `parse_terms_reply` is, plus one more fallback the terms parser
  doesn't need (chat models sometimes answer plain prose for a "write a summary" prompt instead of JSON):
  1. Strip the text; if it starts with `` ``` `` strip the leading/trailing triple-backtick fence (and a leading
     `json` line) before parsing.
  2. Try `json.loads`; if the result is a dict with a non-empty string `"summary"`, return it stripped and truncated
     to `MAX_SUMMARY_CHARS`.
  3. Otherwise (parse failure, or no usable `"summary"` key): return the stripped raw text truncated to
     `MAX_SUMMARY_CHARS`, or `None` if the stripped text is empty.

## Part 3 — `story/summarize.py`: computing and storing summaries

```python
DEFAULT_SUMMARY_MODEL = "gemma4:31b-cloud"

def chapter_final_lines(paths: ChapterPaths) -> list[str]:
    """The chapter's judged English lines in reading order; [] if final.json or ocr.json is missing."""
    ...

@dataclass(frozen=True, slots=True)
class SummarizeSummary:
    done: tuple[str, ...]            # chapters (re)computed this call
    skipped_no_final: tuple[str, ...]   # no final.json yet, or the model gave an unusable reply
    skipped_existing: tuple[str, ...]   # already had a summary and force=False

def run_summarize(
    cfg: Config, series: str, *, client: ChatClient,
    chapters: Sequence[str] | None = None,      # None = every chapter (SeriesPaths.chapters() order)
    model: str = DEFAULT_SUMMARY_MODEL,
    force: bool = False,
) -> SummarizeSummary: ...
```
- `chapter_final_lines(paths)`: load `final.json` (`FinalArtifact`) and `ocr.json` (`RegionsArtifact`); build
  `region_id -> (slice_index, reading_order, natural_key(region.id))` from the ocr regions (`natural_key` from
  `omniscan.core.paths`); keep `FinalLine`s whose `text.strip()` is non-empty and whose `region_id` is a known region;
  sort by that key; return the stripped texts. Missing `final.json` or `ocr.json` → `[]`.
- `run_summarize`: chapters default to `SeriesPaths.from_config(cfg, series).chapters()`. Create `sp.work_dir` if
  missing, open one `SummaryStore(sp.db)` for the whole call. For each chosen chapter, in order: skip to
  `skipped_existing` if `not force and store.get(chapter) is not None`; else compute `chapter_final_lines`; empty →
  `skipped_no_final`; else call `summarize_chapter(client, model, lines)` (a thin wrapper: `if not lines: return None;
  response = client.chat(model, summary_messages(lines), cloud=False, format=SUMMARY_SCHEMA,
  options={"temperature": 0.0}); return parse_summary_reply(response.content)` — put this function in
  `story/summarize.py`, it is the only place that calls the chat client); a usable summary → `store.set(...)`, add to
  `done`; `None` → add to `skipped_no_final` too (treat "no summary produced" the same as "nothing to summarize").

## Part 4 — threading the summary into translate/judge (read side, no LLM call, always cheap)

- `translate/prompts.py::chat_json_messages(regions, entries, *, story_summary: str | None = None)`: when
  `story_summary` is truthy, insert `f"Story so far:\n{story_summary}"` as the **first** element of `parts` (before
  the glossary sections); everything else unchanged. `None`/empty → identical output to today.
- `translate/judge_prompts.py::judge_messages(items, entries, *, story_summary: str | None = None)`: same insertion,
  same position, same rule.
- `translate/run.py`: add `story_summary: str | None = None` to `run_profile`, `_run_chat_json` and
  `_request_translations`; `_request_translations` passes it to `chat_json_messages(regions, entries,
  story_summary=story_summary)`. The `translategemma` path (`_run_translategemma`) is **not** touched — it does not
  use `chat_json_messages` and gets no story context (out of scope, same as it already doesn't use the glossary
  the same way).
- `translate/judge.py::judge_regions(..., *, story_summary: str | None = None, clock=...)`: the inner `chat()`
  closure passes `judge_messages(items, entries, story_summary=story_summary)`.
- `translate/chapter.py::translate_chapter(..., *, force: bool = False, story_summary: str | None = None)`: passes
  it to `run_profile`.
- `translate/judge_chapter.py::judge_chapter(..., *, run_ids=None, force: bool = False, story_summary: str | None =
  None)`: passes it to `judge_regions`.
- `pipeline/stages.py`:
  ```python
  def _series_story_context(series: SeriesPaths, chapter: str, *, max_chapters: int = 3) -> str | None:
      """Up to the last `max_chapters` prior chapters' stored summaries, oldest first; None when none exist."""
  ```
  Implementation: if `not series.db.is_file()`: `None`. `order = series.chapters()`; if `chapter not in order`:
  `None`. `prior = order[: order.index(chapter)]`. Open `SummaryStore(series.db)`, collect `(name, row.summary)` for
  every name in `prior` that has a stored row (skip the rest — gaps are fine), keep only the **last** `max_chapters`
  of those pairs (still oldest-first order), join as `"\n".join(f"- {name}: {summary}" for name, summary in pairs)`;
  empty list → `None`.
  `TranslateStage.run`: `story_summary = _series_story_context(ctx.series, ctx.paths.chapter)`, pass into every
  `translate_chapter(...)` call. `JudgeStage.run`: same helper call, pass into `judge_chapter(...)`.

## Part 5 — `glossary/proposal_prompts.py` + `glossary/proposals.py`: recurring proper nouns from OCR text alone

Reuse, do not redefine: `TERMS_SCHEMA`, `RawTerm`, `parse_terms_reply` from `glossary/reference_prompts.py`, and
`TermCandidate`, `aggregate`, `MergeReport`, `Conflict`, `GlossarySink` from `glossary/reference.py`.

`glossary/proposal_prompts.py`:
```python
def proposal_messages(lines: Sequence[str]) -> list[dict[str, str]]: ...
```
Write a new system prompt (`PROPOSAL_SYSTEM`) in the style of `EXTRACT_SYSTEM` but for the **single-language** case:
no official translation exists, so the model must both identify the term (Korean source, particle stripped, same
particle list as `EXTRACT_SYSTEM`) and *invent* its own best English rendering; same term `type` enum; same JSON
answer shape `{"terms":[{"source":"...","target":"...","type":"..."}]}` reusing `TERMS_SCHEMA` unchanged. User
message: `f"Chapter lines (reading order):\n" + json.dumps(lines, ensure_ascii=False, indent=1)`.

`glossary/proposals.py`:
```python
DEFAULT_PROPOSAL_MODEL = "gemma4:31b-cloud"
DEFAULT_MIN_CHAPTERS = 2   # an aggregated term must recur in at least this many chapters to be written at all

def chapter_lines(sp: SeriesPaths, chapter: str) -> list[str]:
    """The chapter's OCR'd source-language lines worth extracting from (no watermarks, non-empty text),
    reading order; [] when ocr.json is missing."""

def extract_proposals(client: ChatClient, model: str, chapter: str, lines: Sequence[str]) -> list[TermCandidate]:
    """[] for no lines; else one chat call (format=TERMS_SCHEMA, temperature 0.0, cloud=False) -> TermCandidate list."""

@dataclass(frozen=True, slots=True)
class ProposalSummary:
    chapters_scanned: tuple[str, ...]
    chapters_with_lines: int
    lines_scanned: int
    aggregated_above_threshold: int
    merge: MergeReport

def run_proposals(
    cfg: Config, series: str, *, client: ChatClient,
    chapters: Sequence[str] | None = None,          # None = every chapter, SeriesPaths.chapters() order
    model: str = DEFAULT_PROPOSAL_MODEL,
    min_chapters: int = DEFAULT_MIN_CHAPTERS,
    dry_run: bool = False,
) -> ProposalSummary: ...
```
- `chapter_lines`: same filter/sort/collapse as `translate/prompts.py::translatable`+`source_text` (not watermark,
  non-empty, sorted by `(slice_index, reading_order, natural_key(id))`, whitespace collapsed) — read `ocr.json` via
  `RegionsArtifact.load`.
- `run_proposals`: for each chosen chapter, get `chapter_lines`, accumulate `lines_scanned`; non-empty → run
  `extract_proposals`, accumulate into one candidate list, count as `chapters_with_lines`. `aggregated =
  [t for t in aggregate(candidates) if t.chapters >= min_chapters]`. Merge into the store **always as `origin="llm"`,
  never auto-locked**: call `merge_into_store(store_or_sink, aggregated, min_locks=<a value greater than any
  realistic chapter count, e.g. 1_000_000>, write=not dry_run, origin="llm")` (Part 6 adds the `origin` parameter).
  `dry_run` and "no `series.db` yet" follow exactly the same two-branch pattern as `glossary/reference.py::_merge`
  (real store when the db exists or a write is requested; `_NullSink`-style read-free path otherwise) — reuse
  `GlossaryStore`/`_merge`'s idea, write a small `_merge_proposals` helper mirroring it (`glossary/reference.py`'s
  `_NullSink` is private to that module — either import it or write an equivalent one-off local class; also
  re-export `glossary.yaml` via `export_yaml` after a real write, matching `reference.py::_merge`).

## Part 6 — `merge_into_store` gets one new keyword parameter (the only change to `reference.py`)

```python
def merge_into_store(
    store: GlossarySink,
    terms: Sequence[AggregatedTerm],
    *,
    min_locks: int = DEFAULT_MIN_LOCKS,
    write: bool = True,
    origin: Literal["llm", "reference"] = "reference",   # NEW — default preserves today's behaviour exactly
) -> MergeReport: ...
```
Every place inside the function body that currently writes the literal string `"reference"` into a `GlossaryEntry`'s
or updated entry's `origin` field must use the `origin` parameter instead. Nothing else in the function changes.
Every existing call site (`glossary/reference.py::_merge`, its own tests) omits the new parameter and must keep
behaving exactly as before (verified by `test_glossary_reference.py` staying green unmodified, plus one new test for
`origin="llm"`).

## Part 7 — CLI

`omniscan story summarize SERIES [-c/--chapter CHAPTER ...] [--model MODEL] [--force] [--json]` (new `story_app`,
registered `app.add_typer(story_app, name="story")`; no GPU lock, no VRAM manager — build the client with
`OllamaClient(cfg.ollama, get_secrets())` only, same as `cmd_reference` does, but skip everything about
`acquire_gpu_lock`/`build_vram_manager`). `-c` repeatable, narrows to those chapters (order as given); default is
every chapter of the series. Prints one line per outcome (`"story: <chapter>: summarized"` /
`"story: <chapter>: skipped (no final.json yet)"` / `"story: <chapter>: skipped (already summarized, use --force)"`)
and a final count line; `--json`: `{"series", "done": [...], "skipped_no_final": [...], "skipped_existing": [...]}`.
No chapters found for the series → stderr message, exit 2.

`omniscan glossary propose SERIES [-c/--chapter CHAPTER ...] [--model MODEL] [--min-chapters N] [--dry-run]
[--json]` (new command on the existing `glossary_app`, next to `list`/`export`/`import`). Same "no GPU" rule.
Prints a summary line (chapters scanned, lines scanned, terms proposed/updated, any conflicts — reuse
`MergeReport`'s fields the way `cmd_reference` prints `format_summary`, but you can write a shorter one-paragraph
version here since there is no matching/OCR-failure detail to report) plus, for `--dry-run`, the note
`"glossary: dry run — nothing written"`. `--json`: the `ProposalSummary` fields plus `merge: {"locked", "proposed",
"conflicts": [...], "rejected": [...]}` (locked will always be 0 here since proposals never auto-lock — assert that
in a test). No chapters found for the series → stderr message, exit 2.

## Acceptance tests (CPU only; fake `ChatClient`/`httpx` responses; temp dirs; no GPU, no `data/`, no network)
1. **`SummaryStore`**: `set` inserts then `get`/`list` reflect it; a second `set` on the same chapter updates in
   place (no duplicate row, `list()` still has one entry per chapter); `get` on an unknown chapter is `None`;
   `delete` returns `True`/`False` correctly; the table survives closing and reopening the same db file (a second
   `SummaryStore` on the same path sees prior rows); `list()` order is by chapter name ascending.
2. **`story/prompts.py`**: `parse_summary_reply` on clean JSON, on a ```` ```json ... ``` ```` fenced reply, on a
   fenced reply with no `json` tag, on a bare-prose reply with no JSON at all (returns the truncated prose, not
   `None`), on `'{"summary": "  "}'` (whitespace-only → falls back to the truncated raw text, not empty), on `""` →
   `None`; a summary longer than `MAX_SUMMARY_CHARS` is truncated in every branch. `summary_messages` produces a
   system + user message pair; the user content contains every input line.
3. **`chapter_final_lines`**: a chapter with matching `ocr.json`/`final.json` returns lines in
   `(slice_index, reading_order, id)` order regardless of the two files' own internal ordering; a `FinalLine` whose
   `region_id` is not in `ocr.json` is dropped; empty/whitespace-only lines are dropped; missing `final.json` or
   missing `ocr.json` → `[]`.
4. **`run_summarize`**: a fake `ChatClient` scripted to answer per chapter; three chapters — one gets summarized,
   one is `skipped_existing` (a prior `SummaryStore.set` before the call, `force=False`), one is
   `skipped_no_final` (no `final.json`); re-running with `force=True` re-summarizes the previously-existing one;
   a chapter whose fake reply parses to `None` lands in `skipped_no_final`; `chapters=[...]` restricts and preserves
   the given order (not `SeriesPaths.chapters()` order) — pick a case where they differ.
5. **Prompt threading (read side)**: `chat_json_messages(regions, entries, story_summary="line 1")` has `"Story so
   far:\nline 1"` as the first part, before any glossary section, before the regions section; `story_summary=None`
   and `story_summary=""` both produce byte-identical output to calling without the parameter at all (an existing
   `test_translate_prompts.py` case, unmodified, still passes). Same two assertions for `judge_messages`.
6. **`run_profile`/`translate_chapter`, `judge_regions`/`judge_chapter`**: a fake `ChatClient` that records the
   messages it was called with; calling with `story_summary="ctx"` shows `"Story so far:\nctx"` inside the recorded
   user message; calling without it (or `None`) is identical to the pre-existing tests for these functions (do not
   remove or weaken any existing assertion in `test_translate_chapter.py`/`test_judge_chapter.py`).
7. **`_series_story_context`**: no `series.db` → `None`; `chapter` not in `series.chapters()` → `None`; three prior
   chapters summarized, `max_chapters=3` (default) and a 5th chapter being translated → all three prior summaries
   joined, oldest first, each line `"- <chapter>: <summary>"`; more than `max_chapters` prior summaries exist →
   only the most recent `max_chapters` are kept (still oldest-of-those-first); a prior chapter with no summary is
   skipped without breaking the ordering of the ones that do have one; `TranslateStage.run`/`JudgeStage.run` (in
   `test_pipeline_stages.py`, extend the existing fakes) actually pass a non-`None` `story_summary` through to
   `translate_chapter`/`judge_chapter` when one is available, and `None` when `series.db` doesn't exist yet (must
   not create it — mirrors `_series_entries`'s own "never created here" rule).
8. **`merge_into_store` origin parameter**: every existing test in `test_glossary_reference.py` passes unmodified;
   one new test calls `merge_into_store(..., origin="llm")` and asserts the written/updated entries have
   `origin == "llm"` in both the "new entry" and "update an existing machine-proposed entry" branches; the default
   (no `origin` given) still writes `"reference"`.
9. **`glossary/proposals.py`**: `chapter_lines` matches `translate/prompts.py::translatable`'s filter/order on a
   shared synthetic `RegionsArtifact` fixture (watermark and empty-text regions excluded, same sort key); a fake
   `ChatClient` returning different terms per chapter — a term appearing in only 1 chapter is dropped
   (`min_chapters=2` default), a term in 2+ chapters survives into `merge`; `merge.locked == 0` always (nothing
   proposed by this path ever gets auto-locked, whatever `min_chapters` is); a term matching an existing `locked`
   glossary entry with a different target is reported as a `Conflict` and the entry is left untouched (same as
   reference mode); `dry_run=True` writes nothing (a second call sees the same "no db" or same entries as before);
   a series with no `ocr.json` anywhere → `ProposalSummary(lines_scanned=0, ...)`, `merge` empty, no crash.
10. **CLI** (`CliRunner`, temp config + library/work roots, `OllamaClient` replaced with a fake/monkeypatched
    factory the way `test_reference_cli.py` or `test_translate_cli.py` does — check which pattern that file uses):
    `story summarize SERIES` happy path (text output + `--json`), a series with no chapters → exit 2, `--force`
    re-summarizes, `-c` repeated narrows and orders; `glossary propose SERIES` happy path (text + `--json`,
    `merge.locked == 0` in the JSON), `--dry-run` writes nothing, a series with no chapters → exit 2. Neither
    command touches the GPU lock or VRAM manager (assert by not needing `omniscan.gpu.lock`/`omniscan.gpu.groups` at
    all in the test — if the implementation imports them, the test setup would need a fake GPU, which the card
    forbids: keep these commands GPU-free). `tests/unit/test_docs.py` and the whole CPU suite stay green.

## Out of scope
Auto-summarizing as part of `omniscan run` (would add an unconditional LLM call per chapter — a future decision, not
this card), summarizing anything other than the judged English text (no source-language or per-run summaries),
letting a proposed term from this pass ever auto-lock, a UI for reviewing proposals (existing `omniscan glossary
list --status proposed` already covers that), changing `GlossaryEntry`/`core/schemas.py` in any way, translategemma
prompt threading, per-series configuration of `max_chapters`/`min_chapters` (the constants/defaults are enough for
v1 — if you believe a `core/config.py` field is actually required, stop and write the question instead of editing
`core/**`).

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_story_store.py tests/unit/test_story_prompts.py tests/unit/test_story_summarize.py \
  tests/unit/test_glossary_proposals.py tests/unit/test_story_cli.py tests/unit/test_translate_prompts.py \
  tests/unit/test_judge_prompts.py tests/unit/test_translate_chapter.py tests/unit/test_judge_chapter.py \
  tests/unit/test_pipeline_stages.py tests/unit/test_glossary_reference.py tests/unit/test_docs.py -q
uv run pytest -q
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
`docs/reports/C5c.md`: Changes, Tests (commands + results), Deviations, Questions. **Commit early**
(`C5c: WIP story store+prompts` after Parts 1-2 pass, `C5c: WIP story summarize+wiring` after Parts 3-4, `C5c: WIP
glossary proposals` after Parts 5-6); the final commit is `C5c: story memory + glossary proposals from OCR`. You
have 150 tool calls in total — Parts 1-4 (story memory) matter more than Part 5 (proposals) if you are running low;
commit whichever parts are solid and write the rest under Questions rather than leaving anything half-done and
uncommitted. If anything is unclear: stop, write the question under Questions, commit, and end.

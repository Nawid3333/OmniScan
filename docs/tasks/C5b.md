# C5b — The judge: choose / merge / rewrite between candidate translations → `final.json`

**Owner:** GLM builder · **Branch:** `C5b` · **Worktree:** `V:\OmniScan-wt\C5b` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md`, `docs/benchmarks/translation-probe.md` (why parsing must be tolerant and `think:false` is needed) and `docs/tasks/B29.md`
(the translation runner this card mirrors) first. Then read `src/omniscan/translate/{prompts,parse,run,profiles,chapter}.py`,
`src/omniscan/translate/postcheck.py` and `agree.py` (C5a), `src/omniscan/llm/ollama.py` (`ChatResponse`, `OllamaRateLimitError`),
`src/omniscan/glossary/store.py`, and `Region`, `Candidate`, `CandidateRun`, `FinalLine`, `FinalArtifact`, `GlossaryEntry` in `src/omniscan/core/schemas.py`.
`cmd_translate` in `src/omniscan/cli.py` is the model for the CLI command.

## Goal
`omniscan judge SERIES` reads a chapter's `ocr.json`, all its candidate runs (`translations/*.json`) and the glossary, and writes `final.json`
(a `FinalArtifact`): one `FinalLine` per translatable region. To save tokens the judge model is asked **only about regions where the candidates
disagree or a locked glossary term is violated**; everything else is picked deterministically. Answers that break a locked term get one repair
round. Everything is tested with hand-built artifacts and a fake chat client — no network. (Story memory and glossary proposals are a later card.)

## Files you may create / modify
- `src/omniscan/translate/judge_config.py`, `judge_prompts.py`, `judge.py`, `judge_chapter.py` (create)
- `src/omniscan/translate/parse.py` (modify — ONLY add `extract_list(content: str, key: str) -> list[Any] | None`, the generalisation of
  `extract_translations` to any top-level key; make `extract_translations(content)` return `extract_list(content, "translations")`; existing tests stay green)
- `config/judge.toml` (create)
- `src/omniscan/cli.py` (modify — ONLY replace the `judge` stub with the real command and remove `"judge"` from `_STUB_COMMANDS`)
- `tests/unit/test_judge_config.py`, `test_judge_prompts.py`, `test_judge.py`, `test_judge_chapter.py`, `test_judge_cli.py` (create); `tests/unit/test_translate_parse.py`
  (modify — only append tests for `extract_list`); `tests/unit/test_cli.py` (modify — only what the stub removal requires)
- `README.md` (status row `judge` → `working — omniscan judge (final.json)`; drop `judge` from the "Not implemented yet" table) and `docs/USER_GUIDE.md`
  (a `### omniscan judge` subsection in the style of the others; fix the sentences that call `judge` a stub or say no stage writes `final.json`) — `tests/unit/test_docs.py` must stay green
- `docs/reports/C5b.md` (create)
Do not modify `src/omniscan/core/**` or `src/omniscan/glossary/**`.

## Part 1 — config (`judge_config.py`, `config/judge.toml`)
```python
class JudgeConfig(BaseModel):  # extra="forbid"
    model: str = "gemma4:31b-cloud"
    endpoint: Literal["local", "cloud"] = "local"
    think: bool | None = False
    temperature: float = 0.2            # 0.0 <= t <= 2.0
    chunk_regions: int = 20             # >= 1
    max_repair_rounds: int = 1          # >= 0
    agree_threshold: float = 0.9        # 0.0 <= x <= 1.0
    always_judge: bool = False          # True: ask the judge about every region that has >= 2 unique candidates
    prefer: list[str] = []              # run ids in priority order

def load_judge_config(paths: Sequence[Path]) -> JudgeConfig
def default_judge_paths() -> list[Path]
```
`config/judge.toml` (exactly): a `[judge]` table with the defaults above and `prefer = ["gemma4-31b-cloud", "gemma4-12b-local", "translategemma-12b-local"]`.
`load_judge_config` reads the existing files in order; a later file overrides the **keys** it sets (field-level merge); missing files are skipped; a bad TOML / unknown key / invalid
value raises `ValueError` naming the file. `default_judge_paths()` = `[DEFAULT_TOML.parent / "judge.toml", USER_TOML.parent / "judge.toml"]` (constants from `omniscan.core.config`).

## Part 2 — prompts (`judge_prompts.py`)
```python
JUDGE_SYSTEM: str
JUDGE_SCHEMA: dict[str, Any]

@dataclass(frozen=True, slots=True)
class JudgeItem:
    region: Region
    candidates: dict[str, str]          # label ("A", "B", ...) -> text, in label order
    previous: str | None = None         # repair rounds: the rejected final line
    problems: tuple[str, ...] = ()      # repair rounds: what was wrong

def label_for(index: int) -> str        # 0 -> "A", 1 -> "B", ... 25 -> "Z"; more than 26 candidates -> ValueError
def judge_messages(items: Sequence[JudgeItem], entries: Sequence[GlossaryEntry]) -> list[dict[str, str]]
```
- `JUDGE_SYSTEM` (copy verbatim, one Python string):
  ```
  You are the editor of an official English release of a Korean manhwa. For every numbered region you get the Korean source text and one or more candidate English translations labelled A, B, C. Decide per region: "pick" the best candidate unchanged; "merge" to combine the best parts of the candidates into one line; or "rewrite" to write a better line yourself when every candidate is wrong or unnatural. Judge the meaning against the Korean source, not by how many candidates agree. Write natural, idiomatic English suited to comic lettering: concise, in the character's voice, one continuous line without manual line breaks, no translator notes. Keep Korean honorific suffixes and titles romanized when they address a person (-nim, -ssi, hyung, noona, sunbae) unless that reads badly in English. For a region of kind "sfx" give a short English onomatopoeia. Entries under "Glossary (binding)" are mandatory: whenever a source term appears (with or without a particle such as 이/가/은/는/을/를/의), its target must appear in your English exactly as written. If a region has "problems", your earlier answer was rejected: fix exactly those problems (each missing binding target must appear verbatim in your text) and answer again. Answer with JSON only, in exactly this shape: {"judgements":[{"id":"r0001","decision":"pick","pick":"A","text":"","rationale":"short reason"}]} — for "pick" set "pick" to the label and leave "text" empty; for "merge" and "rewrite" set "text" to the final line and "pick" to ""; one entry for every input id, no extra ids, no commentary.
  ```
- `JUDGE_SCHEMA` = `{"type":"object","properties":{"judgements":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"decision":{"type":"string","enum":["pick","merge","rewrite"]},"pick":{"type":"string"},"text":{"type":"string"},"rationale":{"type":"string"}},"required":["id","decision"]}}},"required":["judgements"]}`.
- `judge_messages(items, entries)` → `[{"role":"system","content":JUDGE_SYSTEM}, {"role":"user","content":U}]`; `U` = parts joined by a blank line (`"\n\n"`), a part omitted when empty:
  1. `"Glossary (binding):\n" + lines` for the locked entries of `glossary_subset([i.region for i in items], entries)`, each `- {source} -> {target} ({type})`;
  2. `"Glossary (suggested):\n" + lines` for the proposed ones, same format;
  3. `"Regions (reading order):\n" + json.dumps(list_of_dicts, ensure_ascii=False, indent=1)` where each dict has, in this key order: `"id"`, `"kind"`, `"source"` (= `source_text(region)`),
     `"candidates"` (the label → text dict); plus `"previous"` and `"problems"` (list) ONLY when `item.previous is not None`. Items are used in the order given.

## Part 3 — the judge (`judge.py`)
```python
@dataclass(slots=True)
class JudgeStats:   # all int except seconds
    regions: int; judged: int; auto_picked: int; untranslated: int; violations_left: int
    requests: int; repair_requests: int; prompt_tokens: int; completion_tokens: int; seconds: float

def judge_regions(client: ChatClient, cfg: JudgeConfig, regions: Sequence[Region], runs: Mapping[str, Mapping[str, str]],
                  entries: Sequence[GlossaryEntry], *, clock: Callable[[], float] = time.perf_counter) -> tuple[list[FinalLine], JudgeStats]
```
`runs[run_id][region_id]` = the candidate text (already stripped; empty texts are simply absent). `ChatClient` is the Protocol from `translate/run.py`.
Definitions (exact):
1. `targets = translatable(regions)`; output lines are in that order.
2. **Candidates of a region:** the runs that have a text for it, ordered by priority — run ids listed in `cfg.prefer` first (in that order), then the others sorted alphabetically.
   **Unique candidates:** walk that order and keep a candidate only if its key is new, where key = `normalize_line(text)` or, when that is empty, the raw text; the kept candidates get the labels `A`, `B`, … in that order and remember the run id of their first occurrence.
3. **No candidate** → `FinalLine(region_id, text="", decision="manual", sources=[], rationale="no candidate", flags=["untranslated"])` (counts `untranslated`).
4. **Auto-pick** (no request) when `not cfg.always_judge or len(unique) < 2`, AND `candidates_agree([c.text for c in all candidates], cfg.agree_threshold)`, AND at least one candidate has no violations
   (`check_locked_terms(source_text(region), text, entries)` empty): pick the first such candidate in priority order → `FinalLine(region_id, text, decision="pick", sources=[run_id],
   rationale="candidates agree" if there are >= 2 candidates else "single candidate", flags=[])` (counts `auto_picked`). With `always_judge` and >= 2 unique candidates the region is judged even if they agree.
5. Every other region is **judged**. Round 1: chunks of `cfg.chunk_regions` (in order); per chunk
   `client.chat(cfg.model, judge_messages(items, entries), cloud=(cfg.endpoint == "cloud"), format=JUDGE_SCHEMA, options={"temperature": cfg.temperature}, think=cfg.think)`.
   Parse with `extract_list(content, "judgements")`; per item (tolerant: non-dict entries, ids not requested, duplicated ids → first wins, are ignored):
   - `decision == "pick"` and `pick` is one of the item's labels → text = that candidate, `sources=[its run id]`;
   - `decision == "merge"` and `text` non-empty after stripping → text = `" ".join(text.split())`, `sources` = the run ids of all its unique candidates;
   - `decision == "rewrite"` and `text` non-empty → same text normalisation, `sources=[]`;
   - anything else (unknown decision, unknown label, empty text, id missing from the reply) = **unresolved**.
   `rationale` = the reply's `rationale` when it is a string, whitespace-collapsed, cut to 300 characters, else `""`.
6. After a round, every resolved item whose text violates a locked term (`check_locked_terms`) and every unresolved item is a **repair** item while `repair_rounds_used < cfg.max_repair_rounds`:
   the next round sends only those items (chunked the same way) as `JudgeItem(region, candidates, previous=<rejected text or "">, problems=...)` where `problems` are
   `f"missing binding term: {v.source} -> {v.expected}"` for every violation, or `("the previous answer was invalid or missing",)` for an unresolved item. Each such request counts in `repair_requests`.
7. Final result of a judged region: resolved & no violation → `FinalLine(decision=<decision>, sources=..., text, rationale, flags=[])`; resolved but still violating after the rounds → the same with
   `flags=["glossary_violation"]` (counts `violations_left`); still unresolved → fall back to the first candidate without violations in priority order, else the first candidate:
   `decision="pick"`, `sources=[its run id]`, `rationale="judge failed"`, `flags=["judge_failed"]` plus `"glossary_violation"` when that candidate violates. Every judged region counts in `judged`.
8. `JudgeStats.requests` = all chat calls (first rounds + repairs); `prompt_tokens` / `completion_tokens` = sums of `prompt_eval_count` / `eval_count` (`None` counts as 0); `seconds` = `clock()` delta;
   `regions` = `len(targets)`. Any exception from `client.chat` propagates unchanged (nothing is swallowed, no partial file in this card).

## Part 4 — chapter level and CLI
```python
def judge_chapter(client: ChatClient, paths: ChapterPaths, cfg: JudgeConfig, entries: Sequence[GlossaryEntry], *,
                  run_ids: Sequence[str] | None = None, force: bool = False
                  ) -> tuple[Literal["done", "skipped"], FinalArtifact | None, JudgeStats | None]
```
- Output `paths.artifact("final.json")`; it exists and not `force` → `("skipped", None, None)` with no request.
- `ocr.json` missing → `FileNotFoundError("ocr.json missing — run the ocr stage first")`. Runs = every `translations/*.json` whose name does not start with `.` loaded as `CandidateRun`
  (`run.run_id` is the key; candidates with an empty stripped text are dropped); `run_ids` given → only those (an unknown id → `ValueError(f"unknown run {id!r} (known: a, b)")`);
  no run at all → `FileNotFoundError("no translation runs — run `omniscan translate` first")`.
- `FinalArtifact(judge_model=cfg.model, lines=lines).save(...)`; return `("done", artifact, stats)`.
- CLI `omniscan judge SERIES [--chapter/-c CHAPTER]... [--run/-r RUN_ID]... [--force]`: like `cmd_translate` (chapters default all; none → `judge: no chapters found for series 'X'` on stderr, exit 2; config
  `load_judge_config(default_judge_paths())` — invalid → message, exit 2; glossary `GlossaryStore(paths.db)` only when `paths.db.is_file()`; one `OllamaClient(cfg.ollama, get_secrets())`,
  `get_config()`/`get_secrets()`/`OllamaClient` used through the `omniscan.cli` namespace so tests can monkeypatch them). One line per chapter:
  `{series}/{chapter}: done ({n} regions, {j} judged, {a} auto, {u} untranslated, {v} violations left, {s:.1f}s)` or `{series}/{chapter}: skipped`. Missing `ocr.json` / no runs → message on stderr, counts as
  failure, continue; unknown `--run` → stderr, exit 2; `OllamaRateLimitError` → `judge: Ollama rate limit reached — re-run later` on stderr, **exit 3 immediately**; other `OllamaError` → stderr, failure, continue; exit 1 at the end if any failure.

## Acceptance tests (CPU; a fake client records calls and returns scripted `ChatResponse`s)
**Config:** 1. `config/judge.toml` loads to the documented defaults + `prefer`; 2. a later file overrides only the keys it sets, missing files are skipped; 3. invalid values (`temperature=3`, `chunk_regions=0`,
`agree_threshold=1.5`, unknown key, bad TOML) → `ValueError` naming the file.
**Parse:** 4. `extract_list(content, "judgements")` for plain JSON, a fenced block, prose around it, two objects of which only the second has the key, key not a list, garbage; `extract_translations` behaviour unchanged (existing tests green).
**Prompts:** 5. system message is `JUDGE_SYSTEM`; the user message has the binding/suggested sections only when non-empty (line format `- 성진 -> Seong-jin (person)`), the `Regions` JSON round-trips to the documented dicts with keys in
order, unescaped Korean, and `previous`/`problems` present only for repair items; 6. `label_for(0) == "A"`, `label_for(25) == "Z"`, `label_for(26)` raises `ValueError`.
**Judge — deterministic paths (no request):** 7. two runs with texts `"Get out!"` / `"get out"` → auto-picked, `sources` = the run that comes first in `prefer`, rationale `"candidates agree"`, zero requests; 8. one run only → `"single candidate"`;
9. a region without any candidate → `manual`, `untranslated`, empty text; 10. agreeing candidates that all violate a locked term → judged (a request is made); agreeing candidates of which the second satisfies the term → the second is picked without a request;
11. priority: without `prefer` the alphabetical order decides; with `prefer` the listed runs come first; identical normalised candidates are shown once (labels `A`, `B` only).
**Judge — requests:** 12. two disagreeing runs → one request whose messages equal `judge_messages(...)` for those items; request kwargs (`cloud` False/True by endpoint, `format == JUDGE_SCHEMA`, `options == {"temperature": 0.2}`, `think`);
13. reply `pick` `"B"` → `decision="pick"`, text and `sources` of B; `merge` → `sources` = all candidate run ids, text whitespace-collapsed; `rewrite` → `sources == []`; rationale collapsed and cut to 300 chars;
14. 45 judged regions with `chunk_regions=20` → 3 requests of 20/20/5 items in order; 15. a reply that omits an id, has an unknown label, an unknown decision or an empty merge text → that item is unresolved → one repair request containing only
those items with `problems == ("the previous answer was invalid or missing",)`, `previous == ""`; if it then answers correctly the result is complete and `repair_requests` counts it; 16. a resolved text missing a locked term → repair request with
`problems == ("missing binding term: 성진 -> Seong-jin",)` and `previous` set; fixed on the repair → `flags == []`; still wrong after `max_repair_rounds` → kept with `flags == ["glossary_violation"]` and `violations_left` counted;
17. `max_repair_rounds=0` → no repair requests at all; 18. still unresolved after the rounds → `decision="pick"`, the first violation-free candidate in priority order, `flags == ["judge_failed"]` (plus `"glossary_violation"` when it violates);
19. `always_judge=True` sends even agreeing regions when they have >= 2 unique candidates; 20. usage sums (`None` counts as 0), `seconds` from an injected clock, `regions`, `judged`, `auto_picked`, `untranslated`; 21. `OllamaRateLimitError` from the fake client propagates unchanged.
**Chapter:** 22. writes a loadable `final.json` (`judge_model`, lines in `translatable` order); second call → `("skipped", None, None)` with zero requests; `force=True` re-runs; `run_ids` selects runs; unknown run id → `ValueError`; missing `ocr.json` → `FileNotFoundError`; no runs → `FileNotFoundError`; dot-prefixed partial run files are ignored.
**CLI:** 23. output lines as specified, files written; unknown series/chapters → exit 2; missing `ocr.json` for one of two chapters → exit 1 while the other still runs; rate limit → exit 3 and the message; unknown `--run` → exit 2; no `series.db` → runs without glossary and does NOT create the db;
`tests/unit/test_cli.py` and `test_docs.py` green. 24. The whole suite, ruff and pyright are green.

## Out of scope
Story memory, glossary proposals, condensing/overflow rewrites, partial (resumable) judge files, parallel requests, streaming, changing `translate/run.py`, the web UI (the Translation view already reads `final.json`), running against the real Ollama daemon (the director does the live check).

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_judge_config.py tests/unit/test_judge_prompts.py tests/unit/test_judge.py tests/unit/test_judge_chapter.py tests/unit/test_judge_cli.py tests/unit/test_translate_parse.py tests/unit/test_cli.py tests/unit/test_docs.py -q
uv run pytest -q
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
Write `docs/reports/C5b.md` (Changes, Tests, Deviations, Questions) and commit `C5b: judge and omniscan judge`. If anything above is unclear: stop, write the question under Questions, commit what you have, and end.

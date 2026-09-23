# TL1 — Language-aware LLM prompts (Chinese and Japanese series are currently told their text is Korean)

**Owner:** GLM builder · **Branch:** `TL1` · **Worktree:** `V:\OmniScan-wt\TL1` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read every file listed under "Files you may create / modify" before changing any.

## Why (found by the director on real data)
OCR supports Korean, Chinese and Japanese (`cfg.ocr.lang`, written into every `Region.lang`), but every LLM prompt
hard-codes Korean: "a Korean manhwa", "Translate ... from Korean", "Korean (ko) to English (en)", Korean honorifics
(-nim, -ssi, hyung) and Korean particles (이/가/은/는/을/를/의). A Japanese or Chinese chapter is therefore sent to
every model with the claim that its text is Korean. This card makes each prompt name the region's real source
language. **Korean prompts must stay byte-for-byte identical** (they are tuned and measured; see
`docs/benchmarks/translation-probe.md`).

## The design (decided by the director — implement this shape)
1. New module `src/omniscan/translate/languages.py`:
```python
@dataclass(frozen=True, slots=True)
class SourceLanguage:
    """How the prompts name one source language."""
    code: str            # "ko", "zh", "ja", "en"
    name: str            # "Korean", "Chinese", "Japanese", "English"
    work: str            # "manhwa", "manhua", "manga", "comic"
    honorifics: str      # the whole honorifics sentence incl. its trailing space, or "" (see Definitions)
    particle_hint: str   # e.g. " (with or without a particle such as 이/가/은/는/을/를/의)", or ""

def source_language(code: str) -> SourceLanguage:
    """The SourceLanguage for a Region.lang code; an unknown code raises ValueError."""

def region_language(regions: Sequence[Region]) -> str:
    """The most common `lang` among `regions` (ties: the first region's), "ko" when `regions` is empty."""

def chapter_language(paths: ChapterPaths) -> str:
    """region_language of the chapter's ocr.json regions; "ko" when ocr.json is missing."""
```
2. Every prompt becomes a function of the language. The existing module-level constants stay and **equal the
   Korean rendering exactly** (other code and tests import them):
   - `translate/prompts.py`: `chat_json_system(lang: str) -> str`, `CHAT_JSON_SYSTEM = chat_json_system("ko")`;
     `translategemma_template(lang: str) -> str`, `TRANSLATEGEMMA_TEMPLATE = translategemma_template("ko")`;
     `translategemma_prompt(text: str, lang: str = "ko") -> str`. `chat_json_messages` keeps its signature and uses
     `chat_json_system(region_language(regions))`.
   - `translate/judge_prompts.py`: `judge_system(lang: str) -> str`, `JUDGE_SYSTEM = judge_system("ko")`;
     `judge_messages` keeps its signature and uses `judge_system(region_language([i.region for i in items]))`.
   - `story/prompts.py`: `summary_system(lang)`, `SUMMARY_SYSTEM = summary_system("ko")`,
     `summary_messages(lines, lang: str = "ko")`.
   - `glossary/proposal_prompts.py`: `proposal_system(lang)`, `PROPOSAL_SYSTEM = proposal_system("ko")`,
     `proposal_messages(lines, lang: str = "ko")`.
   - `glossary/reference_prompts.py`: `extract_system(lang)`, `EXTRACT_SYSTEM = extract_system("ko")`,
     `terms_messages(lines, lang: str = "ko")`.
3. Callers pass the language:
   - `translate/run.py::_prompt_for` → `translategemma_prompt(..., region.lang)`. (`_request_translations` needs no
     change: `chat_json_messages` derives the language from its regions.)
   - `story/summarize.py`: `summarize_chapter(client, model, lines, lang: str = "ko")`; `run_summarize` passes
     `chapter_language(sp.chapter(chapter))`.
   - `glossary/proposals.py`: `extract_proposals(client, model, chapter, lines, lang: str = "ko")`; `run_proposals`
     passes `chapter_language(sp.chapter(chapter))`.
   - `glossary/reference.py`: the function that calls `terms_messages` gains `lang: str = "ko"` the same way, and its
     caller passes `chapter_language(...)` of the **source** chapter (the raw, non-English one).

## Definitions (no interpretation needed)
- Each `*_system(lang)` is the current Korean text with every language-specific piece taken from `SourceLanguage`:
  "Korean" → `name`, "manhwa" → `work`, "(ko)" → `f"({code})"`, the honorifics sentence → `honorifics`, the particle
  parenthesis → `particle_hint`. Nothing else in the wording changes.
- Values: `ko`: Korean / manhwa / honorifics = the current sentence ("Keep Korean honorific suffixes and titles
  romanized when they address a person (-nim, -ssi, hyung, noona, sunbae) unless that reads badly in English. ") /
  particle_hint = the current " (with or without a particle such as 이/가/은/는/을/를/의)". `ja`: Japanese / manga /
  "Keep Japanese honorific suffixes romanized when they address a person (-san, -kun, -chan, -sama, senpai) unless
  that reads badly in English. " / " (with or without a particle such as は/が/を/に/の)". `zh`: Chinese / manhua /
  "" / "". `en`: English / comic / "" / "".
- Where the Korean prompt names the language twice in one sentence ("from Korean into natural ... English"), both
  become `name`. Watch the exact spaces around removed sentences: for `zh` the text must not contain a double space
  or a dangling "  " where the honorifics sentence was.
- `region_language` counts `Region.lang` values; a `Region` always has one (default "ko").

## Files you may create / modify
- `src/omniscan/translate/languages.py` (create)
- `src/omniscan/translate/prompts.py`, `src/omniscan/translate/judge_prompts.py`, `src/omniscan/translate/run.py`
  (only `_prompt_for`)
- `src/omniscan/story/prompts.py`, `src/omniscan/story/summarize.py`
- `src/omniscan/glossary/proposal_prompts.py`, `src/omniscan/glossary/proposals.py`,
  `src/omniscan/glossary/reference_prompts.py`, `src/omniscan/glossary/reference.py` (only the `terms_messages` call
  path)
- `tests/unit/test_translate_languages.py` (create), and extend the existing tests of each module above
- `docs/reports/TL1.md` (create)
Anything not listed is off-limits, especially `src/omniscan/core/**` and `src/omniscan/glossary/match.py`.

## Acceptance tests (CPU only, no network)
1. **Korean unchanged**: for each of the five prompts, `*_system("ko")` (and `translategemma_template("ko")`) equals
   the constant as it was before this card — pin by comparing against a literal copy of today's text in the test
   (copy it from `git show main:<file>`), not against the new constant.
2. For `ja` and `zh`, each system prompt contains `name` and `work`, contains no "Korean"/"manhwa"/"이/가", and has no
   double space. `ja` contains "-san"; `zh` contains no honorifics sentence.
3. `region_language`: majority wins; a tie goes to the first region's lang; empty → "ko".
4. `chapter_language`: reads a temp ocr.json with `ja` regions → "ja"; missing file → "ko".
5. `chat_json_messages` with `zh` regions uses the `zh` system prompt; `judge_messages` with `ja` items uses the
   `ja` judge prompt; `run_profile` with a translategemma profile and a `ja` region sends
   `translategemma_template("ja") + text`.
6. `run_summarize` / `run_proposals` on a temp series whose ocr.json regions are `zh` send the `zh` system prompt
   (fake chat client that records messages, following those test files' existing doubles).
7. `uv run pytest -q -m "not gpu"` stays green (including `tests/unit/test_e2e_synthetic.py` and `test_docs.py`).

## Out of scope
- `glossary/match.py`'s Korean particle stripping (harmless on Chinese/Japanese text; a later card).
- Changing any Korean wording, the answer schemas, the model profiles or `config/**`.
- Running real models (the director measures after the merge).

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_translate_languages.py tests/unit/test_translate_prompts.py tests/unit/test_translate_run.py -q
uv run pytest -q -m "not gpu"
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
`docs/reports/TL1.md`: Changes, Tests (commands + results), Deviations, Questions. Commit early
(`TL1: WIP languages module + translate prompts`), final commit `TL1: language-aware LLM prompts`. You have 150 tool
calls in total. If anything is unclear: stop, write the question under Questions, commit, and end.

# TL1 — Language-aware LLM prompts

Branch: `TL1` · Status: done · All acceptance tests pass, ruff and pyright clean.

## Changes

New module:

- `src/omniscan/translate/languages.py` — `SourceLanguage` frozen dataclass (`code`, `name`, `work`,
  `honorifics`, `particle_hint`) with the table for `ko`/`ja`/`zh`/`en` exactly as the card defines
  them, plus `source_language(code)` (unknown code → `ValueError`), `region_language(regions)`
  (majority wins, tie → the first region's lang via `max` over a `Counter` of first-occurrence
  order, empty → `"ko"`), and `chapter_language(paths)` (`region_language` of the chapter's
  ocr.json regions, `"ko"` when the file is missing).

Prompt modules — each system prompt became a function of the language; the module-level constant is
now the `"ko"` rendering (byte-for-byte identical to before, pinned by literal copies of the
pre-card text taken from `git show main:<file>` in `tests/unit/test_translate_languages.py`):

- `translate/prompts.py`: `chat_json_system(lang)` + `CHAT_JSON_SYSTEM = chat_json_system("ko")`;
  `translategemma_template(lang)` + `TRANSLATEGEMMA_TEMPLATE = translategemma_template("ko")`;
  `translategemma_prompt(text, lang="ko")`; `chat_json_messages` keeps its signature and now uses
  `chat_json_system(region_language(regions))`.
- `translate/judge_prompts.py`: `judge_system(lang)` + `JUDGE_SYSTEM = judge_system("ko")`;
  `judge_messages` derives the language from `region_language([item.region for item in items])`.
- `story/prompts.py`: `summary_system(lang)` + `SUMMARY_SYSTEM = summary_system("ko")`;
  `summary_messages(lines, lang="ko")`.
- `glossary/proposal_prompts.py`: `proposal_system(lang)` + `PROPOSAL_SYSTEM = proposal_system("ko")`;
  `proposal_messages(lines, lang="ko")`.
- `glossary/reference_prompts.py`: `extract_system(lang)` + `EXTRACT_SYSTEM = extract_system("ko")`;
  `terms_messages(lines, lang="ko")`.

Callers:

- `translate/run.py::_prompt_for` passes `region.lang` to `translategemma_prompt`.
- `story/summarize.py`: `summarize_chapter(..., lang="ko")`; `run_summarize` passes
  `chapter_language(sp.chapter(chapter))`.
- `glossary/proposals.py`: `extract_proposals(..., lang="ko")`; `run_proposals` passes
  `chapter_language(sp.chapter(chapter))`.
- `glossary/reference.py`: `extract_candidates(..., lang="ko")`; `run_reference` passes
  `chapter_language(sp.chapter(match.a))` — the raw (non-English) side's ocr.json.

Tests:

- `tests/unit/test_translate_languages.py` (new): the six Korean byte-identity pins against
  literals copied from `main`; ja/zh rendering checks for all five system prompts + the
  translategemma template (name/work present, no "Korean"/"manhwa"/"이/가", no double space, ja
  honorifics sentence, zh without it); `SourceLanguage` table values; unknown code raises;
  `region_language` majority/tie/empty; `chapter_language` reads a temp ocr.json / missing file.
- Extended the existing tests of every touched module: `chat_json_messages` with `zh` regions,
  `judge_messages` with a `ja` item, `run_profile` translategemma with a `ja` region (asserts
  `translategemma_template("ja") + text` is sent), `run_summarize` and `run_proposals` on temp
  series whose ocr.json regions are `zh` (fake chat clients recording messages, following those
  files' existing doubles), `extract_proposals`/`extract_candidates` with an explicit language,
  `summarize_chapter` default stays `ko`.

## Tests (commands + results)

```
uv run pytest tests/unit/test_translate_languages.py tests/unit/test_translate_prompts.py \
  tests/unit/test_translate_run.py tests/unit/test_judge_prompts.py tests/unit/test_story_prompts.py \
  tests/unit/test_story_summarize.py tests/unit/test_glossary_proposals.py tests/unit/test_glossary_reference.py -q
→ exit code 0 (123 passed)

uv run pytest -q -m "not gpu"
→ exit code 0 (full CPU suite incl. test_e2e_synthetic.py and test_docs.py; only the pre-existing
  XFAIL tests/unit/test_hw_hf_mutation_gaps.py::test_hf_marker_that_is_a_json_list_is_corrupt)

uv run ruff format . && uv run ruff check .
→ "372 files left unchanged", "All checks passed!"

uv run pyright
→ 0 errors, 0 warnings, 0 informations
```

Note: in this Git Bash environment the pytest counts line does not reach the tool output, so
results were confirmed via pytest's exit code (`pytest.main(...) → 0`) in addition to the progress
dots and short summary.

## Deviations

1. Acceptance test 2 read literally ("`ja` contains '-san'") cannot hold for `summary_system`,
   `proposal_system` and `extract_system`: their Korean originals contain no honorifics sentence at
   all, and the card forbids any other wording change, so their ja renderings have nowhere for
   "-san" to come from. Implemented as: the prompts whose Korean original carries the honorifics
   sentence (`chat_json_system`, `judge_system`) must contain "-san" for `ja`; all five must not
   contain the honorifics sentence for `zh`. No source-code deviation — only the test expresses the
   acceptance criterion per-prompt instead of uniformly.
2. The particle enumeration `(이, 가, 은, 는, 을, 를, 의, 도, 아, 야, 에, 에게, 에서)` inside
   `proposal_system`/`extract_system` was deliberately left language-independent. Substituting
   `particle_hint` there would have changed the byte-identical "ko" rendering (its Korean text is
   "with any attached particle (...) removed", not the "with or without a particle such as (...)"
   form `particle_hint` holds), and the enumeration describes `glossary/match.py`'s Korean particle
   stripping, which the card puts out of scope. For Chinese/Japanese chapters the enumeration is
   inert (their text contains no hangul). See Questions.
3. The `run_reference` caller wiring is not covered by a `run_reference`-level integration test:
   `tests/unit/test_reference_cli.py` (where those tests live) is not on this card's allowed file
   list. Covered instead by the new `extract_candidates(lang=...)` test; the call pattern is
   identical to the tested `run_summarize`/`run_proposals` wiring.

## Questions

1. Should the `(이, 가, ... 에서)` particle enumeration in the proposal/extract prompts become
   language-aware together with `glossary/match.py`'s Korean particle stripping (later card)? For
   zh/ja it is currently harmless but also useless — for Japanese a は/が/を analogue would matter
   once term matching handles non-Korean sources.
2. The translategemma template's English side stays fixed ("...to English (en) translator...") for
   every source language, which is correct as long as the target is always English. Confirm the
   director wants no target-language parameter (card implies target is always English).
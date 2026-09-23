# GL2 — Glossary term matcher: language-aware particle stripping (ja support, zh/en no-ops)

Branch `GL2`. Card: `docs/tasks/GL2.md` (NEXT.md item 4, flagged by TL1's builder).

## Why
`find_terms` stripped a fixed set of Korean particles (조사) after every matched term, unconditionally.
For Chinese that is already correct — Chinese has no attached particles, so the Korean patterns never
match. For Japanese it is a real gap: Japanese attaches its own case particles after nouns exactly the
way Korean does, so a locked glossary term followed by は/が/を/... matched with `particle=None` and
`Match.end` stopping before the particle — harmless for plain locked-substring checks, wrong for
anything that inspects `Match.particle`/`Match.end` (future UI highlight, `substitute_binding`'s
replacement span). `find_terms` now takes the text's source language and selects the particle set from
`PARTICLES_BY_LANG`; Chinese and English deliberately get `()` (documented correct behaviour, not a gap).

## Changes
- `src/omniscan/glossary/match.py` — added `JAPANESE_PARTICLES` (compound forms first) and
  `PARTICLES_BY_LANG` right after `KOREAN_PARTICLES` (kept, still exported); deleted the module-level
  `_PARTICLES_LONGEST_FIRST` constant, replaced by `_particles_longest_first(lang)` (unknown `lang` →
  `()`, never raises); `find_terms(text, entries, lang)` — required positional-or-keyword `lang`, particle
  set computed once per call; docstring gains the `lang` line. Module docstring updated from "Korean
  source text" to language-aware wording (see Deviations).
- `src/omniscan/glossary/__init__.py` — `JAPANESE_PARTICLES` and `PARTICLES_BY_LANG` added to the import
  line and `__all__`.
- `src/omniscan/translate/prompts.py` — `glossary_subset` derives `lang = region_language(regions)` once
  and passes it (own signature unchanged); `substitute_binding` gains a required third parameter `lang`,
  internal call updated, docstring gains the one-clause `lang` mention.
- `src/omniscan/translate/run.py` — the one `substitute_binding(...)` call in `_prompt_for` passes
  `region.lang`.
- `src/omniscan/translate/postcheck.py` — `check_locked_terms` gains a required fourth parameter `lang`
  (internal call updated, docstring gains the same one-clause `lang` mention as `substitute_binding`);
  `check_regions` passes `region.lang` (own signature unchanged).
- `src/omniscan/web/app.py` — `get_glossary_hits` calls `find_terms(region.text, entries, region.lang)`.
- `src/omniscan/translate/judge.py` — **not in the card's file list** (see Deviations): all five
  `check_locked_terms` call sites (`_auto_line`, `_repair_triples`, `_final_line` ×3) now pass
  `analysis.region.lang`, the same mechanical pattern the card prescribes everywhere else; no other edits.
- `tests/unit/test_glossary_match.py` — every existing `find_terms` call gains `lang="ko"` (all 10,
  including `test_find_terms_requires_entry_id`); new tests: Japanese simple `は` (the pinned example:
  `ユジンは学校に行った` → `end=4`, `particle="は"`), compound `とは` (`ユジンとは違う` → `end=5`,
  `particle="とは"`, preferred over its component `と`), Japanese no-particle before space/end of string
  (port of the Korean test to `lang="ja"`), Chinese no-op (`ユジンは行った` with `lang="zh"`: `particle=None`,
  `end == term_end`), English no-op, unknown `lang="xx"` behaves like zh/en and does not raise. The
  `_particles_longest_first` preference is covered through `find_terms`'s observable behaviour (the とは
  pin), per the card's option 5.
- `tests/unit/test_translate_prompts.py` — `substitute_binding` calls gain `lang="ko"`; new
  `test_substitute_binding_japanese_particle_stays_put` (simple `は` and compound `とは`: the replacement
  span does not eat or leave behind part of the particle); new `test_glossary_subset_japanese_particle_hit`
  with `Region(lang="ja", ...)`.
- `tests/unit/test_translate_postcheck.py` — `region()` helper gains a `lang` param (mirroring the
  prompts test file); every `check_locked_terms` call gains `lang="ko"`; new Japanese tests for
  `check_locked_terms` and for `check_regions` (region's own `lang="ja"` flows through).
- `tests/unit/test_translate_run.py`, `tests/unit/test_filter_glossary_importer_mutation_gaps.py` —
  **not in the card's file list** (see Deviations): their existing calls break with the new required
  parameters, so they gained `lang`/`lang="ko"` only.
- `docs/reports/GL2.md` — this file.

Untouched on purpose: `src/omniscan/core/**`, `src/omniscan/translate/languages.py` (`region_language`
still returns plain `str`), `docs/tasks/*.md` / `docs/reports/*.md` of other cards (they document the old
signatures historically), `docs/OPEN_QUESTIONS.md` (not in this card's file list).

## Tests
All commands via `uv run --no-sync ...` — see Deviations 6.

- `uv run --no-sync pytest tests/unit/test_glossary_match.py -p no:warnings` → **16 passed** (10 existing
  with `lang="ko"`, byte-identical expected `Match` values, + 6 new language tests).
- `uv run --no-sync pytest -m "not gpu" -p no:warnings` (run again after `ruff format`) →
  **3952 passed, 26 deselected (gpu), 1 xfailed** — the xfail
  (`test_hw_hf_mutation_gaps.py::test_hf_marker_that_is_a_json_list_is_corrupt`) is pre-existing/expected.
  Includes `tests/unit/test_web_app.py` glossary-hits route tests (fixtures are Korean, default
  `lang="ko"`; the route just passes `region.lang` through, unchanged expectations), the judge tests
  (exercising `judge.py`'s five updated call sites), and `tests/unit/test_docs.py` (checked it exists
  first — README/USER_GUIDE CLI checks, unaffected).
- `uv run --no-sync ruff format .` → 3 files reformatted (wrapping of longened lines in
  `test_glossary_match.py`, `test_translate_postcheck.py`, `judge.py`), 373 unchanged.
- `uv run --no-sync ruff check .` → **All checks passed!**
- `uv run --no-sync pyright` → **0 errors, 0 warnings, 0 informations**.

## Deviations from the card
1. **`src/omniscan/translate/judge.py` was modified although it is not in the card's file list.** The
   card makes `check_locked_terms`'s `lang` a *required* fourth parameter, but the card's call-site list
   covers only `check_regions` — `judge.py` calls `check_locked_terms` five times directly (grep for
   `check_locked_terms(` in `src/` finds `postcheck.py`, `prompts.py`, `run.py`, `web/app.py` for
   `find_terms`/`substitute_binding`, but `judge.py:234/293/312/314/317` for `check_locked_terms`).
   Leaving it alone would have broken pyright and the judge tests, so the card's definition of done was
   unreachable without it. Every site has `analysis.region` in scope; all five pass
   `analysis.region.lang` — no design decision was made, only the pattern the card itself prescribes
   (`region.lang`) applied. If the director prefers judge.py in a separate card, the alternative is to
   revert judge.py **and** make `lang` optional in `check_locked_terms` (the required-parameter decision
   is what forces judge.py into this diff).
2. **`tests/unit/test_translate_run.py` and `tests/unit/test_filter_glossary_importer_mutation_gaps.py`
   were modified although not listed** — their existing calls to `substitute_binding` / `find_terms`
   fail to type-check and run with the new required parameters. The card itself instructs grepping
   `tests/unit/` for the right file instead of creating duplicates, so these are mechanical
   `lang="ko"` / `reg.lang` additions only (one line each + two lines).
3. `match.py`'s module docstring changed from "…over Korean source text…" to language-aware wording —
   it would otherwise be inaccurate after this change. Trivial, same file.
4. `check_locked_terms`' docstring gained the same one-clause `lang` mention the card prescribes for
   `substitute_binding` (consistency; the card was silent on it).
5. Single final commit instead of the suggested early `GL2: WIP particle-by-language table` commit —
   the card completed in one session, so a separate WIP commit would be empty.
6. All commands ran with `uv run --no-sync`: plain `uv run` fails in this worktree right now because
   uv tries to reinstall the project and cannot replace `.venv\Scripts\omniscan.exe`
   ("Der Prozess kann nicht auf die Datei zugreifen … os error 32") — something outside this session is
   holding that file. The venv's editable install points at this worktree's `src/`, so the tests
   exercise the edited code (verified: the new ja tests import `JAPANESE_PARTICLES` live). The
   definition-of-done commands are identical apart from `--no-sync`.

## Questions
1. Please ratify deviation 1 (judge.py in this diff) or say the word and I will revert judge.py and
   switch `check_locked_terms`'s `lang` to a keyword-only default in a follow-up commit on this branch.
2. The locked `omniscan.exe` (deviation 6): is something (a `serve` instance, another builder session)
   running from `V:\OmniScan-wt\GL2\.venv`? Once the holder exits, plain `uv run pytest` should work
   again with no further changes.
3. `docs/tasks/B10.md`, `B29.md`, `C5a.md` still document the old two/three-arg signatures. I left all
   historical cards untouched (not in the file list) — is a small doc-sync wanted, or do historical
   cards stay frozen on purpose?

## Open questions for the director
None beyond the three above; the card's own scope boundaries (zh/en get `()`, `region_language` keeps
returning `str`, no new Korean particles) were implemented exactly as specified, no interpretation needed.
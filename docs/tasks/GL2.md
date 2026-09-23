# GL2 — Glossary term matcher: make particle stripping language-aware (ja support, zh/en no-ops)

**Owner:** GLM builder · **Branch:** `GL2` · **Worktree:** `V:\OmniScan-wt\GL2` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read `src/omniscan/glossary/match.py` in full (the whole module — it's short),
`tests/unit/test_glossary_match.py` in full, and `src/omniscan/translate/languages.py` (`region_language`,
`source_language` — note `region_language` returns plain `str`, not the `Lang` literal type, which is why this
card also types the new parameter as `str`, not `Lang`; see Definitions).

## Why (flagged by TL1's builder, not yet scheduled — see docs/NEXT.md item 4)
`find_terms` strips a fixed set of Korean particles (조사) after every matched glossary term, unconditionally,
regardless of the text's actual language. For Chinese source text this is harmless — Chinese has no attached
grammatical particles of this kind, so the Korean patterns simply never match, which is already correct
behaviour and needs no change. For **Japanese**, this is a real gap: Japanese attaches its own case particles
after nouns exactly the way Korean does (character name + は/が/を/に/... reads as one token in the raw text),
so a locked glossary term followed by a Japanese particle is currently matched as `end = term_end` with
`particle=None` instead of consuming the particle — harmless for exact-substring locked-term checks, but wrong
for anything that inspects `Match.particle` or relies on `Match.end` landing after the particle (e.g. a future
UI highlight, or `substitute_binding`'s replacement span). Since `find_terms` has no idea what language it's
matching, it can't select the right particle set at all today.

## The fix (design decided by the director — implement this shape, not a different one)
1. **`src/omniscan/glossary/match.py`**: rename nothing about `KOREAN_PARTICLES` (kept, still exported). Add,
   right after it:
   ```python
   # Japanese case particles that attach directly after a noun, the same way KOREAN_PARTICLES does. Longer
   # compound forms first so e.g. "とは" is tried before its component "は" (the lookup below sorts by length
   # anyway, but the tuple is written longest-first for readability, matching KOREAN_PARTICLES's style).
   JAPANESE_PARTICLES: tuple[str, ...] = (
       "とは",
       "には",
       "での",
       "から",
       "まで",
       "は",
       "が",
       "を",
       "に",
       "で",
       "と",
       "も",
       "の",
       "へ",
       "や",
   )

   # Particle set to try after a matched term, by the text's source language. Chinese and English have no
   # equivalent attached particles, so they get an empty tuple (no stripping — always correct for them, not a
   # gap: see docs/reports/GL2.md's Why section).
   PARTICLES_BY_LANG: dict[str, tuple[str, ...]] = {
       "ko": KOREAN_PARTICLES,
       "ja": JAPANESE_PARTICLES,
       "zh": (),
       "en": (),
   }
   ```
   Delete the module-level `_PARTICLES_LONGEST_FIRST` constant; replace it with a small helper (memoisation not
   required — entry counts are tiny per call):
   ```python
   def _particles_longest_first(lang: str) -> tuple[str, ...]:
       """`PARTICLES_BY_LANG[lang]` sorted longest-first; unknown `lang` -> no particles (empty tuple)."""
       return tuple(sorted(PARTICLES_BY_LANG.get(lang, ()), key=len, reverse=True))
   ```
   `find_terms` gains a required keyword-only... **no** — make it a required *positional-or-keyword* parameter
   named `lang` placed right after `entries` (not keyword-only: every call site already has the value
   positionally handy and the existing two positional args stay positional):
   ```python
   def find_terms(text: str, entries: Sequence[GlossaryEntry], lang: str) -> list[Match]:
       """... (keep the existing docstring, add one line: "`lang` selects the particle set — see
       PARTICLES_BY_LANG; an unrecognised code strips no particles instead of raising.")"""
   ```
   Inside the function body, replace the one reference to `_PARTICLES_LONGEST_FIRST` with
   `_particles_longest_first(lang)` (compute it once at the top of the function, not per-loop-iteration).
2. **`src/omniscan/glossary/__init__.py`**: add `JAPANESE_PARTICLES` and `PARTICLES_BY_LANG` to both the
   `from omniscan.glossary.match import ...` line and `__all__`, alongside the existing `KOREAN_PARTICLES`.
3. **Every caller of `find_terms` must now pass `lang`** (there are 4 call sites outside the module itself —
   grep to confirm you found them all before finishing):
   - `src/omniscan/web/app.py`, function `get_glossary_hits`: change `find_terms(region.text, entries)` to
     `find_terms(region.text, entries, region.lang)` (the loop already has `region` in scope).
   - `src/omniscan/translate/prompts.py`, function `glossary_subset(regions, entries)`: this function's own
     signature does **not** change (still just `regions`, `entries`) — derive the language once at the top of
     the function body with `lang = region_language(regions)` (already imported in this file) and pass it to
     the inner call: `find_terms(text, [entry], lang)`.
   - `src/omniscan/translate/prompts.py`, function `substitute_binding(text, entries)`: **this function's
     signature does change** — add a required third parameter: `substitute_binding(text: str, entries:
     Sequence[GlossaryEntry], lang: str) -> str`. Update its one internal call from `find_terms(result,
     locked)` to `find_terms(result, locked, lang)`. Update its docstring's first line only if it currently
     doesn't mention language (one clause is enough, e.g. "; `lang` selects the particle set as in
     `find_terms`").
   - `src/omniscan/translate/run.py`: the one call `substitute_binding(source_text(region), entries)` (around
     line 221, inside a function that already has `region` in scope and already passes `region.lang` to
     `translategemma_prompt` on the same line) becomes `substitute_binding(source_text(region), entries,
     region.lang)`.
   - `src/omniscan/translate/postcheck.py`, function `check_locked_terms(source, target, entries)`: **this
     function's signature changes too** — add a required fourth parameter: `check_locked_terms(source: str,
     target: str, entries: Sequence[GlossaryEntry], lang: str) -> list[TermViolation]`. Update its one internal
     call from `find_terms(source, locked)` to `find_terms(source, locked, lang)`.
   - `src/omniscan/translate/postcheck.py`, function `check_regions(...)`: its one call site
     `check_locked_terms(source_text(region), text, entries)` (inside a loop that already has `region` in
     scope) becomes `check_locked_terms(source_text(region), text, entries, region.lang)`. `check_regions`'s
     own signature does not change.

## Definitions (no interpretation needed)
- The new `lang` parameter is typed `str`, not the `Lang` literal from `omniscan.core.schemas`, on purpose:
  `region_language()` (used in `glossary_subset`) already returns plain `str`, and widening its return type is
  out of scope for this card (a separate, unrelated file). Do not import `Lang` anywhere in this card's diff.
- An unrecognised `lang` value (anything not a key of `PARTICLES_BY_LANG`) must not raise — it behaves exactly
  like `zh`/`en` today would (no particle stripping), via `.get(lang, ())`.
- `PARTICLES_BY_LANG["zh"]` and `["en"]` are both `()` on purpose — this is the documented correct behaviour
  for those languages (see Why), not a placeholder to fill in later.
- Existing Korean behaviour must not change at all: every existing test in `tests/unit/test_glossary_match.py`
  keeps its exact expected output, only gaining `lang="ko"` as an extra argument to each `find_terms(...)` call
  in that file (there are 10 such calls — update every one; `test_find_terms_requires_entry_id`'s call also
  needs `lang="ko"` added even though it errors before using the particle set, since the signature itself is
  now required to have it).
- A Japanese particle example to pin exactly: for a glossary entry with `source="ユジン"` (a name), the text
  `"ユジンは学校に行った"` with `lang="ja"` must match `Match(entry_id=..., source="ユジン", start=0, end=4,
  particle="は")` (term is 3 chars, `は` is 1 char, end = 0+3+1 = 4). Also pin a compound particle: text
  `"ユジンとは違う"` with `lang="ja"` must consume `とは` as one particle (not stop at `と`), i.e.
  `end = 0+3+2 = 5`, `particle="とは"`.
- A Chinese pin: the same matcher call with `lang="zh"` over text containing a Korean- or Japanese-looking
  particle character sequence right after the term must **not** strip anything (`particle=None`,
  `end == term_end`) — because `PARTICLES_BY_LANG["zh"] == ()`, not because of any language-detection logic.

## Files you may create / modify
- `src/omniscan/glossary/match.py`
- `src/omniscan/glossary/__init__.py`
- `src/omniscan/translate/prompts.py` (only `glossary_subset` and `substitute_binding`, as described)
- `src/omniscan/translate/run.py` (only the one `substitute_binding(...)` call site, as described)
- `src/omniscan/translate/postcheck.py` (only `check_locked_terms` and `check_regions`, as described)
- `src/omniscan/web/app.py` (only the one `find_terms(...)` call site in `get_glossary_hits`, as described)
- `tests/unit/test_glossary_match.py` (extend)
- `tests/unit/test_translate_prompts.py`, `tests/unit/test_translate_postcheck.py`, `tests/unit/test_web_app.py`
  — find each file's existing tests that exercise `glossary_subset`, `substitute_binding`, `check_locked_terms`,
  `check_regions`, and `get_glossary_hits`/`glossary-hits`; update any call whose signature changed, add one
  small new test per function proving a non-Korean `lang` is honoured (Japanese particle stripped;
  Chinese/English not). If a listed test file does not exist or does not cover the function, grep for the
  function name across `tests/unit/` to find the right file instead of creating a duplicate.
- `docs/reports/GL2.md` (create)
Do not touch `src/omniscan/core/**`, `src/omniscan/translate/languages.py` (`region_language`'s return type
stays `str` — out of scope), or any other file not listed above.

## Acceptance tests (CPU only, no GPU/models)
1. **`find_terms` Korean regression**: every existing test in `tests/unit/test_glossary_match.py` still passes
   with only `lang="ko"` added to each call — same expected `Match` values as before, byte-for-byte.
2. **`find_terms` Japanese**: both pinned examples in Definitions (simple `は` and compound `とは`) match
   exactly as specified; a Japanese term with no particle following it (end of string or a space) yields
   `particle=None` (mirrors the existing Korean `test_no_particle_before_space_or_end_of_string` test, ported
   to `lang="ja"`).
3. **`find_terms` Chinese/English no-op**: the pinned Chinese example in Definitions; also confirm
   `lang="en"` behaves the same way (no stripping) with a simple case.
4. **`find_terms` unknown lang does not raise**: e.g. `lang="xx"` behaves like zh/en (no particles stripped),
   proving `.get(lang, ())` and not a `KeyError`.
5. **`_particles_longest_first`** (or whatever it ends up named — test through `find_terms`'s observable
   behaviour if you'd rather not test a private helper directly): a longer Japanese particle is preferred over
   its component when both would match (the `とは`/`は` pin above already proves this indirectly — an explicit
   unit test on the helper is a nice-to-have, not required, if testing it directly needs exporting it).
6. **`glossary_subset`**: a glossary term followed by a Japanese particle in Japanese source regions is still
   found (i.e. `glossary_subset` doesn't silently break because it derives `lang` from `region_language`
   internally) — extend or add a test with `Region(lang="ja", ...)` fixtures.
7. **`substitute_binding`**: update every existing call in its test file to pass a `lang` (use `"ko"` for
   existing Korean-text tests); add one new test with Japanese text + a Japanese particle proving the
   substitution span is computed correctly (the replaced term does not accidentally eat or leave behind part of
   the particle).
8. **`check_locked_terms`/`check_regions`**: same treatment — existing tests gain `lang="ko"` (or, for
   `check_regions`, keep using its `Region.lang` field, which should already be `"ko"` in those fixtures), plus
   one new Japanese-language test.
9. **`get_glossary_hits`** (web): existing tests for this route still pass; if none inspect `particle` in the
   response body for a non-Korean region, that's fine — the route just needs to not crash and to keep passing
   `region.lang` through.
10. `uv run pytest -q -m "not gpu"` and `tests/unit/test_docs.py` (if such a file exists — grep first) stay
    green; `uv run ruff format . && uv run ruff check .` and `uv run pyright` clean.

## Out of scope
Actual Japanese source-language pipeline testing beyond this module's own unit tests (no real Japanese raw data
exists yet per `docs/CHECKPOINT.md`'s "Not done" list); widening `region_language`'s return type to `Lang`;
adding new particles to the Korean set; a UI to show which particle was stripped; Chinese measure words or any
other zh-specific tokenisation (confirmed out of scope — zh gets no particle stripping, period).

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_glossary_match.py -q
uv run pytest -q -m "not gpu"
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
`docs/reports/GL2.md`: Changes, Tests (commands + results), Deviations, Questions. Commit early
(`GL2: WIP particle-by-language table`), final commit `GL2: language-aware particle stripping (ja support, zh/en no-ops)`.
You have 150 tool calls in total. If anything is unclear: stop, write the question under Questions, commit,
and end.

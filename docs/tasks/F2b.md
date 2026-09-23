# F2b — Promo filter tier 1: reclassify OCR'd ad/spam text as `watermark`, not translated text

**Owner:** GLM builder · **Branch:** `F2b` · **Worktree:** `V:\OmniScan-wt\F2b` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read `src/omniscan/translate/profiles.py` (`load_profiles`, `default_profile_paths`
— the exact TOML-file-pair loading pattern to copy, additive-list style not field-override style),
`src/omniscan/ocr/stage.py` in full, `src/omniscan/core/schemas.py` (`Region`, `RegionKind` — note `"watermark"`
already exists as a kind), and every place that already special-cases `region.kind == "watermark"`:
`src/omniscan/translate/prompts.py` (`translatable`), `src/omniscan/eval/score.py` (`score_chapter`,
`_ocr_hypothesis`), `src/omniscan/inpaint/pipeline.py` (the region loop), `src/omniscan/glossary/reference.py`.
Also skim `docs/PRODUCT_SPEC.md` section 4 and `docs/tasks/F2a.md` (already merged — tier 2, image-hash based;
this card is tier 1, text-based, a separate mechanism, not a continuation of F2a's code).

## Why (found by the director on a real chapter, not invented)
Solo Leveling raws (the owner's own downloaded material) carry a gambling-site ad injected by the source
aggregator, e.g. `구글검색 "먹튀검증 스포위키"` and `라이브스코어 스포츠중계 가상토토 전문가 정기/오목 웹툰`, appearing on
most pages at varying positions (not a fixed banner — `omniscan watermark`'s fixed-fraction-of-page tool,
already built, does not fit). Today this text is detected as an ordinary region: on some pages it is missed by
detection entirely, on others (Solo Leveling ch2) it is OCR'd, **translated and typeset exactly like real
dialogue** — a rendered English caption reading "LiveScore Sports Broadcast Virtual Toto Expert Picks..." now
sits on the page as if it were an official part of the release, which reads worse than leftover Korean would.
The `"watermark"` `RegionKind` already exists and is already excluded, everywhere that matters, by existing
code: `translatable()` (never translated), `score_chapter` (never scored), inpaint's region loop (never
removed — matches the owner's own decision, `docs/OPEN_QUESTIONS.md` E4, not to strip source watermarks). What
is missing is anything that *assigns* `"watermark"` to a region based on its OCR'd text.

## The fix (design decided by the director — implement this shape, not a different one)
1. New standalone config, following `translate/profiles.py`'s exact two-file pattern (shipped + user, **later
   file's list is appended to the shipped one, not replacing it** — this differs from `load_profiles`, which
   replaces by key; here there is no key, just a flat pattern list, so both files' patterns are simply
   concatenated in load order, duplicates kept, order preserved):
   - `config/watermark_text.toml`:
     ```toml
     # Text patterns that mark a detected region as a source-injected ad/watermark, never translated or
     # typeset (docs/OPEN_QUESTIONS.md E4: watermarks are not removed, only excluded from translation).
     # Patterns are matched as a case-insensitive substring of the region's OCR'd text (whitespace collapsed
     # the same way translate.prompts.source_text does). Keep entries specific (brand/site names), not
     # generic words, so real dialogue is never misclassified.
     [watermark_text]
     patterns = [
         "구글검색",       # "Google search" — found injected in Solo Leveling raws (card F2b)
         "라이브스코어",   # "LiveScore" (gambling/sports-betting brand)
         "가상토토",       # "virtual toto" (a Korean sports-betting product)
         "스포위키",       # "SpoWiki" (a specific ad-site brand)
     ]
     ```
   - New module `src/omniscan/ocr/watermark_text.py`:
     ```python
     def default_watermark_text_paths() -> list[Path]:
         """Shipped repo file, then the per-user one (both contribute; see the module docstring)."""

     def load_watermark_patterns(paths: Sequence[Path]) -> list[str]:
         """Every pattern from every existing file in `paths`, in order, duplicates kept."""

     def matches_watermark_text(text: str, patterns: Sequence[str]) -> bool:
         """True if any pattern is a case-insensitive substring of `text`. Empty `patterns` -> always False."""

     def reclassify_watermark_regions(regions: Sequence[Region], patterns: Sequence[str]) -> list[Region]:
         """Regions whose OCR text matches a pattern get kind="watermark" (via model_copy); others unchanged
         (same object, not just equal — callers may rely on identity for regions that pass through)."""
     ```
2. **`src/omniscan/ocr/stage.py`**: after `kept = [r for r in ocr_regions if ...]` and before saving, call
   `reclassify_watermark_regions(kept, load_watermark_patterns(default_watermark_text_paths()))`; add a
   `"watermarked": float(sum(1 for r in reclassified if r.kind == "watermark"))` metric. `config_subset` gains
   a fingerprint of the loaded patterns (sha256 hex of `"\n".join(patterns)`, so editing either TOML file
   invalidates the stage the same way `examples_fingerprint` does in F2a) — add it as `"watermark_patterns":
   <hex>` to the returned dict. Bump `version` from 2 to 3 with a comment (`# 3: watermark text reclassification`).
   A region already classified `"sfx"` is left alone even if its text happens to match (sfx text is short and
   pattern collisions there would be a real risk; only reclassify `"bubble_text"`/`"free_text"`).

## Definitions (no interpretation needed)
- Matching uses the same whitespace-collapse as `translate.prompts.source_text` (join `region.text.split()`
  with single spaces) before the substring check, so a pattern spanning a line break in the OCR text still
  matches.
- `reclassify_watermark_regions` changes only `kind`; every other field of a matched region (`bbox`, `lines`,
  `confidence`, `text`, etc.) is untouched — a `"watermark"` region still has its OCR text, in case a future
  card wants it (e.g. a report of what got filtered).
- The real examples above must match: `구글검색 "먹튀검증 스포위키"` matches on both `구글검색` and `스포위키`;
  `라이브스코어 스포츠중계 가상토토 전문가 정기/오목 웹툰` matches on `라이브스코어` and `가상토토`. A region reading
  ordinary dialogue (use a handful of real lines from `data/raws` fixtures or the project's synthetic Korean
  test text) must not match any shipped pattern — pin this as a real regression test, not just a synthetic one.

## Files you may create / modify
- `src/omniscan/ocr/watermark_text.py` (create)
- `config/watermark_text.toml` (create, exactly the seed list above)
- `src/omniscan/ocr/stage.py` (the one integration point described above only)
- `tests/unit/test_ocr_watermark_text.py` (create), `tests/unit/test_ocr_stage.py` (extend — find the existing
  OCR stage test file; if none exists with that exact name, grep for `class OcrStage` usage in tests and
  extend whichever file already covers it)
- `docs/USER_GUIDE.md` (one short paragraph: what the patterns file is, that watermarked regions are excluded
  from translation but not removed from the image, how to add a site-specific pattern)
- `docs/reports/F2b.md` (create)
Do not touch `src/omniscan/core/**`, `src/omniscan/watermark/**` (the unrelated fixed-position tool — tier 3,
F2c, a different mechanism), `src/omniscan/filter/**` (F2a's image-hash tier), `detect/**`, or any `data/**` file.

## Acceptance tests (CPU only, no GPU/models)
1. **`matches_watermark_text`**: case-insensitive, substring not exact-match, whitespace-collapsed text,
   empty `patterns` always False, the real examples above match, ordinary dialogue does not.
2. **`load_watermark_patterns`**: shipped file alone loads the 4 seed patterns in order; a user file adds more
   (both contribute, order preserved, duplicates allowed); a missing file in the list is skipped, not an error.
3. **`reclassify_watermark_regions`**: a mix of matching and non-matching regions — matching ones get
   `kind="watermark"`, others are returned unchanged (same object identity, not just `==`, for the untouched
   ones — assert `is`); an `"sfx"` region whose text happens to match a pattern is **not** reclassified; a
   `"watermark"`-kind region (already so) stays a watermark either way.
4. **`OcrStage`**: with a fake reader/detector producing the ad text above among real dialogue, `ocr.json`
   ends up with that region as `kind="watermark"`; the `"watermarked"` metric counts it; editing
   `config/watermark_text.toml`'s equivalent (a temp copy) or the user override changes `config_subset`'s
   fingerprint (manifest invalidation: `done` after the change, `skipped` without it, same pattern as F2a's
   `IngestStage` test).
5. **Downstream stays correct with no new code** (regression proof, not new production code): a
   `"watermark"`-kind region from this reclassification is excluded by `translatable()` (already true —
   confirm with a quick assertion) and excluded by `score_chapter` predicted regions.
6. `uv run pytest -q -m "not gpu"` and `tests/unit/test_docs.py` stay green.

## Out of scope
Tier 3 (`omniscan watermark`'s fixed-position tool, already built but unwired — a separate future card),
removing/inpainting watermark regions (the owner's own decision, E4, is not to), a GUI or CLI command to
list/manage patterns (a future card if wanted), fuzzy/regex matching (plain substring is enough for the
evidence gathered so far), re-running the qualification suite.

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_ocr_watermark_text.py tests/unit/test_docs.py -q
uv run pytest -q -m "not gpu"
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
`docs/reports/F2b.md`: Changes, Tests (commands + results), Deviations, Questions. Commit early
(`F2b: WIP watermark_text module`), final commit `F2b: reclassify OCR'd ad text as watermark, not translated`.
You have 150 tool calls in total. If anything is unclear: stop, write the question under Questions, commit,
and end.

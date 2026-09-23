# F2b — Promo filter tier 1: reclassify OCR'd ad/spam text as `watermark`, not translated text

Branch `F2b`, worktree `V:\OmniScan-wt\F2b`. Two commits: `F2b: WIP watermark_text module`,
`F2b: reclassify OCR'd ad text as watermark, not translated`.

## Changes

New files

- `config/watermark_text.toml` — the seed pattern list, exactly as the card specifies:
  `구글검색`, `라이브스코어`, `가상토토`, `스포위키` (with the per-pattern comments from the card).
- `src/omniscan/ocr/watermark_text.py` — the four functions the card defines:
  - `default_watermark_text_paths()`: shipped repo `config/watermark_text.toml`, then the per-user
    `~/.config/omniscan/watermark_text.toml` (built from `DEFAULT_TOML`/`USER_TOML` the same way
    `translate/profiles.py` does).
  - `load_watermark_patterns(paths)`: every pattern from every existing file, in order, duplicates
    kept, missing files skipped. Additive-list semantics: both files' lists are simply concatenated
    (no name key, so nothing replaces anything — the difference from `load_profiles` is in the
    module docstring). Invalid TOML / a non-`[watermark_text]` table / non-string or empty
    patterns raise `ValueError` with the file path (see Deviations).
  - `matches_watermark_text(text, patterns)`: case-folded substring check after collapsing
    whitespace on both sides (same collapse as `translate.prompts.source_text`); empty `patterns`
    → False (empty `any()`).
  - `reclassify_watermark_regions(regions, patterns)`: a `bubble_text`/`free_text` region whose
    text matches gets `kind="watermark"` via `model_copy(update={"kind": "watermark"})` — only the
    kind changes; every other field untouched. Everything else passes through as the **same
    object** (identity preserved), including `sfx` regions that happen to match and regions that
    are already `"watermark"`.
- `tests/unit/test_ocr_watermark_text.py` — the module's tests plus the downstream regression
  proofs (below).

Modified files

- `src/omniscan/ocr/stage.py` — `version = 3  # 3: watermark text reclassification`. In `run()`,
  right after the `drop_conf` filter and before `ocr.json` is saved:
  `kept = reclassify_watermark_regions(kept, load_watermark_patterns(default_watermark_text_paths()))`
  and `metrics["watermarked"] = float(sum(1 for r in kept if r.kind == "watermark"))`.
  `config_subset` gains `"watermark_patterns": <sha256 hex of "\n".join(patterns)>` (the patterns
  are loaded right there — `default_watermark_text_paths()` needs no ctx, unlike F2a's examples,
  so no state on the stage instance), so editing either TOML file invalidates the stage the same
  way F2a's `examples_fingerprint` does.
- `tests/unit/test_ocr_stage.py` — three new tests (see Tests).
- `docs/USER_GUIDE.md` — a short "Watermark text patterns" section after `filter add`: what the
  patterns files are (both, additive), that watermarked regions are excluded from translation but
  **not** removed from the image, and how to add a site-specific pattern (append to the user file;
  case-insensitive substring of the OCR'd text).

Downstream needed **no** new code (regression-proofed by tests): `translatable()` excludes
watermarks, `score_chapter` counts only non-watermark regions as predicted, `inpaint` skips them
(E4), `reference`'s pairing skips them, the debugger/preview colour them gray.

## Tests

```
uv run pytest tests/unit/test_ocr_watermark_text.py tests/unit/test_ocr_stage.py tests/unit/test_docs.py
  → 40 passed
uv run pytest -m "not gpu" → 3880 passed, 26 deselected, 1 xfailed (pre-existing, unrelated)
uv run ruff format . && uv run ruff check . → clean
uv run pyright → 0 errors
```

Coverage per the card's acceptance list:

1. `matches_watermark_text`: case-insensitive (`LiveScore`/`LIVESCORE` vs `livescore`), substring
   not exact-match, whitespace collapse (a pattern spanning an OCR line break matches), empty
   patterns always False, **the two real ad strings from the card match** (each on both of its
   patterns), and ordinary dialogue does not — pinned against all 12 real Korean lines of
   `tests/fixtures/korean_pages.py::KOREAN_LINES` plus the shipped pattern list, not just a
   synthetic string.
2. `load_watermark_patterns`: the shipped file alone loads exactly the 4 seed patterns in order;
   a user file adds more (both contribute, order preserved, duplicates kept); missing files in the
   list are skipped; invalid TOML, a non-`[watermark_text]` table, non-string and empty/blank
   patterns are refused with the file path in the error.
3. `reclassify_watermark_regions`: mixed input — matches become `"watermark"` (new copies),
   non-matches pass through as the **same objects** (`assert ... is ...`), an `sfx` region whose
   text matches stays `sfx`, already-`"watermark"` regions stay watermarks either way (same
   objects), and a reclassified region differs from the input in `kind` only (id/text/bbox/
   confidence/slice_index asserted equal).
4. `OcrStage` (fake crop reader over 4 regions: dialogue bubble, two ad regions, an sfx with ad
   text): `ocr.json` has kinds `[bubble_text, watermark, watermark, sfx]`, `watermarked == 2.0`,
   the watermark regions keep their OCR text; with no matching pattern loaded the region stays
   `bubble_text` and the stage is `skipped` on re-run; adding a pattern to the user override file
   re-runs it (`done`) and the region becomes `watermark` (manifest invalidation, same pattern as
   F2a's IngestStage tests — the two paths are monkeypatched to temp files so no test touches the
   real `~/.config/omniscan/`); `config_subset`'s fingerprint follows both files.
5. Downstream with no new code: a reclassified (now-`"watermark"`) region is excluded by
   `translatable()` (asserted) and by `score_chapter`'s predicted regions (it counts as neither
   detected nor predicted: `regions == 0`, `detected_boxes == 0`, `precision is None`).
6. Full CPU suite green, `test_docs.py` green (USER_GUIDE still matches the real CLI).

## Deviations

- `load_watermark_patterns` refuses **empty or blank patterns** (ValueError): an empty string is a
  substring of every text and would silently reclassify a whole chapter as watermarks. The card
  does not mention this case; refusing it loudly seemed strictly safer than the alternative.
  Likewise non-string entries and invalid TOML raise with the file path, following
  `load_profiles`' error style.
- Patterns are whitespace-collapsed **and** case-folded on both sides (the card pins the collapse
  for the region text; applying the same normalisation to the pattern makes multi-word user
  patterns robust to inconsistent OCR spacing and is a no-op for the shipped single-token seeds).
- The card said to use `data/raws` fixtures or "the project's synthetic Korean test text" for the
  ordinary-dialogue regression: used `tests/fixtures/korean_pages.py::KOREAN_LINES` (the project's
  Solo Leveling-style dialogue fixture, CPU-only and committed) — reading from `data/raws` in a
  test would make the suite machine-dependent.

## Questions

- None blocking. Two observations for future cards: (a) `eval`'s `precision` silently drops
  reclassified ad regions from scoring — if the owner ever wants a "what got filtered" report, the
  OCR text is preserved on the watermark regions and the new `watermarked` stage metric is the
  per-chapter count; (b) tier 3 (`omniscan watermark` fixed-position tool) remains unwired, as the
  card says — F2c territory.
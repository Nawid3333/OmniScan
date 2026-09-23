# F2c — Promo filter tier 3: wire the fixed-position watermark tool into `detect`

Branch `F2c`, worktree `V:\OmniScan-wt\F2c`. Two commits: `F2c: WIP watermark_position module`,
`F2c: wire fixed-position watermark boxes into detect`.

## Changes

New files

- `src/omniscan/detect/watermark_position.py` — the three functions the card defines:
  - `_ioa(region_bbox, watermark_bbox)`: intersection area over the **region's own** area (IoA, not
    IoU), plain arithmetic on `BBox`'s public fields/properties; a zero-area region returns `0.0`
    (no `ZeroDivisionError`).
  - `region_overlaps_watermark(region_bbox, watermark_boxes)`: true when any **one** box holds at
    least `_MIN_OVERLAP_IOA = 0.5` (inclusive) of the region's own area — boxes are checked
    independently, never unioned/merged.
  - `reclassify_watermark_position_regions(regions, watermark_boxes)`: the same shape as F2b's
    `reclassify_watermark_regions` — an eligible region gets `kind="watermark"` via
    `model_copy(update={"kind": "watermark"})` (only the kind changes); only `bubble_text`/
    `free_text` are eligible (an `sfx` region is left alone even when it overlaps, F2b's carve-out);
    a region already `"watermark"` stays one either way; untouched regions are returned as the
    exact same object (`is`, not just `==`); empty `watermark_boxes` → every region unchanged.
- `tests/unit/test_detect_watermark_position.py` — the module's tests plus the downstream
  regression proof (below).

Modified files

- `src/omniscan/detect/stage.py` — `version = 3  # 3: fixed-position watermark reclassification
  (F2c)`. Three changes exactly as the card specifies:
  - `inputs()` gains `ctx.series.work_dir / "watermarks.json"` (right after `"slices.json"`; the
    file may not exist — `hash_inputs` hashes a missing file as `"<missing>"`, so no existence
    check, and a later `omniscan watermark add` invalidates every chapter's cached detect output).
  - `run()`, after `build_regions(...)` and before `regions.json` is saved:
    `watermark_boxes = resolve_watermark_regions(WatermarkStore(ctx.series.work_dir).list(), ingest)`
    (reusing the `ingest` already loaded earlier in the method, not a reload) followed by
    `regions = reclassify_watermark_position_regions(regions, watermark_boxes)`.
  - the returned metrics gain `"watermarked": float(sum(1 for r in regions if r.kind == "watermark"))`,
    alongside the existing `"regions"` key (same metric name/shape as F2b's OCR stage).
- `tests/unit/test_detect_stage.py` — two new tests on the existing harness (see Tests).
- `docs/USER_GUIDE.md` — one paragraph in the `omniscan watermark list` / `watermark remove`
  section: stored regions take effect on the next `detect` run of the series, any detected region
  whose box mostly (>= 50 % of its own area) falls inside a stored zone is excluded from
  translation/scoring/evaluation like the tier 1 and tier 2 matches, and existing chapters
  re-detect automatically because the stage now depends on `watermarks.json`.

Downstream needed **no** new code (regression-proofed by test): `translatable()` excludes
`"watermark"` regions, as do `score_chapter`, the inpaint region loop, `glossary.reference`'s
pairing and `typeset` — all pre-existing `kind == "watermark"` checks (F2b's list, now also fed by
tier 3).

## Tests

```
uv run pytest tests/unit/test_detect_watermark_position.py tests/unit/test_detect_stage.py
  tests/unit/test_docs.py → 27 passed
uv run pytest -m "not gpu" → 3957 passed, 26 deselected, 1 xfailed (pre-existing, unrelated)
uv run ruff format . && uv run ruff check . → clean
uv run pyright → 0 errors
```

Coverage per the card's acceptance list:

1. `_ioa`: the four pinned examples asserted to the exact given values — `10_000 / 15_000`
   (≈ 0.667, above), `5_000 / 20_000 = 0.25` (below), `10_000 / 20_000 = 0.5` (boundary, exact),
   and `0.0` (no exception) for both zero-area shapes (`x0 == x1` and `y0 == y1`, both allowed by
   `BBox`'s validator); plus a disjoint-boxes case.
2. `region_overlaps_watermark`: true for above and boundary, false for below; boxes checked
   independently, not unioned (two boxes each holding only 0.4 of the region, union 0.8 → false);
   a match in either the first or the second of two boxes is enough (both orders asserted); empty
   list → false.
3. `reclassify_watermark_position_regions`: mixed input — the overlapping `bubble_text` region
   becomes `"watermark"` (a new copy, every other field asserted equal), the non-overlapping
   `free_text`, the overlapping `sfx` and the already-`"watermark"` region pass through as the
   **same objects** (`assert ... is ...`); already-watermark regions stay watermarks whether or not
   they overlap (never un-marked, never copied needlessly); empty `watermark_boxes` → all three
   input regions back by identity.
4. `DetectStage` integration (existing harness, real ingest/slice, fake detector): 3 noise pages of
   100x150 → a 100x450 strip; scripted `text_free` detections on tiles 0 and 4 land at strip-space
   bboxes (20, 30, 80, 70) and (20, 230, 80, 270) (hand-computed: tile i sits at strip y = 50·i).
   A watermark stored at fractions (0.2, 0.2, 0.8, 0.4) resolves per page to x 20..80,
   y file.y0+30..file.y0+60 — on page 1 that is (20, 30, 80, 60), holding 60×30 of the first
   region's own 60×40 area (IoA 0.75). `regions.json` then has the first region as
   `kind="watermark"`, the second (elsewhere on the strip, clear of all three per-page resolved
   boxes) still `free_text`, `regions == 2.0` and `watermarked == 1.0`.
5. Manifest invalidation: `DetectStage` with no `watermarks.json` → `done`; unchanged re-run →
   `skipped`; `WatermarkStore(...).add(...)` → `done` again (the input hash flips from
   `"<missing>"` to the real file hash, exactly the `hash_inputs` mechanism the card cites).
6. Downstream with no new code: a position-reclassified `"watermark"` region (with OCR text) is
   excluded by `translatable()`.
7. Full CPU suite green, `test_docs.py` green (USER_GUIDE still matches the real CLI).

## Deviations

- None. The card's shape was followed exactly (module with the three functions and the pinned
  `_MIN_OVERLAP_IOA = 0.5` inclusive threshold, the three stage changes, the metric, the version
  bump with its comment, the guide paragraph).

## Questions

- None blocking. One observation for future cards: at detect time a reclassified region is still
  OCR'd like any other (the card's out-of-scope note); if the owner ever wants that compute
  skipped, `ocr/pipeline.py`'s region loop is where a `kind == "watermark"` early-continue would
  go, and the stage-level `watermarked` metric is already in place to report it.
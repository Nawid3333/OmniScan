# F2c — Promo filter tier 3: wire the fixed-position watermark tool into `detect`

**Owner:** GLM builder · **Branch:** `F2c` · **Worktree:** `V:\OmniScan-wt\F2c` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read, in full: `src/omniscan/watermark/store.py` (`WatermarkStore`,
`WatermarkRegion` — persistence, already CLI-wired via `omniscan watermark add/list/remove`),
`src/omniscan/watermark/resolve.py` (`resolve_watermark_regions` — page-fraction boxes to strip-space
`BBox`, already exists, nothing calls it in the real pipeline yet), `src/omniscan/detect/stage.py` in full
(the exact file you'll extend), `src/omniscan/ocr/watermark_text.py` and its call site in
`src/omniscan/ocr/stage.py` (F2b, already merged — tier 1's text-pattern reclassification; this card is tier
3, position-based, a separate mechanism with the same overall shape: a pure module + one stage call site),
and every place that already special-cases `region.kind == "watermark"`: `src/omniscan/translate/prompts.py`
(`translatable`), `src/omniscan/eval/score.py` (`score_chapter`), `src/omniscan/inpaint/pipeline.py` (the
region loop), `src/omniscan/glossary/reference.py`. Also read `src/omniscan/core/manifest.py`'s
`hash_inputs` (missing files hash as `"<missing>"` — this is why adding a not-yet-existing file to a stage's
`inputs()` is always safe).

## Why
`omniscan watermark add/list/remove` already lets the owner record a fixed-position watermark/logo/URL stamp
that appears at the same fractional position on every page of a series (e.g. a corner site-brand stamp,
different in kind from F2b's tier 1, which matches by OCR'd *text* wherever it happens to sit — that tier
doesn't apply to a stamp too small or stylised to OCR cleanly, or one made of a logo image with no legible
text at all). `resolve_watermark_regions` already converts those stored fractional boxes into real strip-space
`BBox`es for a chapter's `ingest.json`. Nothing in the real pipeline calls it — a stored watermark region today
has zero effect on any chapter's output. Unlike tier 1, tier 3 is purely positional: it needs no OCR'd text at
all, so it belongs in `detect` (before OCR ever runs), not `ocr`.

## The fix (design decided by the director — implement this shape, not a different one)
1. New module `src/omniscan/detect/watermark_position.py`:
   ```python
   """Tier-3 promo filter: reclassify detected regions that overlap a series' stored fixed-position
   watermark box (see watermark/store.py, watermark/resolve.py) as kind="watermark"."""

   _MIN_OVERLAP_IOA = 0.5  # a region needs at least half its own area inside a stored watermark zone

   def _ioa(region_bbox: BBox, watermark_bbox: BBox) -> float:
       """Intersection area over `region_bbox`'s own area (0.0 when `region_bbox` has zero area)."""

   def region_overlaps_watermark(region_bbox: BBox, watermark_boxes: Sequence[BBox]) -> bool:
       """True if `region_bbox` has at least `_MIN_OVERLAP_IOA` of its own area inside any one of
       `watermark_boxes` (checked independently per box, not their union)."""

   def reclassify_watermark_position_regions(
       regions: Sequence[Region], watermark_boxes: Sequence[BBox]
   ) -> list[Region]:
       """Regions overlapping a stored watermark box get kind="watermark" (via model_copy, same
       shape as ocr/watermark_text.py's reclassify_watermark_regions): only "bubble_text"/"free_text"
       are eligible (an "sfx" region is left alone even if it overlaps, same reasoning as F2b: sfx
       text is short and a spurious collision there is a real risk); a region already kind="watermark"
       stays one either way. Untouched regions are returned as the exact same object (identity, not
       just equality — callers may rely on this). Empty `watermark_boxes` -> every region unchanged."""
   ```
   `_ioa` is plain arithmetic on `BBox`'s existing public fields/properties (`x0`, `y0`, `x1`, `y1`, `width`,
   `height` all already exist on `BBox` in `core/schemas.py` — do not add a method to `BBox` itself, that file
   is off-limits): `ix = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))`, `iy` the same for y, `area = a.width *
   a.height`, return `(ix * iy) / area if area else 0.0`.
2. **`src/omniscan/detect/stage.py`**:
   - `inputs()`: add `ctx.series.work_dir / "watermarks.json"` to the returned list (any position; suggested
     right after `"slices.json"`). This file may not exist yet for a series with no stored watermarks —
     `hash_inputs` already hashes a missing file as `"<missing>"`, so no existence check is needed here, and a
     later `omniscan watermark add` on that series correctly invalidates every chapter's cached `detect` output.
   - `run()`: after `regions = build_regions(...)` and before `RegionsArtifact(regions=regions).save(...)`,
     insert:
     ```python
     watermark_boxes = resolve_watermark_regions(WatermarkStore(ctx.series.work_dir).list(), ingest)
     regions = reclassify_watermark_position_regions(regions, watermark_boxes)
     ```
     (`ingest` is already loaded earlier in this same method as `ingest = IngestArtifact.load(ingest_path)` —
     reuse it, do not reload.) Add one new metric to the returned dict, alongside the existing `"regions"` key:
     `"watermarked": float(sum(1 for r in regions if r.kind == "watermark"))`.
   - New imports needed: `from omniscan.watermark.store import WatermarkStore`, `from
     omniscan.watermark.resolve import resolve_watermark_regions`, `from
     omniscan.detect.watermark_position import reclassify_watermark_position_regions`.
   - Bump `version` from `2` to `3` with a comment on the new line, matching the existing comment style:
     `version: ClassVar[int] = 3  # 3: fixed-position watermark reclassification (F2c)`.
3. **`docs/USER_GUIDE.md`**: extend whatever section already documents `omniscan watermark add/list/remove`
   with one short paragraph: stored watermark regions now actually affect chapters — any detected region
   whose box mostly (>= 50% of its own area) falls inside a stored watermark zone is excluded from
   translation the same way tier 1 (text-pattern) and tier 2 (image-hash) matches are, on the next
   `detect` run for that series (existing chapters re-detect automatically because the stage now depends on
   `watermarks.json`).

## Definitions (no interpretation needed)
- "Overlap" is IoA (intersection over the *region's own* area), not IoU — a small detected region fully
  inside a much larger stored watermark zone should count as a full match even though the watermark zone
  itself is far bigger than the region, which plain IoU would score very low.
- `_MIN_OVERLAP_IOA = 0.5` is inclusive: a region with exactly 0.5 IoA against some watermark box **is**
  reclassified (`>=`, not `>`).
- Three pinned numeric examples (compute these by hand to confirm before writing the test, then assert them
  exactly — do not just eyeball a "roughly right" number):
  - `watermark_bbox = BBox(x0=0, y0=0, x1=100, y1=100)` (area 10 000).
  - `region_above = BBox(x0=0, y0=0, x1=100, y1=150)` (area 15 000; intersection 100×100 = 10 000; IoA =
    10 000/15 000 ≈ 0.667) → **reclassified**.
  - `region_below = BBox(x0=0, y0=50, x1=100, y1=250)` (area 20 000; intersection is x:[0,100] × y:[50,100] =
    100×50 = 5 000; IoA = 5 000/20 000 = 0.25) → **not reclassified**.
  - `region_boundary = BBox(x0=0, y0=0, x1=100, y1=200)` (area 20 000; intersection 100×100 = 10 000; IoA =
    10 000/20 000 = 0.5 exactly) → **reclassified** (boundary is inclusive).
  - A region with zero area (`x0 == x1` or `y0 == y1`, which `BBox`'s own validator allows — it only rejects
    `x1 < x0`/`y1 < y0`) must yield `_ioa == 0.0`, not a `ZeroDivisionError`.
- A region overlapping **any one** of several stored watermark boxes above the threshold is reclassified —
  boxes are checked independently, never unioned/merged first.
- An `"sfx"`-kind region that geometrically overlaps a watermark box above the threshold is **not**
  reclassified (same carve-out as F2b, same reasoning). A region already `kind="watermark"` (e.g. from tier 1)
  is left as `"watermark"` either way — this function never un-marks anything.

## Files you may create / modify
- `src/omniscan/detect/watermark_position.py` (create)
- `src/omniscan/detect/stage.py` (only the three changes described above)
- `tests/unit/test_detect_watermark_position.py` (create)
- `tests/unit/test_detect_stage.py` (extend — this is the existing file's test harness to imitate: `stage_cfg`,
  `FakeDetector`/`FakeScheduler`, `prepare(...)`, `run_chapter([IngestStage(), SliceStage()], ctx)` to build
  real `ingest.json`/`slices.json`, then `run_stage(DetectStage(), ctx)`; see
  `test_detect_stage_is_resumable_and_invalidated_by_config` for the manifest-invalidation pattern to copy)
- `docs/USER_GUIDE.md` (the one paragraph described above)
- `docs/reports/F2c.md` (create)
Do not touch `src/omniscan/core/**` (including `BBox` itself), `src/omniscan/watermark/**` (the store/resolve
module is already complete — use it, don't change it), `src/omniscan/ocr/**` (tier 1's own files, a separate
mechanism), `src/omniscan/filter/**` (tier 2, image-hash), or any `data/**` file.

## Acceptance tests (CPU only, no GPU/models)
1. **`_ioa`**: the four pinned examples in Definitions (above/below/boundary/zero-area), asserted to the exact
   numeric values given.
2. **`region_overlaps_watermark`**: true for the above/boundary cases, false for the below case; true when
   only the *second* of two watermark boxes clears the threshold (proves boxes are checked independently, not
   unioned); false for an empty `watermark_boxes` list.
3. **`reclassify_watermark_position_regions`**:
   - A mix of overlapping and non-overlapping regions — overlapping ones get `kind="watermark"`, others are
     returned unchanged (assert `is`, not just `==`, for the untouched ones).
   - An `"sfx"`-kind region that geometrically overlaps a watermark box is **not** reclassified.
   - A region already `kind="watermark"` stays `"watermark"` whether or not it overlaps anything.
   - `reclassify_watermark_position_regions(regions, [])` returns every region unchanged (`is`, all of them).
4. **`DetectStage` integration**: using the existing test file's harness, store a watermark region via
   `WatermarkStore(ctx.series.work_dir).add(...)` at a fractional box known (by hand-computed geometry) to
   land on top of one specific fake-detected region's strip-space bbox after `resolve_watermark_regions`;
   after `run_stage(DetectStage(), ctx)`, `regions.json` has that region as `kind="watermark"`, the
   `"watermarked"` metric counts it, and a region elsewhere on the page is untouched.
5. **Manifest invalidation**: run `DetectStage` once with no `watermarks.json` (`done`), run again unchanged
   (`skipped`), then call `WatermarkStore(...).add(...)` and run a third time — `done` again (mirrors
   `test_detect_stage_is_resumable_and_invalidated_by_config`'s shape, but the change is a new watermark
   region instead of a config field).
6. **Downstream stays correct with no new code** (regression proof only, not new production code): a
   `"watermark"`-kind region produced this way is excluded by `translatable()` (already true — one quick
   assertion is enough, no need to re-test `translatable` itself).
7. `uv run pytest -q -m "not gpu"` and any doc-consistency test (grep for one, e.g. `tests/unit/test_docs.py`)
   stay green.

## Out of scope
Skipping OCR *compute* entirely for a region already reclassified at detect time (today it still gets read by
`ocr/pipeline.py` like any other region — this card only changes what `kind` it ends up with, which is enough
for every existing downstream exclusion to already apply; skipping the OCR pass itself for zero-text-value
regions is a possible future optimisation, not this card), a GUI/CLI way to preview which regions a stored
watermark box would catch before running `detect` for real, fuzzy/partial-page-range watermark boxes (today's
`WatermarkRegion` already applies to every page of the series uniformly — that's existing behaviour, unchanged
here), re-running the qualification suite.

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_detect_watermark_position.py tests/unit/test_detect_stage.py -q
uv run pytest -q -m "not gpu"
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
`docs/reports/F2c.md`: Changes, Tests (commands + results), Deviations, Questions. Commit early
(`F2c: WIP watermark_position module`), final commit `F2c: wire fixed-position watermark boxes into detect`.
You have 150 tool calls in total. If anything is unclear: stop, write the question under Questions, commit,
and end.

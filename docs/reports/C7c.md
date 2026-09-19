# C7c — Glyph renderer, GPU compositing, export stage

## Changes

Nothing outside the card's list was touched.

- `src/omniscan/typeset/render.py` — frozen `GlyphPatch` slots dataclass (`x`, `y`, `rgba` uint8
  `[h, w, 4]`, straight alpha) and `render_item(item, *, font_path=None) -> GlyphPatch | None`
  (`None` for empty `lines`). Geometry exactly per the card: `margin = stroke_px + 2`; patch = box
  grown by `margin`; `pitch = box.height // n`; line i anchored `"la"` at
  `y_i = margin + i*pitch + (pitch - (ascent + descent))//2`; pen x by align (center →
  `margin + (box.width - w)/2`, left → `margin`, right → `margin + box.width - w`). Two coverage
  masks (fill at `stroke_width=0`; stroke at `stroke_width=stroke_px`, both `fill=255`), each
  coloured onto a straight-alpha RGBA layer via `putalpha`, composed fill-over-stroke with
  `Image.alpha_composite`. Helpers `_pen_x`, `_layer`.
- `src/omniscan/export/__init__.py` — empty package marker.
- `src/omniscan/export/composite.py` — `apply_patch(strip, box, pixels, mask)` (masked pixels
  replaced in place via `torch.where`; overlap clipped to the strip and to the shorter of
  pixels/mask, so mismatched crops apply their common area at the box origin; fully-outside boxes
  are no-ops), `blend_rgba(strip, patch)` (straight-alpha blend
  `round(src*a + dst*(1-a))`; exactly one host→device upload per patch; alpha 0 leaves the target
  bit-exact), `apply_patches(strip, items, patches) -> int` (item order, missing region ids
  skipped, later entries overwrite earlier ones), `_overlap` helper.
- `src/omniscan/export/stage.py` — `ExportStage` (`name "export"`, `version 1`, `gpu_group None`).
  Inputs: `ingest.json`, `slices.json`, `inpaint.json`, `patches.npz`, `layout.json`, plus
  `inpaint_lama.json`/`patches_lama.npz` when they exist, plus the raw images; outputs
  `["export.json"]`; `config_subset = cfg.export.model_dump()`. `run` raises
  `FileNotFoundError("<file> missing — run the <ingest|slice|inpaint|typeset> stage first")`, loads
  the strip via `load_strip`, applies flat patches then LaMa patches (LaMa overrides the flat
  fill), renders and blends every layout item, and `_write_slices` deletes stale files matching
  `^\d{4}\.jpg$` in the output dir (nothing else), encodes the non-filtered slices as
  `0001.jpg…` with `[export]` quality/subsampling and writes `ExportArtifact`. Metrics (floats):
  `slices`, `bytes`, `patches`, `glyph_items`, `overflow_items`.
- `src/omniscan/cli.py` — `export` removed from `_STUB_COMMANDS`; real `cmd_export`
  (`series`, `--chapter`/`-c` repeatable, `--force`) running only `[ExportStage()]` via
  `_run_stages`. `test_cli.py` itself needed no change (see Deviations).
- Tests — `tests/unit/test_typeset_render.py` (card items 1–4: geometry, line bands without
  fringes, stroke layer, alignment, empty lines, missing font, font_path override),
  `tests/unit/test_export_composite.py` (items 5–6: alpha levels 128/255/0, mixed channels,
  overhang/outside clipping, masked-only replacement, mismatched crops, apply_patches
  order/overwrite/skip, and the card item 12 GPU test — `torch.equal(cpu, gpu.cpu())` on the RX
  9070 XT), `tests/unit/test_export_stage.py` (items 7–11, 13: round-trip with decoded-pixel
  checks, LaMa override, stale-file deletion, quality-change re-run, layout-change re-run,
  missing-input message, metrics, CLI across two chapters, and the synthetic Korean chapter
  exported end to end).
- Docs — README.md status row `export` → working, removed from the Not-implemented table;
  docs/USER_GUIDE.md: stubs sentence, `export.json` + output-folder folder-table rows,
  `export.jpeg_quality`/`export.subsampling` config rows, new `### omniscan export` section, and
  the two sentences that claimed export does not exist (Reader view, `pack`) fixed.
- `docs/reports/C7c.md` — this report.

## Tests

```
uv run pytest tests/unit/test_typeset_render.py tests/unit/test_export_composite.py
  tests/unit/test_export_stage.py tests/unit/test_cli.py tests/unit/test_docs.py --tb=no -p no:warnings
  → 39 passed
uv run pytest --tb=short -p no:warnings
  → 2470 passed in 66.82s (GPU tests included — RX 9070 XT)
uv run ruff format . && uv run ruff check .
  → all clean
uv run pyright
  → 0 errors, 0 warnings, 0 informations
```

Deterministic points the card calls out, verified by the tests:
- Geometry on `ComicNeue-Bold@32` (metrics 29/8, ascent+descent 37): pitch 36, per-line offset −1,
  ink rows 8–65, outer 2-px borders clean, both line bands non-empty, alignment errors ~2 px
  against the 3.2/4.8 px tolerances.
- Alpha blend: 128 over 100 → 150, 255 → 200, 0 → 100 bit-exact; (10,100,250,51) over
  (250,100,10) → (202,100,58).

## Deviations

- **Test 2 (stroke colours).** The card's test text says "pixels with alpha == 255 are either
  white or black", but the card's own normative recipe (fill mask + stroke mask, fill composited
  over stroke via `Image.alpha_composite`) mathematically yields greys at alpha 255 wherever an
  anti-aliased fill edge lands on the opaque stroke. Verified against PIL: a single
  `stroke_width` text draw produces identical pixels (same greys). Implemented the recipe
  exactly; the test asserts the true invariants (alpha > 0 ⇒ RGB is a greyscale mix of only the
  two colours, pure white and pure black both present, stroke ink box larger, patch grows by
  2·(stroke_px+2) relative to the box). Question below.
- **Positional `BBox` args.** The card's test snippets construct `BBox` positionally; pydantic v2
  `BaseModel` rejects positional args, so tests use kwargs (`BBox(x0=…, y0=…, x1=…, y1=…)`).
- **JPEG boundary ringing.** After encoding, an ~8-level ring can appear at the 8×8 block
  boundary between the white rectangle and the red page colour (quality 95); interior deviation
  is ≤ 2. The white-rect and LaMa-colour assertions therefore check the rectangle inset by 2 px,
  with a comment in the test.
- **`tests/unit/test_cli.py` unchanged.** Its `STUB_COMMANDS` tuple is the help-output
  completeness list (every registered command must appear in `--help`), not a list of stubs —
  `export` still belongs there as a real command, so the card's "modify only as stub removal
  requires" meant no change was required.
- **pyright narrowing.** `make_korean_page` types `bubble_bbox` as `BBox | None`; the synthetic
  Korean test asserts it is not None (all three regions are bubbles) before `inscribed_box`.

## Questions

- The stroke/fill alpha-255 greys above: the recipe was followed exactly and PIL's own
  single-call stroked text agrees pixel-for-pixel, so this looks like a test-text slip in the
  card rather than a spec error. If pure white/black at alpha 255 is actually wanted, the
  composition order (stroke over fill, or one stroked draw call) would need to change — please
  confirm the recipe stands.
## Review addendum (director)
- Answer to the Question: the recipe stands. An anti-aliased fill edge over an opaque stroke legitimately produces greys at alpha 255 (PIL's own single-call stroked text agrees pixel for pixel); the card's "either white or black" wording was a slip.
- Rebase conflicts (README status/stub tables, USER_GUIDE tables and stub sentence, the stub tuple and a stale `cmd_typeset` stub in `cli.py`) resolved: `detect`, `ocr`, `judge`, `inpaint`, `typeset`, `export` are real commands now.
- Mutation check, 25 mutants over `typeset/render.py`, `export/composite.py`, `export/stage.py` (margin, pitch/vertical centring, alignment, stroke order, colour bleed, patch origin, mask ignored, clipping offsets, alpha scale, floor vs round, dst attenuation, negative origin, bottom clip, patch count,
  filtered slices, numbering, stale-file deletion scope, LaMa override, hard-coded quality, overflow metric): **all 25 killed**.
- **Live run** on the KoreanDemo chapter with placeholder English lines (`final.json` hand-built): `inpaint` 1.3 s, `typeset` 0.14 s, `export` 0.73 s. Bubble interiors are cleaned and English is fitted and drawn inside them; free text on the gradient keeps its Korean (flagged `needs_lama`, LaMa card C6b) with the English drawn over it, as expected at this stage.
- **The live run exposed an unrelated pre-existing bug** (fixed separately on `main`): `TurboCodec.decode_into` reuses ONE pinned staging buffer with `non_blocking=True` copies on CUDA, so pages overwrite each other — the exported strip showed page 3 three times. CPU tests cannot see it.

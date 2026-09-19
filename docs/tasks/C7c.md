# C7c — Glyph renderer, GPU compositing and the `export` stage: `omniscan export` writes the English slices

**Owner:** GLM builder · **Branch:** `C7c` · **Worktree:** `V:\OmniScan-wt\C7c` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md`, `docs/ARCHITECTURE.md` (the "Render pass" section — it defines what export reads and writes) and `docs/DECISIONS.md` first. Then read the merged pieces this card connects: `src/omniscan/inpaint/patches.py` (`load_patches`),
`src/omniscan/typeset/fit.py` + `fonts.py` (`load_font`, `fonts_dir`, `line_height`), `LayoutItem`, `LayoutArtifact`, `InpaintArtifact`, `SlicesArtifact`, `Slice`, `ExportArtifact`, `ExportFile`, `BBox`, `RGB` in `src/omniscan/core/schemas.py`,
`ExportConfig` in `src/omniscan/core/config.py` (already present), `src/omniscan/ingest/strip.py` (`load_strip`), `src/omniscan/gpu/codec/base.py` + `select.py` (`JpegCodec.encode`, `get_codec`), `ChapterPaths.output_dir` in `src/omniscan/core/paths.py`,
and `tests/fixtures/korean_pages.py`, `tests/fixtures/images.py`.

## Goal
Produce the released English chapter. Export decodes the chapter strip once (on the GPU in production), replaces the **masked** pixels of every cleaned patch (`patches.npz`, then `patches_lama.npz` when it exists), draws every `LayoutItem` (English lines) onto it, cuts the strip along `slices.json` (skipping
`filtered` slices) and writes `output_root/<Series>/<Chapter>/0001.jpg …` plus `export.json` in the chapter work dir. Glyph drawing (FreeType via PIL) is the one unavoidable CPU step: each item becomes a **small RGBA patch** on the CPU and is alpha-blended into the strip **on the strip's device** —
never per pixel in Python, never the whole strip on the CPU except through the codec's encode (a slice at a time). Unmasked pixels of the raw pages are never modified.

## Files you may create / modify
- `src/omniscan/typeset/render.py`, `src/omniscan/export/__init__.py` (empty), `export/composite.py`, `export/stage.py` (create)
- `src/omniscan/cli.py` (modify — ONLY replace the `export` stub with the real command and remove `"export"` from `_STUB_COMMANDS`)
- `tests/unit/test_typeset_render.py`, `test_export_composite.py`, `test_export_stage.py` (create); `tests/unit/test_cli.py` (modify — only what the stub removal requires)
- `README.md` (status row `export` → `working — omniscan export (needs inpaint.json, patches.npz, layout.json)`; drop `export` from the "Not implemented yet" table) and `docs/USER_GUIDE.md` (a `### omniscan export` subsection in the style of the others, the `export.json` row and the output-folder row of
  the folder table, every `[export]` config key in the config table; fix the sentences that say the export stage does not exist — Reader view and `pack`) — `tests/unit/test_docs.py` must stay green
- `docs/reports/C7c.md` (create)
Do not modify `src/omniscan/core/**` (`ExportConfig`/`ExportArtifact` are finished), the C7a/C7b/C6a modules (import from them) or the web code.

## Part 1 — `typeset/render.py` (CPU, PIL + numpy)
```python
@dataclass(frozen=True, slots=True)
class GlyphPatch:
    x: int                      # top-left in strip pixels (may be negative)
    y: int
    rgba: np.ndarray            # uint8 [h, w, 4], STRAIGHT (non-premultiplied) alpha

def render_item(item: LayoutItem, *, font_path: Path | None = None) -> GlyphPatch | None
```
- `None` when `item.lines` is empty. Font = `load_font(font_path or fonts_dir() / item.font, item.size_px)`.
- Geometry: `margin = item.stroke_px + 2`; the patch covers `item.box` grown by `margin` on every side: `x = box.x0 - margin`, `y = box.y0 - margin`, width `box.width + 2*margin`, height `box.height + 2*margin`.
  `n = len(item.lines)`, `pitch = item.box.height // n`, `ascent, descent = font.getmetrics()`; line `i` is drawn with anchor `"la"` at `y_i = margin + i*pitch + (pitch - (ascent + descent)) // 2` and
  `x_i` by `item.align` with `w_i = font.getlength(line)`: `"center"` → `margin + (item.box.width - w_i) / 2`, `"left"` → `margin`, `"right"` → `margin + item.box.width - w_i`.
- Drawing without dark fringes: build two `"L"` coverage masks of the patch size — `fill_mask` (every line drawn with `fill=255`, `stroke_width=0`) and, when `item.stroke_px > 0`, `stroke_mask` (every line drawn with `fill=255`, `stroke_width=item.stroke_px`, which covers glyph + outline).
  Compose in float or with `Image.alpha_composite` on straight-alpha RGBA layers: layer 1 = `item.stroke_color` with alpha `stroke_mask` (only when stroke), layer 2 = `item.color` with alpha `fill_mask`; result = layer 2 over layer 1. Wherever the result's alpha is > 0 its RGB is a mix of only the two colours (no black fringes from a transparent black background).
  Return `GlyphPatch(x, y, np.asarray(result))`.

## Part 2 — `export/composite.py` (tensor code, no Python loops over pixels)
```python
def apply_patch(strip: torch.Tensor, box: BBox, pixels: np.ndarray, mask: np.ndarray) -> None
def blend_rgba(strip: torch.Tensor, patch: GlyphPatch) -> None
def apply_patches(strip: torch.Tensor, items: Sequence[InpaintItem], patches: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> int
```
- `strip` is uint8 `[3, H, W]`, modified **in place**. `apply_patch`: the region `strip[:, box.y0:box.y1, box.x0:box.x1]` (clamped to the strip; the arrays are cropped identically) gets `pixels` (HWC uint8 → `[3,h,w]` on the strip's device) exactly where `mask` (HW bool) is True and stays unchanged elsewhere.
  A patch entirely outside the strip is ignored.
- `blend_rgba(strip, patch)`: clip the patch to the strip (negative `x`/`y` and overhang allowed); with `a = rgba[..., 3] / 255` (float32) the new pixel is `round(src_rgb * a + dst * (1 - a))` clamped to `0..255`, written back as uint8; alpha 255 replaces, alpha 0 leaves the pixel untouched (bit-exact). One host→device upload per patch (`torch.from_numpy(...).to(strip.device)`), all arithmetic on the strip's device.
- `apply_patches(strip, items, patches)`: for each `InpaintItem` whose `region_id` has an entry in `patches` (in item order) call `apply_patch(strip, item.box, *patches[item.region_id])`; returns the number of patches applied. Items without an entry are skipped.

## Part 3 — `export/stage.py`, CLI
```python
class ExportStage:
    name: ClassVar[str] = "export"
    version: ClassVar[int] = 1
    gpu_group: ClassVar[str | None] = None
```
- `inputs(ctx)` = `[ingest.json, slices.json, inpaint.json, patches.npz, layout.json]` plus `inpaint_lama.json` and `patches_lama.npz` **when they exist**, plus `*list_images(raw_dir)`; `outputs(ctx)` = `["export.json"]`; `config_subset(cfg)` = `cfg.export.model_dump()`.
- `run(ctx, models)`: a missing required input → `FileNotFoundError("<file> missing — run the <ingest|slice|inpaint|typeset> stage first")`. Steps: `ingest = IngestArtifact.load(...)`; `strip = load_strip(ctx, ingest)`; `items = InpaintArtifact.load(inpaint.json).items`, `patches = load_patches(patches.npz)`, `apply_patches(strip, items, patches)`;
  when `patches_lama.npz` and `inpaint_lama.json` exist do the same with them **afterwards** (LaMa overrides the flat-fill placeholder); `layout = LayoutArtifact.load(layout.json)`; for every `LayoutItem` `render_item(item)` and `blend_rgba(strip, patch)` (skip `None`);
  then for the `slices.json` slices with `filtered == False`, in order, number `k = 1, 2, …`: `data = codec.encode(strip[:, s.y0:s.y1, :], quality=cfg.export.jpeg_quality, subsampling=cfg.export.subsampling)` with `codec = get_codec(ctx.cfg)` (closed afterwards if it has `close`), written to `ctx.paths.output_dir / f"{k:04d}.jpg"`
  (create the directory; **first delete the files of this directory that match `^\d{4}\.jpg$`** so a re-export with fewer slices leaves no stale files; nothing else is deleted). Write `ExportArtifact(quality, subsampling, files=[ExportFile(name, slice_index, width, height, bytes)])` to `export.json`.
  Metrics (all `float`): `slices` (written), `bytes` (total), `patches` (applied, flat + LaMa), `glyph_items` (rendered), `overflow_items` (layout items with `overflow` true).
- `omniscan export SERIES [--chapter/-c CHAPTER]... [--force]`: same options and help style as `slice`; runs only `[ExportStage()]` through `_run_stages`.

## Acceptance tests (CPU tensors unless marked `gpu`; real OFL fonts from `fonts/`)
**Render (`typeset/render.py`):** 1. `LayoutItem(region_id="r1", font_role="dialogue", font="ComicNeue-Bold.ttf", size_px=32, lines=["Hello", "world"], box=BBox(100,200,220,272), align="center", color=(0,0,0))` (`stroke_px=0`): the patch is at `(98, 198)` with `rgba.shape == (76, 124, 4)` (box 120×72 grown by 2 px on every side) — assert `x == box.x0 - 2`, `y == box.y0 - 2` and that shape; alpha is non-zero somewhere in each of the two line bands (rows `[2, 2+36)` and `[2+36, 2+72)`, pitch 36) and zero in the outer 2-px border;
   wherever alpha > 0 the RGB is exactly `(0,0,0)` (no fringes); 2. with `stroke_px=3`, `color=(255,255,255)`, `stroke_color=(0,0,0)`: shape grows by 2·(3+2); pixels with alpha == 255 are either white or black; some pixels are white and some black; the alpha bounding box is larger than without stroke;
   3. alignment: for one line, the horizontal centre of the alpha bounding box is within `0.1 * size_px` of the box centre for `"center"`, the bounding box starts within `0.15 * size_px` of `box.x0` for `"left"` and ends within `0.15 * size_px` of `box.x1` for `"right"` (use a box wider than the text); 4. `lines=[]` → `None`; a missing font → `FileNotFoundError` (from `load_font`); `font_path` override works.
**Composite (`export/composite.py`):** 5. `blend_rgba`: dst 100 (all channels) with src `(200,200,200)` alpha 128 → 150; alpha 255 → 200; alpha 0 → 100 (bit-exact); mixed channels `(10, 100, 250)` over `(250, 100, 10)` with alpha 51 (0.2) → `(202, 100, 58)`; a patch hanging over the strip's left/top/right/bottom edges (negative `x`, `y`) blends only the overlapping part and
   does not crash; a patch entirely outside changes nothing; the strip is modified in place and stays uint8; 6. `apply_patch`: masked pixels replaced, unmasked untouched, a patch partly outside the strip is clipped, mismatched crops handled; `apply_patches` counts applied patches, skips items without an entry, applies in item order (a second entry for the same box overwrites the first).
**Stage (CPU config as in `tests/unit/test_stages_ingest_slice.py`; pages written with `tests/fixtures/images.py`; hand-built artifacts):** 7. a chapter of 2 solid-colour JPEG pages (800×600 each), `slices.json` with 3 slices (one `filtered`), one `InpaintItem` with a patch whose mask covers a 100×40 rectangle (white pixels) and one `LayoutItem`: output has `0001.jpg`, `0002.jpg` (the filtered slice is skipped), sizes match the slices, `export.json` lists them with byte sizes equal to the files;
   decoded JPEGs (PIL): the patched rectangle is white within tolerance 6, a pixel outside the mask keeps the page colour within 6, and the text area contains pixels that differ from the background (the rendered English); 8. `patches_lama.npz` + `inpaint_lama.json` override the flat patch of the same region (the rectangle takes the LaMa colour); a chapter without the LaMa files works;
   9. re-export with fewer slices deletes the stale `0002.jpg` but not other files in the folder (e.g. `notes.txt`, `cover.png`); second run `skipped`; changing `export.jpeg_quality` re-runs it (and yields a different byte size); changing `layout.json` or any input re-runs it; a missing required input → recorded failure with the documented message;
   10. metrics `slices/bytes/patches/glyph_items/overflow_items` as defined; 11. `omniscan export S` (CLI runner) runs only the export stage, prints one line per chapter, exit 0; `--force`/`--chapter` work; unknown series → exit 2; `omniscan slice S` unaffected; `tests/unit/test_docs.py` and the whole suite stay green.
**GPU (`@pytest.mark.gpu`):** 12. `blend_rgba` and `apply_patch` on a strip on the GPU give bit-identical results to the same operations on a CPU copy of the strip; the output strip stays on the GPU device until `encode`.
**Synthetic Korean chapter:** 13. (uses `make_korean_page`/`to_regions_artifact`) render a page's regions with English placeholder lines through `plan_layout`-style hand-built `LayoutItem`s (one per bubble, box = the truth bubble's inner area) after `apply_patches` with hand-built flat-fill patches (mask = truth `text_mask` cropped to each region's bbox grown by 8, pixels = `page.clean` crop):
    the exported page equals `page.clean` outside the layout boxes (mean absolute difference < 3 after JPEG) and contains rendered English inside every layout box (at least 30 px differ from `page.clean` by more than 60 per box).

## Out of scope
The judge/typeset/inpaint stages themselves, LaMa, JPEG settings other than quality/subsampling, WebP/PNG/CBZ (`omniscan pack` exists), soft edges for patches, sub-pixel text positioning, per-line width adaptation, a GPU glyph rasteriser, changing `ExportConfig`, the web UI.

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_typeset_render.py tests/unit/test_export_composite.py tests/unit/test_export_stage.py tests/unit/test_cli.py tests/unit/test_docs.py -q
uv run pytest -q
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
Write `docs/reports/C7c.md` (Changes, Tests, Deviations, Questions) and commit `C7c: glyph renderer, compositing and export stage`. If anything above is unclear: stop, write the question under Questions, commit what you have, and end.

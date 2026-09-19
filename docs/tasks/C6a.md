# C6a — Inpaint v1: text masks and flat fill → `inpaint.json` + `patches.npz`

**Owner:** GLM builder · **Branch:** `C6a` · **Worktree:** `V:\OmniScan-wt\C6a` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md`, `docs/ARCHITECTURE.md` (the "Render pass" section — it defines the artifacts you write) and `docs/DECISIONS.md` first. Then read
`InpaintItem`, `InpaintArtifact`, `Region`, `OcrLine`, `RegionsArtifact`, `BBox` in `src/omniscan/core/schemas.py`, `InpaintConfig` in `src/omniscan/core/config.py` (already present),
`src/omniscan/ingest/strip.py` (`load_strip`), `src/omniscan/slicer/stage.py` (a stage without a GPU group) and `tests/fixtures/korean_pages.py` (synthetic Korean pages with the text mask and the text-free page).

## Goal
Remove the source text from the chapter in the simplest way that is right for speech bubbles: for every OCR region a **mask** covers its text lines, and where the surroundings of the text are one flat
colour (a white or dark bubble interior) the masked pixels are replaced by exactly that colour ("flat fill"). Regions on textured art cannot be cleaned this way in v1; they are recorded with
`needs_lama = True` so that a later card (LaMa) can redo them. The result is `inpaint.json` (`InpaintArtifact`) and `patches.npz` (cleaned crops + masks per region); export (a later card) applies them.
All pixel work is tensor code on the strip's device (GPU in production, CPU in tests) — no Python loops over pixels or rows, no PIL/numpy in the compute path; numpy is used only to write `patches.npz`.

## Files you may create / modify
- `src/omniscan/inpaint/__init__.py` (create, empty), `inpaint/flat.py`, `inpaint/patches.py`, `inpaint/pipeline.py`, `inpaint/stage.py` (create)
- `src/omniscan/cli.py` (modify — ONLY replace the `inpaint` stub with the real command and remove `"inpaint"` from `_STUB_COMMANDS`)
- `tests/unit/test_inpaint_flat.py`, `test_inpaint_patches.py`, `test_inpaint_pipeline.py`, `test_inpaint_stage.py` (create); `tests/unit/test_cli.py` (modify — only what the stub removal requires)
- `README.md` (status row `inpaint` → `working — omniscan inpaint (flat fill only; needs ocr.json)`; drop `inpaint` from the "Not implemented yet" table) and `docs/USER_GUIDE.md`
  (a `### omniscan inpaint` subsection in the style of the others, the `inpaint.json` / `patches.npz` rows of the folder table, every `[inpaint]` config key in the config table) — `tests/unit/test_docs.py` must stay green
- `docs/reports/C6a.md` (create)
Do not modify `src/omniscan/core/**` (`InpaintConfig` and `InpaintArtifact` are finished).

## Part 1 — `inpaint/flat.py`
```python
@dataclass(frozen=True, slots=True)
class FlatResult:
    pixels: torch.Tensor          # uint8 [3, h, w], the cleaned crop (the input crop unchanged where mask is False)
    fill: tuple[int, int, int] | None
    ok: bool                      # True when the ring was uniform enough for a flat fill

def line_mask(height: int, width: int, boxes: Sequence[tuple[int, int, int, int]], *, dilate_px: int,
              device: torch.device) -> torch.Tensor
def flat_fill(crop: torch.Tensor, mask: torch.Tensor, *, flat_tol: float, min_ring_px: int) -> FlatResult
```
- `line_mask`: bool `[height, width]` on `device`; for each box `(x0, y0, x1, y1)` (patch coordinates, half-open) the rectangle grown by `dilate_px` on every side and clamped to the mask is set True (slice assignment per box).
- `flat_fill(crop [3,h,w] uint8, mask [h,w] bool, ...)`: the **ring** is every pixel where `mask` is False. If the ring has fewer than `min_ring_px` pixels, or the mask is empty → `FlatResult(crop.clone(), None, False)`.
  Otherwise `ring = crop[:, ~mask]` (`[3, N]`); `median = ring.float().median(dim=1).values` (per channel; `median` of an even count = the lower middle element, torch's behaviour);
  `dev = (ring.float() - median[:, None]).abs().amax(dim=0)` (per pixel, worst channel); `q = torch.quantile(dev, 0.9)`. If `q <= flat_tol` → `ok=True`, `fill = tuple(int(v) for v in median.round())`,
  `pixels = crop.clone()` with `pixels[:, mask] = fill` (broadcast); else `ok=False`, `fill=None`, `pixels = crop.clone()` (unchanged). (`torch.quantile` needs float input and < 16 M elements; a ring is far smaller — if `N` exceeds 1 000 000, quantile over a strided sample `dev[::ceil(N/1_000_000)]`.)

## Part 2 — `inpaint/patches.py`
```python
def save_patches(path: Path, patches: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> None   # id -> (pixels [3,h,w] uint8, mask [h,w] bool)
def load_patches(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]                       # id -> (pixels uint8 [h,w,3], mask bool [h,w])
```
`save_patches` writes with `np.savez_compressed` to a temporary file in the same directory and renames it over `path` (atomic); keys `"<region_id>.pixels"` (HWC uint8) and `"<region_id>.mask"` (bool); one `.cpu()` per array. Zero patches → a valid empty npz.
`load_patches` returns arrays in the same layout (HWC / HW); it never returns a key that lacks its partner.

## Part 3 — `inpaint/pipeline.py`
```python
def inpaint_regions(strip: torch.Tensor, regions: Sequence[Region], cfg: InpaintConfig
                    ) -> tuple[InpaintArtifact, dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, float]]
```
`strip` is uint8 `[3, H, W]`. Process `regions` in the given order; a region is **skipped entirely** (no item, no patch) when its `kind == "watermark"` or it has no `lines`. For every other region:
1. `union` = the union of its line boxes (`OcrLine.bbox`); `patch = BBox(x0 = max(0, union.x0 - cfg.pad_px), y0 = ..., x1 = min(W, union.x1 + cfg.pad_px), y1 = min(H, ...))`.
2. `crop = strip[:, patch.y0:patch.y1, patch.x0:patch.x1]`; line boxes shifted into patch coordinates; `mask = line_mask(h, w, boxes, dilate_px=cfg.mask_dilate_px, device=strip.device)`.
3. `kind == "sfx"` → never flat: `InpaintItem(method="none", fill=None, needs_lama=True, mask_px=int(mask.sum()))`, pixels = the unchanged crop.
4. Otherwise `flat_fill(crop, mask, flat_tol=cfg.flat_tol, min_ring_px=cfg.min_ring_px)`: `ok` → `method="flat"`, `fill=result.fill`, `needs_lama=False`; not ok → `method="none"`, `fill=None`, `needs_lama=True`, pixels = the unchanged crop.
5. The patch entry is `(result.pixels, mask)` keyed by `region.id`; `InpaintItem.box = patch` and `mask_px = int(mask.sum())` (one `.item()` per region is fine).
Metrics (all `float`): `regions` (items), `flat`, `needs_lama`, `skipped` (regions without item), `mask_px` (sum). No regions → an empty artifact, no patches.

## Part 4 — `inpaint/stage.py` and CLI
```python
class InpaintStage:
    name: ClassVar[str] = "inpaint"
    version: ClassVar[int] = 1
    gpu_group: ClassVar[str | None] = None
```
- `inputs(ctx)` = `[ingest.json, slices.json, ocr.json, *list_images(raw_dir)]`; `outputs(ctx)` = `["inpaint.json", "patches.npz"]`; `config_subset(cfg)` = `cfg.inpaint.model_dump()`.
- `run(ctx, models)`: missing `ingest.json` / `slices.json` / `ocr.json` → `FileNotFoundError("<file> missing — run the <ingest|slice|ocr> stage first")`; `strip = load_strip(ctx, IngestArtifact.load(...))`;
  `regions = RegionsArtifact.load(ocr.json).regions`; `inpaint_regions(...)`; `save_patches(patches.npz, ...)` first, then `InpaintArtifact.save(inpaint.json)`; return the metrics.
- `omniscan inpaint SERIES [--chapter/-c CHAPTER]... [--force]`: same options and help style as `slice`; it runs **only** `[InpaintStage()]` through `_run_stages` (the user runs `ocr` first, like `translate`).

## Acceptance tests (CPU tensors; no downloads)
**Mask:** 1. `line_mask(20, 30, [(5,5,10,8)], dilate_px=2, device=cpu)` is True exactly on rows 3..9 (y 3 ≤ y < 10) and columns 3..11 (x 3 ≤ x < 12) — sum 7·9 = 63; a box at the edge is clamped (no negative indices, shape stays `[20,30]`);
   two boxes give the union; no boxes → all False; `dilate_px=0` equals the plain boxes.
**Flat fill:** 2. crop = 3×60×200 all `(250, 250, 250)` with a black rectangle in the middle, mask covering the rectangle plus 2 px → `ok`, `fill == (250,250,250)`, `pixels` equals the constant everywhere; the input tensor is not modified;
3. crop with a uniform ring of `(20, 20, 32)` (dark bubble) → `fill == (20,20,32)`; 4. ring of random noise uniform in 0..255 (seeded `torch.Generator`) → `ok False`, `fill None`, `pixels == crop`;
5. ring with 5 % of its pixels set to black (a bubble outline crossing) but the rest `(240,240,240)` → still `ok` (the 90th percentile ignores them); with 20 % black → `ok False`; the threshold is inclusive (`q == flat_tol` → ok — construct a ring where `q` is exactly 8 with `flat_tol=8.0`);
6. fewer ring pixels than `min_ring_px`, or an empty mask → `ok False` and `pixels == crop`.
**Patches:** 7. round trip of two patches (values, dtypes uint8/bool, HWC/HW layout); an empty mapping gives a loadable empty file; the file is replaced atomically (no `.tmp` left behind); overwriting works.
**Pipeline (hand-built strips and regions):** 8. a white 400×300 strip with a black text rectangle and one `bubble_text` region whose line box is that rectangle → method `flat`, fill `(255,255,255)`, `mask_px` equals the dilated rectangle area, the patch pixels are white everywhere and `box` equals the rectangle grown by `pad_px` (clamped);
9. a `watermark` region and a region without lines produce no item; `sfx` → `method="none"`, `needs_lama=True`, pixels unchanged; a region on a noisy background → `needs_lama=True`, `method="none"`; metrics `regions/flat/needs_lama/skipped/mask_px` as defined; no regions → empty artifact;
10. **Synthetic Korean pages** (`make_korean_page`, `to_regions_artifact`, the page as a uint8 `[3,H,W]` CPU tensor): with `background="flat"`, `n_bubbles=4, n_free=1, n_sfx=1` over 6 seeds: every `bubble_text` region is `flat`; for those regions the share of truth-text pixels that still differ from `page.clean` by more than 40 (max over channels) after applying the patch
    (replace `mask` pixels of the crop) is below 1 %, and pixels outside the mask are untouched (the patch pixels equal the original crop there); `free_text` regions on the flat background are `flat` as well (their ring is the flat background); `sfx` → `needs_lama`.
    With `background="noise"` every `free_text` region has `needs_lama=True`.
**Stage / CLI:** 11. `InpaintStage` with a CPU config (pattern of `tests/unit/test_stages_ingest_slice.py`), hand-written `ocr.json` and generated JPEG pages: writes `inpaint.json` (loadable) and `patches.npz` (its keys match the items), manifest `done`, second run `skipped`;
    changing `cfg.inpaint.flat_tol` or `ocr.json` re-runs it; a missing `ocr.json` → recorded failure with the documented message; 12. `omniscan inpaint S` runs only the inpaint stage, prints one line per chapter, exit 0; `--force`, `--chapter` work; unknown series → exit 2;
    `omniscan slice S` is unaffected; `test_docs.py` and the whole suite stay green.

## Out of scope
LaMa or any model, GPU groups, soft edges / feathering, glyph rendering, typesetting, export, re-detection of text in the cleaned image, colour correction, bubble-outline handling, changing `InpaintConfig`.

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_inpaint_flat.py tests/unit/test_inpaint_patches.py tests/unit/test_inpaint_pipeline.py tests/unit/test_inpaint_stage.py tests/unit/test_cli.py tests/unit/test_docs.py -q
uv run pytest -q
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
Write `docs/reports/C6a.md` (Changes, Tests, Deviations, Questions) and commit `C6a: flat-fill inpainting and omniscan inpaint`. If anything above is unclear: stop, write the question under Questions, commit what you have, and end.

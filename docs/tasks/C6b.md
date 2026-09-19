# C6b — LaMa inpainting stage: `inpaint_lama` → `inpaint_lama.json` + `patches_lama.npz`

**Owner:** GLM builder · **Branch:** `C6b` · **Worktree:** `V:\OmniScan-wt\C6b` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md`, `docs/ARCHITECTURE.md` (the "Render pass" section, incl. the `inpaint_lama` bullet), `docs/benchmarks/lama-probe.md` (**every design choice below is measured there**) and `docs/DECISIONS.md` first.
Then read the merged flat-fill code: `src/omniscan/inpaint/{flat,patches,pipeline,stage}.py`, `InpaintConfig` in `src/omniscan/core/config.py` (the `lama_*` keys are already present), `InpaintItem`/`InpaintArtifact`/`BBox` in
`src/omniscan/core/schemas.py`, `src/omniscan/gpu/groups.py`, `src/omniscan/gpu/vram.py`, `src/omniscan/ingest/strip.py` (`load_strip`), and how `cmd_inpaint` / `_run_stages` are wired in `src/omniscan/cli.py`.

## Goal
The flat-fill stage (C6a) leaves regions on textured art marked `needs_lama = True`. This card cleans exactly those with **LaMa** (TorchScript `big-lama.pt`, Apache-2.0): for every such region a fixed 512×512 window of the strip around it is inpainted on the GPU
and the result is stored as a patch in `patches_lama.npz` (same layout as `patches.npz`), recorded in `inpaint_lama.json`. Export (a later card) applies `patches.npz` first and `patches_lama.npz` after it. Facts you must respect (measured):
**fp32 only** (fp16 fails), **one fixed window shape** (each new shape costs a 10–25 s warm-up; steady state is 30 ms per 512² crop), the weights are 205 MB and must be fetched once with a **sha256 check**.
Unit tests use a fake inpainter and a mocked HTTP transport (no network, no GPU); one `@pytest.mark.gpu` test runs the real model.

## Files you may create / modify
- `src/omniscan/inpaint/lama_weights.py`, `inpaint/lama.py`, `inpaint/lama_pipeline.py`, `inpaint/lama_stage.py` (create)
- `src/omniscan/gpu/groups.py` (modify — ONLY register the new group, see Part 4)
- `src/omniscan/cli.py` (modify — ONLY add the `--lama` flag to the `inpaint` command, see Part 4)
- `tests/unit/test_lama_weights.py`, `test_lama_pipeline.py`, `test_lama_model.py`, `test_lama_stage.py` (create); `tests/unit/test_gpu_groups.py`, `tests/unit/test_cli.py` (modify — only what the changes require)
- `README.md` (status row `inpaint` → `working — omniscan inpaint (flat fill; --lama for textured art)`) and `docs/USER_GUIDE.md` (extend the `### omniscan inpaint` section with `--lama`, the `inpaint_lama.json` / `patches_lama.npz` rows of the folder table, the six `inpaint.lama_*` keys in the config table) — `tests/unit/test_docs.py` must stay green
- `docs/reports/C6b.md` (create)
Do not modify `src/omniscan/core/**`, `pyproject.toml`, the C6a modules (import from them) or the web code.

## Part 1 — `inpaint/lama_weights.py`
```python
def lama_path(models_dir: Path, cfg: InpaintConfig) -> Path                  # models_dir / "lama" / cfg.lama_file
def sha256_file(path: Path) -> str
def ensure_lama_weights(models_dir: Path, cfg: InpaintConfig, *, client: httpx.Client | None = None) -> Path
```
- `ensure_lama_weights`: if the file exists and `sha256_file(path) == cfg.lama_sha256` → return it (no request). Otherwise download `cfg.lama_url` (follow redirects, streamed in 1 MiB chunks, `timeout=None` for the body) into `<file>.part` in the same directory
  while hashing; a non-200 status → `RuntimeError(f"LaMa download failed: HTTP {status}")`; a hash mismatch → delete the `.part` file and `RuntimeError("LaMa weights checksum mismatch: expected <sha> got <sha>")`; on success rename the `.part` file over the target (atomic replace) and return the path. A stale wrong-hash file is replaced.
  `client` is injectable for tests (`httpx.Client(transport=httpx.MockTransport(...))`); create and close a private client when `None`. Never log the hash of anything but the two values above.

## Part 2 — `inpaint/lama.py`
```python
class LamaInpainter:
    def __init__(self, model: Any, device: torch.device, *, window: int) -> None: ...
    @property
    def window(self) -> int: ...
    @classmethod
    def load(cls, cfg: InpaintConfig, models_dir: Path, device: torch.device) -> LamaInpainter: ...
    def inpaint(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: ...
```
- `load`: `path = ensure_lama_weights(models_dir, cfg)`; on a CUDA device `torch.cuda.set_device(device)` first (MIOpen launches on the *current* device); `model = torch.jit.load(str(path), map_location=device).eval()` (**no `.half()`**);
  warm-up: two forward passes on zero tensors of the fixed window shape (`[1,3,W,W]` image, `[1,1,W,W]` mask) under `torch.inference_mode()`, with a `log.info("warming up LaMa (one-time, ~10-25 s)")` before; return `cls(model, device, window=cfg.lama_window)`.
- `inpaint(image uint8 [3,S,S], mask bool [S,S])` with `S == self.window` (else `ValueError`): `x = image.to(device).float().div(255)[None]`, `m = mask.to(device).float()[None, None]`; `y = self._model(x, m)` under `torch.inference_mode()`;
  `out = (y[0].clamp(0, 1) * 255).round().to(torch.uint8)`; return `torch.where(mask.to(device)[None], out, image.to(device))` — pixels outside the mask are **bit-identical** to the input. The result stays on `device` (no `.cpu()`).

## Part 3 — `inpaint/lama_pipeline.py`
```python
def dilate_mask(mask: torch.Tensor, px: int) -> torch.Tensor                 # bool [h,w] -> bool [h,w]
def window_origin(box: BBox, strip_w: int, strip_h: int, window: int) -> tuple[int, int, int, int]   # (x0, y0, w, h) of the crop actually taken
def lama_regions(strip: torch.Tensor, inpaint: InpaintArtifact, patches: Mapping[str, tuple[np.ndarray, np.ndarray]],
                 inpainter: Any, cfg: InpaintConfig) -> tuple[InpaintArtifact, dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, float]]
```
- `dilate_mask(mask, px)`: `px <= 0` → the same mask (a copy); else `F.max_pool2d(mask.float()[None,None], kernel_size=2*px+1, stride=1, padding=px)[0,0] > 0` (same shape, on the mask's device).
- `window_origin(box, W, H, window)`: `cx = (box.x0 + box.x1) // 2`, `cy` likewise; `w = min(window, W)`, `h = min(window, H)`; `x0 = clamp(cx - window // 2, 0, W - w)`, `y0 = clamp(cy - window // 2, 0, H - h)`; returns `(x0, y0, w, h)`.
- `lama_regions(strip uint8 [3,H,W], inpaint.json content, patches from `load_patches` (id → (pixels HWC uint8, mask HW bool)), inpainter, cfg)`: for every item of `inpaint.items` **with `needs_lama` true** in order:
  1. Its flat-fill mask is `patches[id][1]` (numpy bool `[h,w]`, the size of `item.box`); move it to the strip's device as a torch bool tensor, then `mask = dilate_mask(mask, cfg.lama_dilate_px)`.
  2. If `item.box` is wider or taller than `cfg.lama_window - 2 * cfg.lama_context_px` → **skipped**: no output for it (metrics `skipped_too_large`).
  3. `(wx, wy, w, h) = window_origin(item.box, W, H, cfg.lama_window)`; `window = strip[:, wy:wy+h, wx:wx+w]`; if `w` or `h` is smaller than `cfg.lama_window` (a strip smaller than the window), pad to `S×S` on the right/bottom with `F.pad(..., mode="replicate")` on the image
     (`window.float()[None]` → pad → `.round().to(uint8)[0]`) and with `False` on the mask; build `full_mask` (bool `[S,S]`, False) and set `full_mask[item.box.y0-wy : +bh, item.box.x0-wx : +bw] = mask`.
  4. `result = inpainter.inpaint(padded_window, full_mask)`; the patch pixels are `result[:, item.box.y0-wy : item.box.y1-wy, item.box.x0-wx : item.box.x1-wx]` and the patch mask is the dilated `mask` (both exactly the size of `item.box`).
  5. Output `InpaintItem(region_id, box=item.box, method="lama", fill=None, needs_lama=False, mask_px=int(mask.sum()))` and the patch `(pixels [3,bh,bw] uint8, mask [bh,bw] bool)`.
  Items with `needs_lama` false are ignored. Metrics (all `float`): `regions` (items produced), `skipped_too_large`, `mask_px`. No item to process → an empty artifact, no calls to the inpainter.
  The strip never leaves its device except through `inpaint`'s own device move; use no Python loops over pixels.

## Part 4 — stage, group, CLI
```python
class LamaStage:
    name: ClassVar[str] = "inpaint_lama"
    version: ClassVar[int] = 1
    gpu_group: ClassVar[str | None] = INPAINT_GROUP  # "inpaint", defined in gpu/groups.py
```
- `gpu/groups.py`: `INPAINT_GROUP = "inpaint"`; `build_vram_manager` additionally registers it with the loader `lambda device: {"lama": LamaInpainter.load(cfg.inpaint, cfg.paths.models_dir, device)}` (import inside the loader) and `est_gib=2.0`. The `vision` registration stays as it is.
- `inputs(ctx)` = `[ingest.json, slices.json, inpaint.json, patches.npz, *list_images(raw_dir)]`; `outputs(ctx)` = `["inpaint_lama.json", "patches_lama.npz"]`; `config_subset(cfg)` = `cfg.inpaint.model_dump()`.
- `run(ctx, models)`: a missing input → `FileNotFoundError("<file> missing — run the inpaint stage first")`; `inpainter = models["lama"]`; `strip = load_strip(ctx, IngestArtifact.load(...))`; `lama_regions(...)` with `load_patches(patches.npz)`; `save_patches(patches_lama.npz, ...)` first, then `InpaintArtifact.save(inpaint_lama.json)`; return the metrics.
- CLI: `omniscan inpaint SERIES [--chapter/-c]... [--force] [--lama]`: without `--lama` unchanged (only `InpaintStage`); with `--lama` the stages are `[InpaintStage(), LamaStage()]`. Help text of the flag: "Also clean regions on textured art with LaMa (downloads 205 MB on first use).".

## Acceptance tests (CPU unless marked `gpu`; the inpainter is faked)
**Weights:** 1. an existing file with the right sha256 → returned, no request (mock transport that fails on any call); 2. missing file → downloaded via the mock transport (content bytes of a tiny fake "model", `cfg` copied with `lama_sha256` = their hash), written atomically, no `.part` left;
3. wrong hash from the server → `RuntimeError` mentioning both hashes, no target file and no `.part`; 4. HTTP 404 → `RuntimeError("LaMa download failed: HTTP 404")`; 5. a stale file with a wrong hash is replaced by a correct download; `lama_path` layout.
**Mask / window math:** 6. `dilate_mask` with `px=2` on a single True pixel gives a 5×5 block (25 True), `px=0` returns an equal copy, edges are clipped, the shape/device are kept; 7. `window_origin`: box `(100,100,200,150)` on a 800×1400 strip, window 512 → `(0, 0, 512, 512)`
   (`cx=150` → clamps to 0); box `(700,1300,780,1350)` → `(288, 888, 512, 512)`; a 300×300 strip → `(0, 0, 300, 300)`; box centred in the middle of a big strip → `x0 = cx - 256` exactly.
**Pipeline (fake inpainter):** 8. the fake records `(image, mask)` and returns `image` with every masked pixel set to `(1,2,3)`; one `needs_lama` item on a noisy 800×1400 strip → the call gets a `[3,512,512]` uint8 image equal to the strip window and a mask whose True pixels are exactly the dilated patch mask at the right offset;
   the produced patch has the size of `item.box`, its pixels equal `(1,2,3)` where the mask is True and the original strip pixels elsewhere; `method == "lama"`, `needs_lama False`, `fill None`, `mask_px` equals the mask sum; items with `needs_lama` false are ignored;
9. a region wider than `lama_window - 2*lama_context_px` (e.g. 460 px with window 512, context 32) is skipped (`skipped_too_large == 1`, no output, the fake is not called); 10. strip smaller than the window (300×300): the window is padded to 512×512 by replication, the mask is False in the padding, the output patch still has the size of the box;
11. two items → two calls in order; no item → empty artifact, zero calls; metrics as defined.
**Stage / CLI:** 12. `LamaStage` with a CPU config, hand-built `inpaint.json` + `patches.npz` (write them with C6a's `save_patches`/`InpaintArtifact`) and a fake inpainter in `models`: writes loadable `inpaint_lama.json` and `patches_lama.npz` whose keys match the items, manifest `done`, second run `skipped`, changing `inpaint.lama_window` re-runs;
    a missing `patches.npz` → recorded failure with the documented message; 13. `build_vram_manager` registers `inpaint` (loader monkeypatched) and still `vision`; `omniscan inpaint S` (no flag) runs only the flat stage (`build_vram_manager` must not be called — monkeypatch it to raise); `omniscan inpaint S --lama` runs both (fake manager), prints one line per stage per chapter, exit 0; `test_docs.py` and the whole suite stay green.
**GPU (`@pytest.mark.gpu`; downloads the 205 MB weights on first use — skip the test with `pytest.skip` if the download raises `httpx.HTTPError`):**
14. `LamaInpainter.load(InpaintConfig(), models_dir, device)` (a `tmp_path_factory` session directory shared by the GPU tests) returns an inpainter whose model runs fp32; `inpaint` on `make_korean_page(seed=2, n_bubbles=0, n_free=1, n_sfx=1, background="gradient")`: build the window from the page (800×1400 tensor), the mask = the truth `text_mask` of the free-text region grown by 7 px (use `dilate_mask`),
    and check: pixels outside the mask are bit-identical to the input; inside the mask the mean absolute difference to `page.clean` is below 15 (of 255) — print the number; steady-state time of a second call is printed (do not assert); 15. the full path on the same page: `InpaintStage` (flat) → `LamaStage` with the real model on a hand-built `ocr.json` from `to_regions_artifact`: the free-text and SFX items become `method == "lama"` and applying `patches.npz` then `patches_lama.npz` to the strip leaves at most 5 % of the truth text pixels differing from `page.clean` by more than 60 (max over channels).

## Out of scope
Batching several windows (a different batch size is a different shape → another warm-up), windows other than the configured size, regions larger than the window (skipped by design), text-ink-aware masks, blending/feathering, applying the patches (export card), fp16, changing `InpaintConfig`, the web UI.

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_lama_weights.py tests/unit/test_lama_pipeline.py tests/unit/test_lama_model.py tests/unit/test_lama_stage.py tests/unit/test_gpu_groups.py tests/unit/test_cli.py tests/unit/test_docs.py -q
uv run pytest -q
uv run ruff format . && uv run ruff check .
uv run pyright
```
The full run includes the `gpu` tests (the first one downloads the weights, ~205 MB, and warms LaMa for 10–25 s); report the measured numbers of tests 14 and 15.

## Report
Write `docs/reports/C6b.md` (Changes, Tests, Deviations, Questions) and commit `C6b: LaMa inpainting stage`. If anything above is unclear: stop, write the question under Questions, commit what you have, and end.

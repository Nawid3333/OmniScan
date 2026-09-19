# C6b — LaMa inpainting stage

## Changes

Created (all under `src/omniscan/inpaint/` unless noted):

- `lama_weights.py` — `lama_path`, `sha256_file` (streamed), `ensure_lama_weights`: existing file with
  the expected sha256 is returned untouched; otherwise `cfg.lama_url` is streamed in 1 MiB chunks into
  `<file>.part` while hashing (redirects followed, no body timeout), a non-200 raises
  `RuntimeError("LaMa download failed: HTTP <status>")`, a hash mismatch deletes the `.part` file and
  raises `RuntimeError("LaMa weights checksum mismatch: expected <sha> got <sha>")`, and success
  renames the `.part` file over the target. `httpx.Client` injectable for tests.
- `lama.py` — `LamaInpainter`: `load` fetches the weights, `torch.cuda.set_device(device)` on CUDA,
  `torch.jit.load(...).eval()` in fp32 (no `.half()`), two zero-tensor warm-up passes on the fixed
  `[1,3,W,W]` / `[1,1,W,W]` shape after a `log.info("warming up LaMa (one-time, ~10-25 s)")`;
  `inpaint(image uint8 [3,S,S], mask bool [S,S])` validates `S == window`, runs the model under
  `torch.inference_mode()` and returns `torch.where(mask, out, image)` so pixels outside the mask are
  bit-identical to the input; the result stays on the device.
- `lama_pipeline.py` — `dilate_mask` (max-pool, `px<=0` → copy), `window_origin` (box-centred window
  clamped into the strip), `lama_regions`: for every `needs_lama` item of `inpaint.json`, grows its
  flat-fill mask by `lama_dilate_px`, skips regions wider/taller than
  `lama_window - 2 * lama_context_px` (`skipped_too_large`), replicate-pads the window when the strip
  is smaller than the window (mask False in the padding), inpaints the fixed `[512, 512]` window and
  cuts the region's own patch back out. Output: `InpaintItem(method="lama", fill=None,
  needs_lama=False)` per region + patches `(pixels [3,bh,bw] uint8, mask [bh,bw] bool)` + metrics
  `regions` / `skipped_too_large` / `mask_px`.
- `lama_stage.py` — `LamaStage` (`inpaint_lama`, version 1, `gpu_group = INPAINT_GROUP`): inputs
  ingest/slices/inpaint.json/patches.npz/raw images, outputs `inpaint_lama.json` +
  `patches_lama.npz` (patches saved first), `config_subset = cfg.inpaint.model_dump()`, a missing
  input raises `FileNotFoundError("<file> missing — run the inpaint stage first")`.

Modified:

- `src/omniscan/gpu/groups.py` — `INPAINT_GROUP = "inpaint"`; `build_vram_manager` registers it with
  the loader `{"lama": LamaInpainter.load(cfg.inpaint, cfg.paths.models_dir, device)}` (import inside
  the loader) and `est_gib=2.0`; the vision registration is unchanged.
- `src/omniscan/cli.py` — `omniscan inpaint` gained `--lama` ("Also clean regions on textured art
  with LaMa (downloads 205 MB on first use)."); without the flag the command is unchanged.
- `tests/unit/test_lama_weights.py`, `test_lama_pipeline.py`, `test_lama_model.py`,
  `test_lama_stage.py` (new); `tests/unit/test_gpu_groups.py` (inpaint group registration +
  `inpaint --lama` CLI tests); README status row and `docs/USER_GUIDE.md` (`--lama`, the two folder
  rows, the six `inpaint.lama_*` keys).

No changes to `src/omniscan/core/**`, `pyproject.toml`, the C6a modules or the web code.

## Tests

```
uv run pytest tests/unit/test_lama_weights.py tests/unit/test_lama_pipeline.py \
  tests/unit/test_lama_model.py tests/unit/test_lama_stage.py tests/unit/test_gpu_groups.py \
  tests/unit/test_cli.py tests/unit/test_docs.py -q        -> 41 passed
uv run pytest                                              -> 2466 passed, 5 warnings (incl. both gpu tests)
uv run ruff format . && uv run ruff check .                -> clean
uv run pyright                                             -> 0 errors
```

Measured GPU numbers (RX 9070 XT, ROCm 10, fp32):

- Test 14 (`test_real_lama_removes_text_on_a_gradient_page`): mean absolute difference inside the
  mask to `page.clean` **6.04 of 255** (limit 15); pixels outside the mask bit-identical; steady-state
  second call **42 ms** per 512×512 window (probe measured 30 ms for the bare forward; the extra
  ~12 ms is the `torch.where` compositing and the mask/image device moves).
- Test 15 (`test_lama_stage_full_path_on_a_korean_page`): seed=2, gradient background,
  `n_bubbles=0, n_free=1, n_sfx=1`; after `InpaintStage` (both regions `needs_lama`) → `LamaStage`
  both items have `method == "lama"`; applying `patches.npz` then `patches_lama.npz` to the strip
  leaves **0 / 31390** truth text pixels differing from `page.clean` by more than 60 (limit 5 %).

## Deviations

- `docs/USER_GUIDE.md`: beyond the listed additions, the `paths.models_dir` config row changed from
  "no consumer yet" to the LaMa weights — it would have been false otherwise.
- The full-suite run before the last one had one flaky failure
  (`tests/unit/test_codec_turbo.py::test_decode_into_gpu_strip`, GPU strip shape assertion); it
  passes in isolation and in the final full run. Unrelated to this card (the turbo codec is not
  touched); flagged for the director in case it recurs.
- `LamaStage.run` uses "run the inpaint stage first" as the documented message for *every* missing
  input, including `ingest.json`/`slices.json`, exactly as the card specifies (slightly odd wording
  for the upstream stages, kept per card).

## Questions

- None blocking. For a later card: `torch.jit.load` prints a DeprecationWarning on Python 3.14
  ("switch to torch.export"); when big-lama ships an `torch.export` artefact we should migrate
  (out of scope here — the config pins this exact file/sha256).
## Review addendum (director)
- Rebase conflicts (README status table, USER_GUIDE tables, `est_gib` of the vision group vs the new inpaint group in `gpu/groups.py`, imports and tests in `test_gpu_groups.py`) resolved.
  `test_build_vram_manager_registers_inpaint` predated C4a's change to the vision group (it loaded the real OCR models when acquired): it now fakes all three vision loaders and only checks `["detector"]`.
- The flaky `test_decode_into_gpu_strip` failure the builder saw is the CUDA staging-buffer race in `TurboCodec.decode_into` (fixed on `main` in `8c3a32f`; a large-page GPU regression test guards it).
- Mutation check, 21 mutants over `lama_pipeline.py` and `lama_weights.py` (dilation kernel/padding, window origin centring and both clamps, width limit, context limit and its `or`, `needs_lama` inverted, replicate padding, offset into the window, method name, `mask_px`,
  patch mask dropped, skipped metric, no extra dilation, hash check inverted, checksum mismatch ignored, HTTP status ignored, `.part` left behind): **all killed**.

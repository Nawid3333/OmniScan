# C6a — Inpaint v1: text masks and flat fill → `inpaint.json` + `patches.npz`

## Changes

- `src/omniscan/inpaint/__init__.py` — empty package marker.
- `src/omniscan/inpaint/flat.py` — frozen `FlatResult` (pixels / fill / ok), `line_mask`
  (bool `[h, w]` on the given device; per box one slice assignment with explicit `max(0, …)` /
  `min(size, …)` clamps — negative slice starts would wrap in Python) and `flat_fill`
  (ring = `crop[:, ~mask]`; empty mask or ring < `min_ring_px` → unchanged clone, not ok;
  per-channel `median(dim=1)`, worst-channel deviation `amax`, 90th-percentile check `q <= flat_tol`
  (inclusive), strided sample of `dev` above 1 000 000 elements for `torch.quantile`; on success the
  masked pixels are overwritten with the rounded median colour; the input crop is never modified).
- `src/omniscan/inpaint/patches.py` — `save_patches` (one `.cpu()` per array, HWC uint8 pixels /
  HW bool mask, `np.savez_compressed` into a `mkstemp` file object in the target directory — a file
  object, so numpy cannot append `.npz` to the name — then `replace` over the target, `unlink`
  in `finally` so no `.tmp` survives; zero patches → a valid empty npz) and `load_patches`
  (pairs `<id>.pixels`/`<id>.mask` by `rpartition(".")`; an id lacking its partner is never returned).
- `src/omniscan/inpaint/pipeline.py` — `inpaint_regions`: skips watermark regions and regions
  without lines entirely; patch = union of line boxes grown by `pad_px`, clamped to the strip; line
  boxes shifted into patch coordinates; `sfx` → `method="none"`, `needs_lama=True`; otherwise
  `flat_fill` decides flat vs. none/lama. Metrics `regions` / `flat` / `needs_lama` / `skipped` /
  `mask_px` (all float). No regions → empty artifact, no patches.
- `src/omniscan/inpaint/stage.py` — `load_strip(ctx, ingest)` (decodes via `get_codec` +
  `build_strip`/`jpeg_paths`, memoised under `ctx.lazy("strip", …)` so a full pass shares one decode;
  the codec is created and closed inside the factory, mirroring `SliceStage`) and `InpaintStage`
  (`name="inpaint"`, `version=1`, `gpu_group=None`; inputs `ingest.json` + `slices.json` +
  `ocr.json` + `list_images(raw_dir)`; outputs `inpaint.json`, `patches.npz`;
  `config_subset = cfg.inpaint.model_dump()`; `run` checks the three upstream artifacts with the
  documented `FileNotFoundError` messages, saves `patches.npz` first, then `inpaint.json`).
- `src/omniscan/cli.py` — `cmd_inpaint` is now real (`series`, repeatable `--chapter/-c`, `--force`,
  same help style as `slice`; runs only `[InpaintStage()]` through `_run_stages`), registered with
  `app.command("inpaint")`; `"inpaint"` removed from `_STUB_COMMANDS`. Nothing else touched.
- `tests/unit/test_inpaint_flat.py` (12 tests), `test_inpaint_patches.py` (4), 
  `test_inpaint_pipeline.py` (16 incl. the two parametrised Korean-page groups), 
  `test_inpaint_stage.py` (5) — all acceptance behaviours 1–12. `tests/unit/test_cli.py`: only
  `"inpaint"` removed from its `STUB_COMMANDS` tuple.
- `README.md` (status row + stub table), `docs/USER_GUIDE.md` (folder-table rows for `inpaint.json`
  and `patches.npz`, the `### omniscan inpaint` subsection, the four `[inpaint]` config keys, inpaint
  dropped from the stub sentence), `docs/reports/C6a.md` (this report).

## Tests

```
uv run pytest tests/unit/test_inpaint_flat.py tests/unit/test_inpaint_patches.py \
  tests/unit/test_inpaint_pipeline.py tests/unit/test_inpaint_stage.py \
  tests/unit/test_cli.py tests/unit/test_docs.py -q
  → 50 passed
uv run pytest -q
  → 2306 passed in 30.74s (the only warnings are the pre-existing turbo-codec ones)
uv run ruff format . && uv run ruff check .
  → all clean
uv run pyright
  → 0 errors, 0 warnings, 0 informations
uv run omniscan inpaint --help
  → shows series / --chapter / --force; `omniscan --help` lists inpaint as a real command
```

Notes on deterministic points the card calls out, verified by the tests:
- Mask geometry: `(5,5,10,8)` + dilate 2 → rows 3–9, cols 3–11, sum 63; edge box clamps; union;
  empty; dilate 0.
- Threshold inclusivity: a ring of exactly 11 pixels with 9 uniform + 2 off-by-8 pixels has
  `torch.quantile(dev, 0.9) == 8.0` exactly (position `0.9·(11−1) = 9` picks `sorted[9]`), and
  `q == flat_tol == 8.0` is accepted.
- Korean pages, `background="flat"`, seeds 0–5: every `bubble_text` and `free_text` item is `flat`,
  `sfx` is `needs_lama`; after applying each patch, < 1 % (measured 0 %) of the truth-text pixels
  differ from `page.clean` by more than 40, and pixels outside the mask equal the original crop.
- Korean pages, `background="noise"`, seeds 0–5: every `free_text` region is `needs_lama`.

## Deviations

- The card's `strip = load_strip(ctx, IngestArtifact.load(...))` referenced a `load_strip` that did
  not exist anywhere yet; it is implemented in `inpaint/stage.py` as described above. Behaviour
  matches the card's usage exactly.
- `flat_fill` builds the fill colour from `median.round().tolist()` instead of iterating the tensor
  (`tuple(int(v) for v in median.round())` is `tuple[int, ...]` to pyright, and would sync the GPU
  once per channel); semantics identical, one host sync instead of three.
- `np.savez_compressed(fh, **arrays)` carries a targeted `# type: ignore[arg-type]` (pyright checks
  the unpacked `dict[str, ndarray]` keys against numpy's `allow_pickle: bool` parameter); the arrays
  are plain numeric/bool arrays, never pickled. The same one-line ignore sits on the orphan-key test,
  which writes an npz by hand.
- Patches for sfx / not-flat regions store a `clone()` of the crop (the card says "the unchanged
  crop") so no patch ever aliases the strip tensor.
- In acceptance test 10 the "share of truth-text pixels" is computed inside the patch box (a
  region's text lies entirely inside its patch, so this equals the global share).

## Questions

- None blocking. One observation for the later LaMa/export cards: `patches.npz` currently always
  contains a full crop per region even when nothing was cleaned (`method="none"`), which is what the
  card specifies ("The patch entry is `(result.pixels, mask)`") — worth remembering when sizing the
  npz for dense chapters, since a `needs_lama` region's pixels are just the raw crop until LaMa or
  export overwrites them.
## Review addendum (director)
- The builder's worktree was cut before C3 merged, so it wrote its own `load_strip` in `inpaint/stage.py` (Deviations). After the rebase it duplicated `omniscan.ingest.strip.load_strip`; the stage now imports the shared one.
- Rebase conflicts (README status/stub table, USER_GUIDE tables and command sections, the stub tuple in `cli.py`/`test_cli.py`) resolved: `detect`, `judge` and `inpaint` are all real commands now.
- Visual check on `make_korean_page(seed=1, n_bubbles=4, n_free=1, n_sfx=1, background="gradient")`: all four bubbles (white ellipse, white rounded boxes, dark box) come out identical to the text-free page (`page.clean`), outlines untouched;
  the free text on the gradient and the SFX are left as they were with `needs_lama=True` (metrics: 6 regions, 4 flat, 2 needs_lama). Exactly the v1 design.
- Answer to the note about the npz size: a `needs_lama` region stores its raw crop until LaMa/export overwrite it; fine for now (`np.savez_compressed`).

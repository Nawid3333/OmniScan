# O1d — OCR engine `paddleocr_vl` (PaddleOCR-VL reads region crops)

## Summary

`paddleocr_vl` is now a third OCR engine next to `ppocr` and `manga_ocr`: a new `PaddleOcrVlReader`
in `ocr/crop_readers.py` loads `PaddlePaddle/PaddleOCR-VL-1.6` (catalog id `ocr-vl-1.6`, the engine
default in `DEFAULT_REC_MODEL`) with `AutoModelForImageTextToText` fp32 and reads every region crop
with **one sequential chat-formatted `generate`** each (`"OCR:"` prompt, greedy, batched generation
explicitly left to a later card — `ocr.crop_batch_size` is ignored). The score is the exp of the
mean generated-token log-probability via `compute_transition_scores(..., normalize_logits=True)`.
The VRAM group loads the reader for this engine and budgets 6.6 GiB (3.0 + 3.6); the stage needed no
code change (the `engine` label in `ocr.json` is `ocr-vl-1.6` via `engine_rec_model`). The O1b
`ValueError("OCR engine 'paddleocr_vl' is not available yet")` is gone.

## Changes

- `src/omniscan/ocr/crop_readers.py`: `VL_PROMPT = "OCR:"`, `VL_MIN_SIDE = 28`,
  `VL_MAX_PIXELS = 1280 * 28 * 28`; `prepare_vl_image` (crop → host → RGB PIL, LANCZOS-upscaled so
  `min(w, h) >= 28`, 1-px crops included); `clean_vl_text` (`" ".join(text.split())`, nothing else);
  `PaddleOcrVlReader` — `load` resolves `engine_rec_model` (None → `ValueError("paddleocr_vl needs
  ocr.rec_model")`), checks the catalog role is `vlm_ocr` (`_check_model_role`), loads model +
  processor fp32 via `model_source`/`load_kwargs` (installed folder with `local_files_only=True`, or
  the pinned hub revision with the standard warning), `torch.cuda.set_device` first on CUDA,
  `.to(device).eval()`; `read` builds the card's exact message list per crop, applies the chat
  template with `images_kwargs={"size": {"shortest_edge": processor.image_processor.size["shortest_edge"],
  "longest_edge": 1280*28*28}}`, slices the decode at the prompt length and scores through
  `compute_transition_scores` (raises → 1.0 + `log.debug`; empty text → `("", 0.0)`; no `.cpu()` in
  scoring except the final `float(...)`).
- `src/omniscan/ocr/engines.py`: `DEFAULT_REC_MODEL["paddleocr_vl"] = "ocr-vl-1.6"`.
- `src/omniscan/gpu/groups.py`: a `paddleocr_vl` branch loading
  `{"reader": PaddleOcrVlReader.load(cfg.ocr, device, models_dir=cfg.paths.models_dir)}` next to the
  always-loaded comic detector (same shape as `manga_ocr`); the group's estimate is now
  `6.6 if engine == "paddleocr_vl" else 3.0` GiB. The defensive unknown-engine `ValueError` stays
  (see Deviations 2).
- `src/omniscan/ocr/stage.py`: only the module docstring (paddleocr_vl is no longer "later") — the
  stage already routes every non-`ppocr` engine through `models["reader"]` and labels with
  `engine_rec_model(cfg.ocr)`.
- `tests/unit/test_ocr_vl_reader.py` (new, 16): the card tests 1–4; `tests/unit/test_ocr_vl_gpu.py`
  (new, 1, `gpu`); `test_ocr_engines.py` (+engine_rec_model/default-table rows), `test_gpu_groups.py`
  (VL group + estimates, unknown-engine test reworked), `test_ocr_stage.py` (+2 VL stage tests).
- `docs/USER_GUIDE.md`: the engine table row is `available` (default `ocr-vl-1.6`, speed/memory),
  the crop-reading paragraph explains sequential reads and `ocr.vl_max_new_tokens`, one more env
  example line; validated by `test_docs.py`.

## Tests

```
uv run --frozen pytest tests/unit/test_ocr_vl_reader.py tests/unit/test_ocr_engines.py
             tests/unit/test_ocr_stage.py tests/unit/test_gpu_groups.py tests/unit/test_docs.py
             tests/unit/test_ocr_crop_readers.py tests/unit/test_ocr_pipeline.py
             tests/unit/test_ocr_model.py -q
  → 99 passed
uv run --frozen pytest -m "not gpu"           → all passed (1 pre-existing xfail), no failures
uv run --frozen ruff format . && uv run --frozen ruff check .
  → clean
uv run --frozen pyright                        → 0 errors, 0 warnings
```

Acceptance tests, mapped to the card:

1. `prepare_vl_image` — 100×60 unchanged (and its pixels equal the tensor's bytes, mode RGB),
   20×60 → (28, 84), 1×50 → (28, 1400), exact 28 unchanged, `requires_grad=False` cpu tensor.
2. `clean_vl_text` — the three-line golden, whitespace runs collapsed, empty/whitespace-only → "",
   punctuation (incl. `…`) untouched.
3. `read` with a fake model/processor — exact message list + `add_generation_prompt`/`tokenize`/
   `return_dict`/`return_tensors="pt"` + exact `images_kwargs` (shortest_edge from the processor,
   longest_edge 1280*28*28), `.to(device)` recorded, generate got `max_new_tokens=cfg.vl_max_new_tokens`,
   `do_sample=False`, `output_scores=True`, `return_dict_in_generate=True`, decode slice = the
   generated tail only (fake decodes 100, 101, 102), one generate per crop in order, text cleaned,
   hand-computed scores (`exp(-0.2)=0.8187`, `exp(-1.0)=0.3679`), clamped to 1.0, raising
   `compute_transition_scores` → 1.0, empty decode → `("", 0.0)`, `[] → []` with zero calls, a
   20×60 crop reaches the processor as (28, 84).
4. Loader (recorders on the real `Auto*` classes' `from_pretrained` — see Deviations 1) — installed
   → folder with `local_files_only=True` + `dtype=torch.float32`, `.to(device)` + `.eval()` called,
   `max_new_tokens` from `cfg.vl_max_new_tokens`; not installed → the pinned repo/revision + the
   warning naming `ocr-vl-1.6`; `models_dir=None` → hub, quiet; role `recognizer` →
   `ValueError("… has role 'recognizer', expected 'vlm_ocr'")`; `engine_rec_model` default
   `ocr-vl-1.6`, explicit `ocr-vl-1.5` wins; missing rec_model → `ValueError`.
5. Groups/stage — `paddleocr_vl` loads the comic detector + the VL reader and provably not
   `LineDetector.load`; the registered estimate is 6.6 GiB for this engine and 3.0 for `manga_ocr`/
   `ppocr`; the stage writes `ocr.json` with `engine == "ocr-vl-1.6"`, `drop_conf`/empty-text drops
   work, the stage re-run is skipped.
6. GPU smoke — `test_ocr_vl_gpu.py` renders two compact English blocks (word-wrapped to ≤ 18 chars)
   with the bundled `ComicNeue-Regular.ttf`, reads them with the real reader, asserts
   `0 < score <= 1` and ≥ 60 % alphanumeric-character recall (case-insensitive). **Skipped in this
   worktree** with a clear reason (`PaddlePaddle/PaddleOCR-VL-1.6 is not installed in
   V:\OmniScan-wt\O1d\models - run 'omniscan models download ocr-vl-1.6'`); the model is also not
   installed in the main checkout's models dir, so no live run was possible without the ~1.9 GB
   download (see Questions).
7. Whole non-GPU suite green.

## Deviations

1. **The loader tests patch `AutoModelForImageTextToText.from_pretrained` / `AutoProcessor.from_pretrained`
   methods instead of module attributes.** Patching the `transformers.Auto*` module attributes breaks
   silently: transformers 5's `_LazyModule` re-caches the real classes when a second attribute is
   resolved, so the first patch is undone (verified with a minimal repro). Method patching is also
   the established pattern in `test_ocr_model.py`.
2. **The unknown-engine group test now patches `cfg.ocr.engine` to a fake value.** `OcrConfig.engine`
   is a closed `Literal` and `paddleocr_vl` is valid now, so the old test construction is impossible;
   the defensive `ValueError(f"OCR engine {engine!r} is not available yet")` stays in `groups.py`
   (it also keeps `ocr` from being possibly-unbound for pyright) and the test covers it.
3. **`prepare_vl_image` calls `.convert("RGB")`** after `to_pil_image` (the card allows either
   `to_pil_image` or `Image.fromarray(...)`): this guarantees the RGB contract even if a caller ever
   passes a 1- or 4-channel crop.

## Questions

1. The GPU smoke test needs `ocr-vl-1.6` installed — `omniscan models download ocr-vl-1.6` (~1.9 GB;
   owner's call, same pattern as the O1b smoke test) — then
   `uv run pytest tests/unit/test_ocr_vl_gpu.py -m gpu -rs` on a checkout with the model. The
   director's spike numbers (3.6 s/region fp32 on the RX 9070 XT) say the reader itself is right;
   what is unverified live is only the smoke test's scoring/recall assertions and the exact
   `processor.image_processor.size["shortest_edge"]` key path on the real processor.
2. The 6.6 GiB vision-group estimate is the card's 3.0 + 3.6 constant, not measured peak VRAM with
   the detector resident. Should O1c (benchmarks) refine the estimate from a measurement?

## Commits

```
88ee2d1 O1d: WIP vl reader
<final> O1d: PaddleOCR-VL engine
```
# O1d — OCR engine `paddleocr_vl` (PaddleOCR-VL reads region crops)

**Owner:** GLM builder · **Branch:** `O1d` · **Worktree:** `V:\OmniScan-wt\O1d` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read `docs/tasks/O1b.md` (the design this card extends — **O1b is merged**: `ocr/engines.py` `model_source`/`load_kwargs`/`DEFAULT_REC_MODEL`/`engine_rec_model`, `ocr/crop_readers.py` `TextReader`/`MangaOcrReader`, `ocr/pipeline.py` `read_region_crops`, `ocr/stage.py`, `gpu/groups.py`) and its tests `tests/unit/test_ocr_crop_readers.py`, `test_ocr_engines.py`, `test_ocr_stage.py`, `test_gpu_groups.py`. `src/omniscan/core/config.py` (`OcrConfig.engine`, `rec_model`, `vl_max_new_tokens`; **director-owned, do not edit**).

## Why
Owner wish: the user can choose PaddleOCR-VL (a 0.9 B vision-language model that reads a whole region, any of the CJK scripts and Latin) as an OCR engine, next to PP-OCR and manga-ocr. It is the accuracy mode: slower (~3.6 s per region on the RX 9070 XT) but it read the Japanese Pepper&Carrot bubbles better than manga-ocr on stylised lettering (`ペッパーとキャロット` correct, where manga-ocr wrote `代ッパー…`). Director's spike (2026-09-20, transformers 5.17, fp32, cuda:1), which is a fixed part of the spec:
- `PaddlePaddle/PaddleOCR-VL-1.6` (catalog id `ocr-vl-1.6`; 1.5 and the first release load the same way) loads with `AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float32)` + `AutoProcessor.from_pretrained(path)`: **3.4 GiB fp32, loads in ~2.5 s**, no extra dependency. (`torch_dtype=` still works but is deprecated: use `dtype=`.)
- One crop per call: `messages = [{"role": "user", "content": [{"type": "image", "image": <PIL RGB>}, {"type": "text", "text": "OCR:"}]}]`; `inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", images_kwargs={"size": {"shortest_edge": processor.image_processor.size["shortest_edge"], "longest_edge": 1280 * 28 * 28}}).to(device)`; `out = model.generate(**inputs, max_new_tokens=…, do_sample=False)`; the answer is `processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)`. (`processor.image_processor.min_pixels` does **not** exist in transformers 5.17 — use `.size["shortest_edge"]`.) Crops with a side below 28 px must be upscaled first (the vision tower works on 28-px patches).
- It reads multi-line bubbles with `\n` between the lines (e.g. `きっと... / うっかりして / 寝ちゃったんだ!`); some crops come back with `…` where the page has `...` — both are normal.

## Files you may create / modify
- `src/omniscan/ocr/crop_readers.py` (modify: add `PaddleOcrVlReader` and its pure helpers), `src/omniscan/ocr/engines.py` (modify: `DEFAULT_REC_MODEL["paddleocr_vl"] = "ocr-vl-1.6"`), `src/omniscan/gpu/groups.py` (modify: load the reader for the engine), `src/omniscan/ocr/stage.py` (only if it needs the engine name for the label; keep it minimal)
- `tests/unit/test_ocr_vl_reader.py` (create), extend `tests/unit/test_ocr_engines.py`, `test_gpu_groups.py`, `test_ocr_stage.py`; `tests/unit/test_ocr_vl_gpu.py` (create; `gpu`-marked, skipped when the model is not installed)
- `docs/USER_GUIDE.md` (extend the "OCR engines and models" section: the engine, its speed and memory), `docs/reports/O1d.md`
Anything else is off-limits (especially `src/omniscan/core/**`, `config/models.toml`).

## Interfaces (exact)
```python
# ocr/crop_readers.py (additions)
VL_PROMPT = "OCR:"
VL_MIN_SIDE = 28
VL_MAX_PIXELS = 1280 * 28 * 28

def prepare_vl_image(crop: torch.Tensor) -> Image.Image: ...        # uint8 [3, h, w] (any device) -> PIL RGB, upscaled so that min(w, h) >= VL_MIN_SIDE
def clean_vl_text(text: str) -> str: ...

class PaddleOcrVlReader:
    def __init__(self, model: Any, processor: Any, device: torch.device, *, max_new_tokens: int) -> None: ...
    @classmethod
    def load(cls, cfg: OcrConfig, device: torch.device, models_dir: Path | None = None) -> PaddleOcrVlReader: ...
    def read(self, crops: Sequence[torch.Tensor]) -> list[tuple[str, float]]: ...
```
## Definitions (exact)
- **`prepare_vl_image`**: move the crop to the CPU (`crop.detach().cpu()`), `to_pil_image` (or `Image.fromarray(crop.permute(1, 2, 0).numpy())`), RGB. If `min(w, h) < VL_MIN_SIDE`: `scale = VL_MIN_SIDE / min(w, h)`, new size `(ceil(w * scale), ceil(h * scale))` with `Image.Resampling.LANCZOS`; otherwise unchanged. A 1-pixel-wide crop must not crash (scale 28).
- **`clean_vl_text(text)`**: `" ".join(text.split())` (all whitespace runs, including the `\n` between lines, become one space; leading/trailing whitespace removed). Nothing else is changed (no punctuation folding).
- **`PaddleOcrVlReader.load`**: `rec_model = engine_rec_model(cfg)` (`None` → `ValueError("paddleocr_vl needs ocr.rec_model")`); `model_source(rec_model, models_dir)`; `load_kwargs(...)`; `torch.cuda.set_device(device)` first when `device.type == "cuda"`; `AutoModelForImageTextToText.from_pretrained(path_or_repo, dtype=torch.float32, **kwargs)` → `.to(device).eval()`; `AutoProcessor.from_pretrained(path_or_repo, **kwargs)`; `max_new_tokens = cfg.vl_max_new_tokens`. A catalog entry of another role than `vlm_ocr` → `ValueError` naming the id and the expected role (as O1b does for the other readers).
- **`read(crops)`**: `[]` for no crops. **Sequential** — one `generate` per crop, in the given order (batched generation is measured/implemented in a later card; `cfg.crop_batch_size` is ignored here and the docstring says so). Per crop: `prepare_vl_image` → the message list above → `processor.apply_chat_template(..., images_kwargs={"size": {"shortest_edge": processor.image_processor.size["shortest_edge"], "longest_edge": VL_MAX_PIXELS}})` → `.to(self._device)` → `torch.inference_mode()` → `model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False, output_scores=True, return_dict_in_generate=True)`. Text = `clean_vl_text(processor.decode(out.sequences[0][n_prompt:], skip_special_tokens=True))` with `n_prompt = inputs["input_ids"].shape[-1]`. **Score** = `exp(mean(log-probabilities of the generated tokens))` from `model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)` (greedy search: no beam indices), clamped to `[0, 1]`, rounded to 4 decimals; when the computation raises → `1.0` with a `log.debug`; an empty text → `("", 0.0)`. No `.cpu()` on the tensors used for scoring except the final `float(...)`.
- **Group loader** (`gpu/groups.py`): `paddleocr_vl` → `{"reader": PaddleOcrVlReader.load(cfg.ocr, device, models_dir=cfg.paths.models_dir)}` next to the always-loaded comic detector (same shape as `manga_ocr`); the group's VRAM estimate becomes `3.0 + 3.6` GiB for this engine (`register(..., est_gib=…)`), unchanged for the others. The O1b `ValueError("OCR engine 'paddleocr_vl' is not available yet")` disappears.
- `OcrStage` needs no new code path (it already uses `models["reader"]` for every non-`ppocr` engine); the `engine` label in `ocr.json` is the catalog id (`ocr-vl-1.6`).

## Acceptance tests (CPU, fakes; no downloads)
1. **`prepare_vl_image`**: a 100×60 crop stays 100×60; 20×60 → min side 28 (`(28, 84)`); 1×50 → `(28, 1400)`; exact 28 unchanged; the pixels of an unscaled crop equal the tensor's; a CUDA-like tensor path is exercised by passing a tensor with `requires_grad=False` on cpu (no device assumptions); mode is `RGB`.
2. **`clean_vl_text`**: `"きっと...\nうっかりして\n寝ちゃった"` → `"きっと... うっかりして 寝ちゃった"`; multiple spaces/tabs/newlines collapse; empty and whitespace-only → `""`; punctuation untouched.
3. **`read` with a fake model/processor** (fake `apply_chat_template` recording the messages and `images_kwargs`, fake `generate` returning an object with `.sequences` and `.scores`, fake `decode`, fake `compute_transition_scores`): the message list is exactly `[{"role": "user", "content": [{"type": "image", "image": <PIL>}, {"type": "text", "text": "OCR:"}]}]`; `add_generation_prompt`, `tokenize`, `return_dict`, `return_tensors="pt"` as specified; `images_kwargs == {"size": {"shortest_edge": <processor value>, "longest_edge": 1280 * 28 * 28}}`; `generate` got `max_new_tokens == cfg.vl_max_new_tokens`, `do_sample=False`, `output_scores=True`, `return_dict_in_generate=True`; the decoded slice starts at the prompt length; crops are processed one by one in order (call log); text cleaned; score = the exp of the mean of the fake log-probabilities (hand-computed), clamped; `compute_transition_scores` raising → score `1.0`; empty decode → `("", 0.0)`; `[]` → `[]`; a crop with a side below 28 reaches the processor upscaled.
4. **Loader**: monkeypatched `AutoModelForImageTextToText`/`AutoProcessor` recorders: installed model → the installed folder with `local_files_only=True` and `dtype=torch.float32`; not installed → the repo + pinned revision and the warning naming `ocr-vl-1.6`; `rec_model` of role `recognizer` → `ValueError`; `engine_rec_model` for `paddleocr_vl` is `ocr-vl-1.6`, an explicit `rec_model="ocr-vl-1.5"` wins; `.to(device)` and `.eval()` called.
5. **Groups/stage**: `build_vram_manager` with `engine="paddleocr_vl"` loads the comic detector + the VL reader and not the PP-OCR models (recorders); the registered estimate is 6.6 GiB for this engine and 3.0 for the others; the stage writes `ocr.json` through the fake reader with `engine == "ocr-vl-1.6"`.
6. **GPU smoke** (`@pytest.mark.gpu`; skip with a clear reason when `ocr-vl-1.6` is not installed in `paths.models_dir` — never download): render two compact English text blocks with a bundled font (e.g. `ComicNeue-Regular.ttf`: `"I even brought my clothes, hat, and potions!"` wrapped to lines of ≤ 18 characters), read them with the real reader, assert `0 < score <= 1` and that ≥ 60 % of the alphanumeric characters of each truth appear (case-insensitive) in the text.
7. `uv run --frozen pytest -q -m "not gpu"` stays green.

## Out of scope
Batched generation, bf16/fp16, the GGUF/llama.cpp builds, spotting/table/formula prompts, choosing the engine automatically, benchmarks (card O1c).

## Commands to run before finishing
```bash
uv run --frozen pytest tests/unit/test_ocr_vl_reader.py tests/unit/test_ocr_engines.py tests/unit/test_ocr_stage.py tests/unit/test_gpu_groups.py tests/unit/test_docs.py -q
uv run --frozen pytest -q -m "not gpu"
uv run --frozen ruff format . && uv run --frozen ruff check .
uv run --frozen pyright
```

## Report
`docs/reports/O1d.md`: Changes, Tests (commands + results), Deviations, Questions. **Commit early** (`O1d: WIP vl reader`) once the reader and its tests pass; the final commit is `O1d: PaddleOCR-VL engine`. You have 150 tool calls in total. If anything is unclear: stop, write the question under Questions, commit, and end.

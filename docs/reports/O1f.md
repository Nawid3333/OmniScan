# O1f — Batch PaddleOCR-VL crop reading

## Changes

- `src/omniscan/ocr/crop_readers.py` — `PaddleOcrVlReader` only:
  - `__init__` takes `batch_size` (keyword, stored as `self.batch_size`) next to `max_new_tokens`,
    mirroring `MangaOcrReader`'s shape; `.load()` passes `batch_size=cfg.crop_batch_size` (the field
    existed in `core/config.py` and was previously ignored by this class — no `core/**` change).
  - `read()` chunks crops (`for start in range(0, len(crops), self.batch_size)`, like
    `MangaOcrReader.read()`) and makes **one batched `apply_chat_template` + one batched
    `generate()` call per chunk**. The per-conversation message dicts are unchanged; the chat
    template now receives a list of one-conversation-each plus `padding=True, padding_side="left"`.
  - New `_readings(out, n_prompt)` helper decodes/scores **per row of the batch**: row text from
    `out.sequences[index][n_prompt:]` (same decode + `clean_vl_text` as before), row score from a
    single per-chunk `compute_transition_scores(..., normalize_logits=True)` call with
    `generated != pad_id` masking of output-side pad tokens (the `MangaOcrReader._readings`
    pattern; `pad_id` from `model.config.pad_token_id`, falling back to 0 — the real model's
    `generation_config.json` sets `pad_token_id: 0`). Empty text still → `("", 0.0)`.
  - `_score` is gone (subsumed by `_readings`; no dead code). `n_prompt` is one shared value per
    chunk (`inputs["input_ids"].shape[-1]`) — valid only because batches are left-padded.
  - Module docstring updated (the "one generate per crop" sentence).
- `tests/unit/test_ocr_vl_reader.py` — fakes rebuilt for batches (hand-built left-padded
  `input_ids`/`attention_mask`, per-row scripted `gen`/`logs`), plus the card's acceptance tests;
  `load()` asserts the default `crop_batch_size=16` threads through and a new test pins a
  non-default `crop_batch_size=7`.
- `tests/unit/test_ocr_vl_gpu.py` — new `@pytest.mark.gpu` director helper (see below).
- No changes to `core/**`, `MangaOcrReader`, `ocr/model.py`, `engines.py`, `pipeline.py`,
  `stage.py`, or any `config/*.toml`.

## Which processor-API path (Part 2 step 2) — investigated, not assumed

Empirically probed the **real** `PaddleOCRVLProcessor` on CPU (tokenizer + configs from the already
installed `V:\OmniScan\models\ocr-vl-1.6`; no model weights loaded, nothing downloaded) with 3
conversations whose images have different aspect ratios (→ different image-token counts, prompt
lengths 165/169/167):

- **Direct batched path works**: `apply_chat_template(batch_messages, ..., padding=True)` accepts a
  list of one-conversation-each and returns equal-length `input_ids`/`attention_mask` plus batched
  `pixel_values` (1848, 3, 14, 14) and `image_grid_thw` (one row per image) and
  `mm_token_type_ids`. transformers 5.17.0's `ProcessorMixin.apply_chat_template` detects the
  batch (`is_batched`), collects per-conversation images into `batch_images`, and calls
  `self(text=prompt, images=batch_images, ...)` internally.
- **`padding=True` is mandatory for batched tensors**: without it the call raises
  `ValueError: Unable to create tensor, you should probably activate truncation and/or padding...`
  (ragged rows can't stack). So no unpadded-then-manual-pad fallback is needed or used.
- **Default padding side is `right`** — confirmed on the returned mask (`111...000`, padding at the
  end). The card's silent-garbage warning is exactly this; the code passes `padding_side="left"`
  explicitly on every call.
- **`padding_side` is a per-call kwarg here**: `padding_side="left"` forwarded through
  `apply_chat_template` reaches the tokenizer call and pads on the left (row masks
  `00...11`, prompts ending flush right at the shared index), **without mutating**
  `processor.tokenizer.padding_side` (verified it stayed `"right"` after the call). So the card's
  "set + restore `tokenizer.padding_side`" dance is unnecessary on this stack — no global state is
  touched. If a future transformers release drops the kwarg forwarding, the restore-style fallback
  would be needed (noted under Questions).

## Tests

- `uv run pytest tests/unit/test_ocr_vl_reader.py -q` → 25 passed.
- `uv run pytest -q -m "not gpu"` → see "Results" below (recorded after the run).
- `uv run ruff format . && uv run ruff check .` → clean.
- `uv run pyright` → no new errors.

New/extended coverage (all CPU, fake model+processor, no download):

1. batching happens: `crop_batch_size=2`, 5 crops → exactly 3 `generate()` calls with batch dims
   2+2+1 (`test_read_batches_five_crops_into_three_generate_calls`);
2. order preserved across chunk boundaries (same test, plus the messages test's 2+1 chunks);
3. per-row decoding independent: rows with different generated tails, one padded to the other's
   length with pad id 0, decode to their own texts (`test_read_decodes_each_rows_own_tail_ignoring_output_pads`);
4. per-row scores: different `logs` per row → 0.9512 vs 0.1353; output-side pads carry −9.0 log-probs
   that would wreck the mean if masking broke (`test_read_scores_are_per_row`);
5. left padding: the fake returns real prompt lengths 2 and 4 left-padded to 4; the test asserts the
   `padding_side="left"` kwarg, the left-padded `attention_mask`, and that row 0's decode slice
   starts after the **shared** width (a per-row-width slice would bleed prompt tokens
   `1000, 1001` into the decode) (`test_read_left_pads_the_prompt_and_offsets_the_tail_by_one_shared_length`);
6. `read([])` → `[]`, no `generate` call (`test_read_no_crops_never_calls_the_model`);
7. single crop goes through the batched path as one chunk of size 1
   (`test_read_single_crop_goes_through_the_batched_path`);
8. card messages/kwargs per conversation, `.to(device)`, generate args unchanged
   (`test_read_sends_the_card_messages_and_generate_arguments`); plus the pre-existing
   decode-tail/clean/clamp/raise/empty/upscale tests updated to the batched fakes.

## Real verification (director's job — not run here)

`tests/unit/test_ocr_vl_gpu.py::test_paddleocr_vl_batched_read_matches_and_times_chunk_size_one`
(a `@pytest.mark.gpu` test; the choice the card offered). Invocation from the repo root:

```bash
uv run pytest "tests/unit/test_ocr_vl_gpu.py::test_paddleocr_vl_batched_read_matches_and_times_chunk_size_one" -m gpu -s
```

(`-s` shows the printed wall-clock for both paths.) It loads the real `PaddleOcrVlReader`, renders
3 English crops of different aspect ratios (two speech-bubble texts + the very short `"Pepper!"`),
reads them batched (default `crop_batch_size=16` → one chunk), then re-reads the same crops through
a chunk-size-1 reader **sharing the same loaded model/processor** (the new code path, no old
sequential code kept), asserts per-crop text similarity ≥ 0.8 (greedy decoding may legitimately
shift a token under left padding) and scores in (0, 1], and prints both wall-clock times. It skips
itself when `ocr-vl-1.6` is not installed and never downloads.

## Deviations

- Test file: the card offered `tests/unit/test_ocr_crop_readers.py` (extend) or
  `tests/unit/test_paddleocr_vl_reader.py` (create); the reader was already covered by
  `tests/unit/test_ocr_vl_reader.py`, so the tests live there (the card's intent: "match whichever
  file already covers" the reader — for `PaddleOcrVlReader` that is `test_ocr_vl_reader.py`).
- Left padding via per-call `padding_side="left"` kwarg instead of mutating
  `processor.tokenizer.padding_side` and restoring — verified equivalent and mutation-free on the
  real processor (see above).
- The `padding=True`/`padding_side`/`images_kwargs` kwargs produce transformers' cosmetic warning
  "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict" — the pre-existing
  single-crop code triggered the same warning via `images_kwargs`; not a regression.

## Questions

- None blocking. Observations for the director (both out of scope here):
  - The model's `generation_config.json` has **`use_cache: false`** — every `generate()` re-runs the
    full prefill per token. Enabling the KV cache (if it works on this ROCm stack) is likely a much
    bigger speed lever than batching alone.
  - `pad_token` in the tokenizer is `<unk>` (id 0): a genuinely generated `<unk>` would be dropped
    from a row's score mean (same trade-off as `MangaOcrReader`'s pad fallback).
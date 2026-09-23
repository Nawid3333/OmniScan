# O1f — Batch PaddleOCR-VL's crop reading instead of one `generate()` per region

**Owner:** GLM builder · **Branch:** `O1f` · **Worktree:** `V:\OmniScan-wt\O1f` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read `src/omniscan/ocr/crop_readers.py` in full — both classes. `MangaOcrReader.read()`
(lines 107-122) **already does exactly this batching pattern** for a different model and is the structural template
to imitate: chunk `crops` into `self.batch_size`-sized groups, one batched `generate()` per chunk, one
`_readings(outputs)` helper that turns the batch output back into a `list[(text, score)]` in order. Also read
`src/omniscan/core/config.py`'s `OcrConfig.crop_batch_size` (default 16) and `vl_max_new_tokens` (both already
exist and are real config, `crop_batch_size` is just not read by `PaddleOcrVlReader` today) — read-only, no
`core/**` change needed for this card.

## Why (measured live tonight, real hardware, real chapter)
`PaddleOcrVlReader.read()` calls `self.model.generate()` **once per crop**, in a Python `for crop in crops:` loop —
its own docstring says so ("Sequential by design today (batched generation is a later card)"). Measured on
`PepperCarrotKR` Episode 06 (real GPU, real model): **818.7 seconds for one chapter.** A batch-size-1 autoregressive
`generate()` call badly underuses a 16 GB GPU; `MangaOcrReader` next to it in this same file already proves the
fix shape works for a different model. This card makes `PaddleOcrVlReader` do the same: batch `self._batch_size`
crops per `generate()` call, respecting the existing `cfg.crop_batch_size` config field that today does nothing
for this class.

**The correctness risk this card is really about:** batching a decoder-only VLM's `generate()` call is not just
"pass more tensors in" — mixing variable-length chat-template prompts (different crops produce different numbers
of image tokens depending on their aspect ratio after `prepare_vl_image`) in one batch needs **padding**, and
`generate()` needs that padding on the **left** for a causal LM (so every sequence's real content ends at the same
position and newly generated tokens line up across the batch) — get this wrong and it doesn't crash, it silently
produces garbage or truncated text for some crops while looking fine for others. This is exactly the kind of
GPU-only bug this project's own history warns about (a GPU codec race once survived ~2000 tests and only showed up
when someone compared real output pixels) — treat correctness here as the hard part, not the batching mechanics.

## Files you may create / modify
- `src/omniscan/ocr/crop_readers.py` (`PaddleOcrVlReader` only — do not touch `MangaOcrReader`,
  `decode_wordpieces`, `clean_manga_text`, or the module-level helpers below `MangaOcrReader` unless Part 2 needs a
  small addition to `prepare_vl_image`/`clean_vl_text`, which should stay behavior-identical for a single crop)
- `tests/unit/test_ocr_crop_readers.py` if it exists (extend) or `tests/unit/test_paddleocr_vl_reader.py` (create,
  matching whichever file already covers `MangaOcrReader` — check first and follow that file's naming/pattern)
- `docs/reports/O1f.md` (create)
- Do not modify `src/omniscan/core/**`, `MangaOcrReader`, `ocr/model.py`, `ocr/engines.py`, `ocr/pipeline.py`,
  `ocr/stage.py`, or `config/*.toml`. Do not download or load the real PaddleOCR-VL model (~3.4 GiB) in any test —
  everything here is CPU-only with a fake/stub model+processor. **The real-hardware correctness/speed check is the
  director's job afterward, not yours** — see "Real verification (not your job)" below.

## Part 1 — thread `crop_batch_size` into `PaddleOcrVlReader`
```python
class PaddleOcrVlReader:
    def __init__(
        self, model: Any, processor: Any, device: torch.device, *, max_new_tokens: int, batch_size: int
    ) -> None:
        ...
        self.batch_size = batch_size

    @classmethod
    def load(cls, cfg: OcrConfig, device: torch.device, models_dir: Path | None = None) -> PaddleOcrVlReader:
        ...  # unchanged except the final line:
        return cls(model, processor, device, max_new_tokens=cfg.vl_max_new_tokens, batch_size=cfg.crop_batch_size)
```
(Mirror `MangaOcrReader`'s constructor/`.load()` shape exactly — same keyword name `batch_size`, same attribute
name `self.batch_size`, for consistency between the two readers.)

## Part 2 — batch `read()`
Restructure `read()` to loop over chunks (`for start in range(0, len(crops), self.batch_size):`, exactly like
`MangaOcrReader.read()`), and inside each chunk:

1. Build one `messages` list per crop in the chunk (the existing per-crop dict, unchanged) →
   `batch_messages = [messages_for_crop_0, messages_for_crop_1, ...]` (a list of one-conversation-each, since
   `apply_chat_template` batches over the **outer** list dimension).
2. **Investigate before assuming an API shape** (this project's processor is `AutoProcessor` for the PaddleOCR-VL
   model specifically — its behavior needs checking, not guessing): try calling `self.processor.apply_chat_template`
   with the batched `batch_messages` list and the same kwargs `read()` already uses, plus `padding=True`. If that
   raises, doesn't accept a list of conversations, or doesn't return padded `input_ids`/`attention_mask` of equal
   length across the batch, fall back to: call `apply_chat_template(msgs, ..., tokenize=False)` per crop to get
   each prompt as **text** (with the image placeholder token already inserted), then a single batched call to
   `self.processor(text=list_of_prompt_strings, images=list_of_pil_images, padding=True, return_tensors="pt", ...)`
   to tokenize+pad the whole chunk together. Whichever path you take, verify empirically (a CPU-fake test can at
   least prove your code *calls* the right method with the right shape of arguments) and document which one this
   model's processor actually needs in the report, with the evidence (a traceback or the mismatched shapes you saw
   when you tried the direct approach, if you had to fall back).
3. **Left-padding is required.** Before tokenizing, set the tokenizer/processor's padding side to `"left"` for the
   duration of this batch (most HF processors expose this as `self.processor.tokenizer.padding_side = "left"`;
   restore whatever it was before, if anything else in this class's lifetime could depend on it — it likely
   doesn't, but don't leave a global mutation as a surprise for the next call if it's cheap to avoid). Confirm this
   is actually what the resulting `input_ids`/`attention_mask` show (padding tokens on the left, real content
   flush right) before trusting the rest — a wrong padding side is exactly the silent-garbage failure mode
   described above.
4. `n_prompt = inputs["input_ids"].shape[-1]` stays a **single shared value** for the whole batch (this only works
   correctly with left padding — every sequence's prompt, real content plus padding, ends at the same index, so
   generation for every sample in the batch starts at that same index in `out.sequences`).
5. One `self.model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False, output_scores=True,
   return_dict_in_generate=True)` call per chunk (unchanged arguments, just batched inputs now).
6. A new `_readings(self, out: Any, n_prompt: int) -> list[tuple[str, float]]` helper (mirror
   `MangaOcrReader._readings`'s per-row-of-the-batch loop structure) that, per row `index`:
   - Decodes `out.sequences[index][n_prompt:]` the same way `read()` does today (`self.processor.decode(...,
     skip_special_tokens=True)`, then `clean_vl_text`).
   - Computes that row's score the same way `_score` does today, but **per-row**, not just row 0: pull
     `out.sequences[index]`/`out.scores` sliced appropriately, mask out any generated PAD tokens (a shorter
     completion in the batch gets padded to the longest one's length — mirror `MangaOcrReader._readings`'s
     `generated != pad_id` masking pattern) before averaging the log-probs. Reuse `self.model.compute_transition_scores`
     (called once per chunk, not per crop) rather than the old single-sample call.
   - Empty text → `("", 0.0)`, same as today.
7. `read([])` still returns `[]` immediately (unchanged).

## Acceptance tests (CPU only; a fake/stub model+processor; no real PaddleOCR-VL download, no GPU)
Build a minimal fake `model`/`processor` pair (a small `SimpleNamespace`-based double, or a tiny hand-written stub
class) that: `apply_chat_template` (or the fallback `processor(...)` call, whichever path your implementation takes)
returns a `dict`-like batch with `input_ids`/`attention_mask` you construct by hand for 2-3 crops of deliberately
different "prompt lengths" (simulate this any way that's convenient — e.g. your fake doesn't need to actually run a
real chat template, it just needs to return shapes your code must then use correctly); `model.generate` returns a
scripted `out.sequences`/`out.scores` you control, so you can assert the exact text/score your code extracts per
row. Cover:
1. **Batching happens**: `crop_batch_size=2` (patch `cfg.crop_batch_size` or construct the reader directly with
   `batch_size=2`) and 5 crops → exactly 3 `generate()` calls (2+2+1), each call's fake asserts how many prompts it
   received.
2. **Order preserved**: the returned `list[(text, score)]` is in the same order as the input `crops`, across chunk
   boundaries.
3. **Per-row decoding is independent**: a batch of crops whose fake `out.sequences` differ per row (including one
   row that's shorter than another, i.e. real content followed by pad tokens on the *output* side) decodes each
   row to its own distinct text — not row 0's text repeated, not one row's tokens bleeding into another's.
4. **Score is per-row**: two rows with different fake `out.scores` produce two different score values (this would
   have silently passed as identical before batching if you copy row 0's score for every row by mistake — make the
   test's fake scores different enough that such a bug fails loudly).
5. **Padding side**: assert your code actually sets/uses left padding — e.g. by asserting the tokenizer's
   `padding_side` attribute on your fake, or by constructing the fake's returned `attention_mask` in a way that
   only makes sense read left-padded and asserting the decode slice math handles it (`n_prompt` shared across rows).
6. **Empty crops**: `read([])` → `[]`, no `generate` call.
7. **Single crop** (`len(crops) < batch_size`): still goes through the batched path (one chunk of size 1) and
   produces the same result shape as before — this is really a batch-size-1 case of test 1-4, but pin it
   explicitly since it's the case every existing caller currently exercises.
8. `uv run pytest -q -m "not gpu"`, `ruff format`/`ruff check`, `pyright` all clean.

## Real verification (not your job — the director does this after merge)
Write, but do not run, one clearly-marked helper the director can run by hand on real hardware: a short script or
a `@pytest.mark.gpu`-marked test (your choice, note which in the report) that loads the real `PaddleOcrVlReader`,
runs a handful of real crops (from an existing synthetic/real fixture already in the repo — check
`tests/fixtures/`) through both `read()` (batched) and, temporarily, a `batch_size=1`-forced call (still your new
code path, just chunk size 1 — do **not** keep the old sequential code around as a second method) and asserts the
text output is identical or near-identical between the two, plus prints the wall-clock time for each. Do not run
this yourself (it needs the real 3.4 GB model) — just make sure it exists, is easy to find, and is described in the
report so the director knows exactly how to invoke it.

## Out of scope
`MangaOcrReader` (already batches, untouched), changing `crop_batch_size`'s default value, batching *across*
different candidates/models (the qualification suite's own concern, unrelated), any change to `ocr/stage.py`'s
dispatch, right-padding fallback modes, mixed dtypes/quantization.

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_ocr_crop_readers.py -q   # or wherever the new/extended tests live — say which in the report
uv run pytest -q -m "not gpu"
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
`docs/reports/O1f.md`: Changes, Tests (commands + results), which processor-API path you ended up needing and why
(the investigation from Part 2 step 2), the real-hardware verification helper's exact invocation command for the
director, Deviations, Questions. **Commit early** (`O1f: WIP batch_size threaded + read() restructured`) once Part
1 and the basic chunking of Part 2 pass; the final commit is `O1f: batch PaddleOCR-VL crop reading`. You have 150
tool calls in total. If the processor's chat-template batching behavior is genuinely ambiguous or contradicts what
this card assumes: stop, write the question under Questions with what you tried and saw, commit, and end.

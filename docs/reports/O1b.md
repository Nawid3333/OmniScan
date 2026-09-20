# O1b — OCR engines: model ids for `ppocr`, and the `manga_ocr` crop reader

## Summary

`ocr.det_model` / `ocr.rec_model` now name catalog entries: a new `ocr/engines.py` maps a model id to
its installed folder (offline) or its pinned Hugging Face revision, and `LineDetector.load` /
`LineRecognizer.load` take that path while keeping today's exact behaviour when the ids are unset
(role checked, wrong role → `ValueError`). A second kind of engine reads **whole region crops**
instead of detecting lines first: `ocr/engines.py` + `ocr/crop_readers.py` implement `manga_ocr`
(`jzhang533/manga-ocr-base-2025`, a VisionEncoderDecoder read with its own `vocab.txt` — no
`fugashi`, no new dependency) behind the same `read(crops) -> [(text, score)]` protocol
`LineRecognizer` already satisfies. `ocr.engine` picks the path in the stage and the VRAM group;
`paddleocr_vl` raises "not available yet" (card O1d). Artifacts, drop rule, metrics keys and
downstream code are unchanged.

## Changes

- `src/omniscan/ocr/engines.py` (new): `ModelSource` (repo, revision, local, id — see Deviations 1),
  `model_entry` (id → catalog entry, unknown → `ValueError("unknown OCR model …")`), `model_source`
  (non-`hf`/`zip` or missing `upstream_repo` → `"{id} is not a Hugging Face model"`; `local` via
  `local_model_source`), `load_kwargs` (installed → `(folder, {"local_files_only": True})` + info
  log; else `(repo, {"revision": …})` +, when `models_dir` is given, the loaders' warning text now
  naming the catalog id and `omniscan models download <id>`), `DEFAULT_REC_MODEL`
  (`{"manga_ocr": "ocr-rec-manga-ocr-2025"}`) and `engine_rec_model` (`rec_model` or the default).
- `src/omniscan/ocr/crop_readers.py` (new): `TextReader` protocol, `decode_wordpieces` (specials and
  `<unusedN>` skipped, `##` continues, stops at `[SEP]`, out-of-range id → `IndexError`),
  `clean_manga_text` (whitespace squeezed, `…` → `...`, runs of ≥2 `.`/`・` collapsed to dots,
  printable ASCII widened except `.` and `~`), `MangaOcrReader` — `load` resolves
  `engine_rec_model`, loads `VisionEncoderDecoderModel` + `AutoImageProcessor` fp32, reads
  `vocab.txt` from the install folder or `hf_hub_download`; `read` chunks crops by
  `cfg.crop_batch_size` in input order, keeps crops on their device when the processor accepts them
  (falls back to one `to_pil_image` round trip per crop), generates with the model's own generation
  config inside `torch.inference_mode()`, decodes via the vocab, and scores `exp(mean log-prob)` of
  the non-padding generated positions through
  `model.compute_transition_scores(sequences, scores, beam_indices, normalize_logits=False)`
  (raises → 1.0 + `log.debug`; empty text → 0.0; clamped to [0, 1], rounded to 4 decimals).
- `src/omniscan/ocr/model.py`: both `load` classmethods branch on `cfg.det_model` / `cfg.rec_model`:
  set → role check (`_check_model_role`, `ValueError` naming the id and the expected role) +
  `model_source`/`load_kwargs`; unset → the previous code path verbatim (`det_repo`/`rec_repo`,
  revisions, `local_model_source`, `--required` warning).
- `src/omniscan/ocr/pipeline.py`: `read_region_crops(strip, regions, reader, cfg, *, engine)` —
  per region the padded/clamped crop (reuse `_crop_box`, a view of the strip), one `reader.read`
  call, `by_region` = one `LineBox` per region (the region's own bbox, score 1.0), readings keyed
  `(region.id, 0)`, assembled with `build_ocr_regions`; metrics `{"tiles": 0, "lines": len(regions),
  "orphan_lines": 0, "regions": len(regions), "regions_empty", "regions_low_conf"}` as in
  `read_regions`. `read_regions` untouched.
- `src/omniscan/ocr/stage.py`: `engine == "ppocr"` → today's `read_regions` call with the engine
  label = `rec_model` id when set, else the repo name; any other engine → `read_region_crops` with
  `models["reader"]` and the `engine_rec_model` id (None → `ValueError`). Drop rule, `ocr.json`
  writing, `config_subset` (already hashes all of `cfg.ocr`) unchanged.
- `src/omniscan/gpu/groups.py`: the vision group loads the comic detector always plus — `ppocr`:
  line detector + recognizer (as before), `manga_ocr`: `{"reader": MangaOcrReader.load(...)}`, else
  `ValueError("OCR engine '…' is not available yet")`.
- `tests/unit/test_ocr_engines.py` (new, 12), `tests/unit/test_ocr_crop_readers.py` (new, 10),
  `tests/unit/test_ocr_manga_gpu.py` (new, 1, `gpu`); `test_ocr_model.py` (+4), `test_ocr_pipeline.py`
  (+2), `test_ocr_stage.py` (+4), `test_gpu_groups.py` (+2).
- `docs/USER_GUIDE.md`: new "OCR engines and models" section after `omniscan ocr` (engine table,
  model selection, env/series.toml examples); validated by `test_docs.py`.

## Tests

```
uv run --frozen pytest tests/unit/test_ocr_engines.py tests/unit/test_ocr_crop_readers.py
             tests/unit/test_ocr_pipeline.py tests/unit/test_ocr_stage.py
             tests/unit/test_ocr_model.py tests/unit/test_gpu_groups.py tests/unit/test_docs.py -q
  → 75 passed
uv run --frozen pytest -m "not gpu"            → 3281 passed (34 tests added by this card, 1 more gpu-marked)
uv run --frozen ruff format . && uv run --frozen ruff check .
  → clean
uv run --frozen pyright                        → 0 errors
```

Acceptance tests, mapped to the card:

1. `model_source` — `test_ocr_engines.py`: installed folder / missing / `models_dir=None` / unknown
   id / `ollama` entry, with a hand-made catalog and a real `.installed.json` install.
2. `load_kwargs` — installed (info log) / hub with revision (warning naming the id and the
   `omniscan models download <id>` command) / no revision `{}` / `models_dir=None` never warns.
3. Model-id loads — `test_ocr_model.py::test_line_detector_load_by_model_id_*` (installed folder +
   `local_files_only`; pinned hub repo from the catalog, warning naming the id), wrong role →
   `ValueError`, ids unset → exactly the pre-O1b arguments.
4. `decode_wordpieces` (goldens, `##` first, `<unusedN>`/specials, `[SEP]` stop, `IndexError`) and
   `clean_manga_text` (5 goldens, empty, whitespace-only, mixed width, `.`/`~` kept, tab/newline).
5. `MangaOcrReader.read` with a fake model/processor — order across chunks (batch_size 2, 5 crops,
   hand-computed `exp(mean log-prob)` scores incl. pad masking), `[]` no crops, empty text →
   `("", 0.0)`, `compute_transition_scores` raising → 1.0, 1×1 crop, PIL fallback when the
   processor refuses tensors, missing `ocr.rec_model` → `ValueError`.
6. `read_region_crops` — fake reader records crops: shapes equal the padded/clamped boxes,
   `data_ptr()` inside the strip's storage (views), corner regions clamped, readings mapped back in
   input order, empty text drops the line, `OcrLine.engine`/`bbox`, metrics incl. `regions_low_conf`.
7. Stage — `engine="manga_ocr"` with a fake reader writes `ocr.json` with the fake text and
   `engine == "ocr-rec-manga-ocr-2025"`, `drop_conf` and empty-text drops, re-run skipped, `rec_model`
   change re-runs and the label follows the id; all pre-existing ppocr stage tests pass unchanged.
8. VRAM group — `ppocr` unchanged; `manga_ocr` loads comic detector + reader and provably not
   `LineDetector.load`; `paddleocr_vl` → `ValueError`.
9. GPU smoke — `test_ocr_manga_gpu.py`: **skipped in this worktree** with a clear reason
   (`jzhang533/manga-ocr-base-2025 is not installed in <worktree>/models - run 'omniscan models
   download ocr-rec-manga-ocr-2025'`); it renders Japanese with a `fonts/` font chosen by
   `font.getmask("あ")`, reads with the real `MangaOcrReader` on `resolve_device()` and asserts ≥
   half of the rendered characters come back. Needs a live run on a machine with the model
   installed (see Questions).
10. Whole non-GPU suite green.

## Deviations

1. **`ModelSource` has a fourth field `id`.** The card fixes three (`repo`, `revision`, `local`)
   but also requires the not-installed warning to "name the catalog id" (acceptance test 2) — which
   `load_kwargs(source, models_dir=...)` cannot do without the id, since the catalog is not a
   parameter. The id is set by `model_source` and used only for the warning; `repo`/`revision`/
   `local` keep their meaning and order. (`engine_rec_model` was also needed by `stage.py`, which
   the card's interface block already lists.)
2. **`clean_manga_text` widens the backtick.** The card's punctuation list omits `` ` `` (impossible
   to type inside a code span) between `_` and `{`; it is part of printable ASCII 0x21–0x7E and is
   widened with the rest. `.` and `~` stay half-width exactly as the card says. No golden involves
   the backtick.
3. **`MangaOcrReader.load` raises on `rec_model is None`** rather than literally on
   `cfg.engine != "manga_ocr" and cfg.rec_model is None` — equivalent for every reachable config
   (with `engine="manga_ocr"` the default id always resolves) and simpler; the message is the
   card's `"manga_ocr needs ocr.rec_model"`.
4. **Stage guard**: with a non-ppocr engine and no resolvable rec model id the stage raises a clear
   `ValueError` instead of passing `engine=None` into `OcrLine.engine` (unreachable through
   `groups.py`, defensive only).
5. The GPU smoke test could not run live here: the worktree's models dir has no manga-ocr install
   (the test therefore skips, as the card specifies — it never downloads).

## Questions

1. The GPU smoke test (acceptance 9) needs a machine with `ocr-rec-manga-ocr-2025` installed
   (`omniscan models download ocr-rec-manga-ocr-2025`) — please run it once on the main checkout
   (`uv run pytest tests/unit/test_ocr_manga_gpu.py -m gpu -rs`); the fp32/beam-search scoring path
   (`compute_transition_scores` with beam indices) is only covered by fakes here.
2. `read_region_crops` reports `"lines": len(regions)` (one pseudo-line per region) per the card —
   the debugger's line metrics will read differently under crop engines. Confirm that is intended
   for O1d too.

## Commits

```
c42955a O1b: WIP engines + crop reader
<final> O1b: OCR model ids and the manga_ocr engine
```
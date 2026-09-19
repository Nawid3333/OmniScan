# C4a — OCR stage: PP-OCRv5 line detection + Korean recognition → `ocr.json`

## Changes

New `src/omniscan/ocr/` package (five modules, `__init__.py` empty):

- **`ocr/lines.py`** — pure line logic, no torch. `LineBox` (frozen dataclass, strip-pixel box + score);
  `polygon_box(points)` (bounds of a detection polygon); `merge_lines(...)` (NMS by descending score
  over `nms_iou`, then a containment rule that drops a survivor almost fully inside a strictly larger
  one at/above `contain_thr` — this is what rescues lines cut by a tile edge; result sorted by
  `(y0, x0)`); `assign_lines(...)` (per line, best region by `(-ioa_share, region_area, region.id)`
  against `region_pad_px`-padded boxes with the `assign_min_ioa` gate; returns
  `by_region` — one entry per region id, lines in `reading_order(...)` — and orphans in input order).
- **`ocr/assemble.py`** — `build_ocr_regions(...)`: lookup `readings[(region.id, index)]`, `.strip()`
  the text (empty after strip → line dropped), `OcrLine(bbox=..., text, score, engine)` with the box
  rounded outwards (floor x0/y0, ceil x1/y1) and clamped to the strip; per region
  `model_copy(update={lines, text (newline-joined), confidence = min surviving line score (0.0 if
  none), lang, orientation="h", ocr_alt=None})`. Inputs never mutated.
- **`ocr/model.py`** — thin wrappers. `LineDetector.load(cfg, device)`: HF
  `AutoModelForObjectDetection` + `AutoImageProcessor` from `ocr.det_repo`/`det_revision`;
  `torch.cuda.set_device(device)` before load (MIOpen launches on the *current* device);
  `_to_device(model, device).eval()`, fp32. `detect(tiles)`: per tile, processor on the uint8 GPU
  tensor (no copy), forward under `inference_mode`, `post_process_object_detection` with
  `det_threshold/box_threshold/unclip_ratio/min_size` and the tile's own size; polygons →
  `polygon_box`, clipped to the tile, sub-`min_size` boxes dropped. `LineRecognizer.load(cfg, device)`
  with `AutoModelForTextRecognition` from `rec_repo`/`rec_revision`; `read(crops)`: crops sorted by
  width, forwarded in `rec_batch_size` chunks under `inference_mode`, results restored to input order
  (`(text, score)` tuples).
- **`ocr/pipeline.py`** — `read_regions(strip, regions, detector, recognizer, cfg, *, direction,
  engine)`: tiles = `keep_tiles(plan_tiles(...))` restricted to tiles overlapping some region's
  `(y0, y1)`; detector called in batches of 4 tile views (never copies); boxes shifted to strip
  pixels; `merge_lines`; `assign_lines`; per-region-per-line padded crops (`_crop_box`: floor/ceil,
  clamped, at least 1×1) read by the recognizer in one `read` call; `build_ocr_regions`; metrics
  `{tiles, lines, orphan_lines, regions, regions_empty, regions_low_conf}` (all float).
- **`ocr/stage.py`** — `OcrStage` (`name="ocr"`, `version=1`, `gpu_group=VISION_GROUP`); inputs =
  ingest.json + slices.json + regions.json + raw images; outputs `ocr.json`; config subset =
  `cfg.ocr.model_dump()` + `detect.reading_direction`; run guards (`regions.json missing — run the
  detect stage first`, then ingest.json), loads the strip via `load_strip`, runs `read_regions`,
  saves `RegionsArtifact` to `ocr.json`, returns the metrics.

Modified:

- **`gpu/groups.py`** — the vision group's loader also builds `line_detector` +
  `recognizer` from `cfg.ocr` (imports deferred; module import still does not pull transformers);
  `est_gib` 2.0 → 3.0.
- **`cli.py`** — `ocr` removed from `_STUB_COMMANDS`; real `cmd_ocr` (same surface as `detect`:
  `series`, repeatable `--chapter/-c`, `--force`), docstring "Read the text of detected regions with
  PP-OCRv5 (ocr.json). Runs ingest, slice and detect first if needed.", runs
  Ingest → Slice → Detect → Ocr through `_run_stages`.

Tests: `test_ocr_lines.py`, `test_ocr_assemble.py`, `test_ocr_model.py`, `test_ocr_pipeline.py`,
`test_ocr_stage.py` (new); `test_gpu_groups.py` (vision group registers three models, `est_gib ==
3.0`, CLI ocr runs the pipeline through the manager, only region-touching tiles reach the detector,
ocr no longer a stub); `test_cli.py` (ocr out of `STUB_COMMANDS`, stub test now uses `judge`);
`test_docs.py` (one line: the stub-flag helper test now documents `omniscan judge` — see Deviations);
README + USER_GUIDE per the card.

## Tests

All commands from `V:\OmniScan-wt\C4a`:

- `uv run pytest tests/unit/test_ocr_lines.py tests/unit/test_ocr_assemble.py
  tests/unit/test_ocr_model.py tests/unit/test_ocr_pipeline.py tests/unit/test_ocr_stage.py
  tests/unit/test_gpu_groups.py tests/unit/test_cli.py tests/unit/test_docs.py -m "not gpu"` →
  **49 passed, 2 deselected** (all 16 card tests that are CPU-runnable; the 2 GPU tests deselected here).
- `uv run pytest -q` (full suite, GPU included) → **exit 0, 2249 passed** (0 failures).
- `uv run pytest tests/unit/test_ocr_pipeline.py -q -s -m gpu` (to capture the printed numbers) →
  both GPU tests pass.
- `uv run ruff format . && uv run ruff check .` → clean.
- `uv run pyright` → **0 errors, 0 warnings**.

### Test 15 measured (RX 9070 XT, fp32, `make_korean_page(seed=1, n_bubbles=4, n_free=1)`)

`test_real_ocr_reads_a_korean_page` (asserted run, NanumGothic):

- non-empty regions **5/5** (≥ 90% required), orphan_lines 0
- mean CER (whitespace removed) **0.032** (≤ 0.15 required)
- wall time **14.21 s** — that call includes the first CUDA inferences of the session (MIOpen/kernel
  warmup); the very next `read_regions` call with the same loaded models (Gaegu run) took **0.27 s**,
  so steady-state is sub-second for a 5-region page
- metrics: tiles 2, lines 6, regions 5, regions_empty 0, regions_low_conf 0

`test_real_ocr_robustness`: blank page + no regions → `[]`, no crash; clean page with truth regions
→ every region empty (`regions_empty == regions`) since there is no text drawn.

Printed only (never asserted), Gaegu handwriting font: non-empty 5/5, mean CER **0.313**, 0.27 s,
18 lines, regions_low_conf 4. Expected: PP-OCRv5 server-det finds the lines fine, but the mobile
Korean recognizer trained on print text reads the handwritten font poorly — the low-conf counter is
doing its job.

## Deviations

- **`tests/unit/test_docs.py` (1 line) — not in the card's file list.** Removing `ocr` from
  `_STUB_COMMANDS` (a card requirement) made `test_extraction_helper_flags_stub_as_documented` fail:
  it documents `omniscan ocr DemoSeries` as its example stub command. Since `ocr` is no longer a
  stub, the example became `omniscan judge DemoSeries` (the same still-stub command `test_cli.py`
  uses). No doc text changed.
- The card's test 16 wording ("empty image and no regions → no crash; *real* regions on a clean
  image → nothing detected") is implemented as: blank synthetic page with `[]` regions → `[]`, and a
  clean (textless) page with the truth regions passed in → every region empty
  (`regions_empty == regions`), which is the same claim (nothing to detect → no text read).

## Questions

None.
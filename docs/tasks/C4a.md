# C4a — OCR stage: PP-OCRv5 line detection + Korean recognition → `ocr.json`

**Owner:** GLM builder · **Branch:** `C4a` · **Worktree:** `V:\OmniScan-wt\C4a` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md` (Detection and OCR rows) and `docs/benchmarks/ocr-probe.md` (the
measurements behind every choice below) first. Then read the merged detection code: `src/omniscan/detect/tiles.py`,
`src/omniscan/detect/postprocess.py` (`ioa`, `area`, `intersection`, `reading_order`), `src/omniscan/detect/model.py`,
`src/omniscan/detect/stage.py`, `src/omniscan/gpu/groups.py`, `OcrConfig` in `src/omniscan/core/config.py` (already present),
`Region` / `OcrLine` / `RegionsArtifact` in `src/omniscan/core/schemas.py`, and `tests/fixtures/korean_pages.py` (synthetic Korean
pages with ground truth).

## Goal
`omniscan ocr SERIES` reads `regions.json` (regions found by detection, `text` empty) and writes `ocr.json` (same regions, `lines`,
`text`, `confidence` filled). Per chapter: the strip is cut into overlapping tiles (only tiles that touch a region), the PP-OCRv5
**text-line detector** runs on each tile on the GPU, its boxes are moved to strip pixels, de-duplicated across tiles and assigned to
the detected regions; every assigned line is cropped from the strip on the GPU and read by the **Korean recognition model**; the
results are assembled into regions. The pure logic (merging, assignment, ordering, assembling) is separate from the model wrappers so
it is unit-tested on CPU with fakes; the wrappers are tested with fake models; one `@pytest.mark.gpu` test runs the real models on a
synthetic Korean page and checks accuracy.

## Files you may create / modify
- `src/omniscan/ocr/__init__.py` (create, empty), `ocr/lines.py`, `ocr/assemble.py`, `ocr/model.py`, `ocr/pipeline.py`, `ocr/stage.py` (create)
- `src/omniscan/gpu/groups.py` (modify — ONLY extend the `vision` loader, see Part 5)
- `src/omniscan/cli.py` (modify — ONLY replace the `ocr` stub with the real command and remove `"ocr"` from `_STUB_COMMANDS`)
- `tests/unit/test_ocr_lines.py`, `test_ocr_assemble.py`, `test_ocr_model.py`, `test_ocr_pipeline.py`, `test_ocr_stage.py` (create);
  `tests/unit/test_gpu_groups.py`, `tests/unit/test_cli.py` (modify — only what the changes require)
- `README.md` (status row `ocr` → `working — omniscan ocr (ocr.json)`; drop `ocr` from the "Not implemented yet" table) and
  `docs/USER_GUIDE.md` (a `### omniscan ocr` subsection in the style of the others, the `ocr.json` row of the folder table, every `[ocr]`
  config key in the config table, and fix the "no OCR stage exists yet" sentences in the OCR view and `translate` sections) —
  `tests/unit/test_docs.py` must be green
- `docs/reports/C4a.md` (create)
Do not modify `src/omniscan/core/**` (`OcrConfig` is finished), `pyproject.toml` (`opencv-python-headless` is already a dependency), the
`detect/` package (import from it), or the web code.

## Facts you may rely on (measured, see the probe report)
- `PaddlePaddle/PP-OCRv5_server_det_safetensors` loads with `AutoModelForObjectDetection`, its `AutoImageProcessor` accepts a **uint8
  `[3,h,w]` tensor on the GPU** and keeps it there (`pixel_values` `[1,3,H',W']` with H', W' multiples of 32, long side ≤ 960; it also returns
  `target_sizes`). `processor.post_process_object_detection(outputs, target_sizes=…)` returns per image `{"boxes": [N,4,2] corner points in
  the ORIGINAL tile pixels, "scores": [N], "labels": [N]}`; it uses OpenCV on the CPU (3.4 ms per page — measured 12× faster than a GPU
  component-labelling attempt) and accepts any object with a `last_hidden_state` attribute.
- **The detector must run in fp32**: fp16 and bf16 both fail in MIOpen (`invalid device function`) on this GPU stack. Do not call `.half()`.
- `PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors` loads with `AutoModelForTextRecognition`; its processor accepts a **list of uint8
  `[3,h,w]` GPU tensors** (resizes each to height 48, pads to the widest of the batch) and `processor.post_process_text_recognition(outputs)`
  returns one `{"text": str, "score": float}` per crop. It does not read "..." / spaces reliably; the accuracy target below ignores spaces.
- Accuracy measured on a synthetic Korean page: character error rate 0.048 on the detector's boxes (spaces ignored).

## Part 1 — `ocr/lines.py` (pure, no torch)
```python
Box = tuple[float, float, float, float]

@dataclass(frozen=True, slots=True)
class LineBox:
    box: Box            # strip pixels
    score: float

def polygon_box(points: Sequence[Sequence[float]]) -> Box
def merge_lines(lines: Sequence[LineBox], *, nms_iou: float, contain_thr: float = 0.8) -> list[LineBox]
def assign_lines(regions: Sequence[Region], lines: Sequence[LineBox], *, min_ioa: float, pad_px: int,
                 direction: Literal["ltr", "rtl"]) -> tuple[dict[str, list[LineBox]], list[LineBox]]
```
- `polygon_box(points)` = `(min x, min y, max x, max y)` of the points.
- `merge_lines`: sort by descending `score` (stable); keep a line when its IoU with every already-kept line is `<= nms_iou`; afterwards remove every kept
  line that has `ioa(line, other) >= contain_thr` for some other kept line with a **strictly larger area** (a line cut by a tile edge loses against the whole
  line seen in the neighbouring tile). Return the survivors sorted by `(y0, x0)`.
- `assign_lines(regions, lines, ...)`: for every line find the regions whose text box, grown by `pad_px` on every side (`x0-pad, y0-pad, x1+pad, y1+pad`), contains at least
  `min_ioa` of the line (`ioa(line.box, padded)`); the line belongs to the one with the highest ioa (ties: smaller region area, then the smaller region `id` as a string); a
  line matching no region is an **orphan**. Returns `(by_region, orphans)`: `by_region` has an entry for **every** region id (possibly `[]`); the lines inside a region are ordered
  with `reading_order(boxes, direction)` from `detect.postprocess`; `orphans` keep the input order.

## Part 2 — `ocr/assemble.py` (pure)
```python
def build_ocr_regions(regions: Sequence[Region], by_region: Mapping[str, Sequence[LineBox]],
                      readings: Mapping[tuple[str, int], tuple[str, float]], *, engine: str, lang: Lang,
                      strip_width: int, strip_height: int) -> list[Region]
```
`readings[(region_id, line_index)]` = `(text, score)` for the line at that position of `by_region[region_id]`. Returns new `Region` objects (input untouched), same order as `regions`:
all existing fields are kept (`id`, `slice_index`, `kind`, `bbox`, `bubble_bbox`, `polygon`, `reading_order`, colours, `mask_ref`) except:
- `lines` = one `OcrLine(bbox, text, score, engine)` per line that has a non-empty stripped text, in line order; `bbox` = the line box rounded outwards (`floor` for x0/y0,
  `ceil` for x1/y1) and clamped to the strip; `polygon=None`. A line whose text is empty after `.strip()` is dropped.
- `text` = `"\n".join(line.text for line in lines)`; `confidence` = the **minimum** line score rounded to 4 decimals, `0.0` when there are no lines; `lang` = `lang`; `orientation` = `"h"`; `ocr_alt = None`.

## Part 3 — `ocr/model.py`
```python
class LineDetector:
    def __init__(
        self,
        model: Any,
        processor: Any,
        device: torch.device,
        *,
        det_threshold: float,
        box_threshold: float,
        unclip_ratio: float,
        min_size: int,
    ) -> None: ...
    @classmethod
    def load(cls, cfg: OcrConfig, device: torch.device) -> LineDetector: ...
    def detect(self, tiles: Sequence[torch.Tensor]) -> list[list[LineBox]]: ...


class LineRecognizer:
    def __init__(self, model: Any, processor: Any, device: torch.device, *, batch_size: int) -> None: ...
    @classmethod
    def load(cls, cfg: OcrConfig, device: torch.device) -> LineRecognizer: ...
    def read(self, crops: Sequence[torch.Tensor]) -> list[tuple[str, float]]: ...
```
- `LineDetector.load`: `AutoModelForObjectDetection.from_pretrained(cfg.det_repo, revision=cfg.det_revision).to(device).eval()` (fp32!), `AutoImageProcessor.from_pretrained(...)`.
- `detect(tiles)`: tiles are uint8 `[3,h,w]`; for each tile (they may differ in size, so one call per tile): `inputs = processor(images=tile.to(device), return_tensors="pt")`;
  `with torch.inference_mode(): out = model(pixel_values=inputs["pixel_values"])`; then
  `processor.post_process_object_detection(out, threshold=det_threshold, box_threshold=box_threshold, unclip_ratio=unclip_ratio, min_size=min_size, target_sizes=torch.tensor([[h, w]]))`
  and turn each `[4,2]` point set into a `LineBox(polygon_box(points), score)` with the box clipped to `[0,w]×[0,h]`. Result: one list per tile, in input order; empty input → `[]`.
  Boxes narrower or lower than 3 px after clipping are dropped.
- `LineRecognizer.read(crops)`: empty → `[]` without calling anything. Otherwise sort the crop indices by crop width (ascending), process them in chunks of `batch_size`:
  `inputs = processor(images=chunk, return_tensors="pt")`, `with torch.inference_mode(): out = model(**inputs)`, `processor.post_process_text_recognition(out)`; return
  `[(text, float(score))]` in the ORIGINAL crop order. The text is returned as given (no strip).
- Neither wrapper ever calls `.cpu()`/`.numpy()` on the strip or on crops.

## Part 4 — `ocr/pipeline.py` and `ocr/stage.py`
```python
def read_regions(strip: torch.Tensor, regions: Sequence[Region], detector: LineDetector, recognizer: LineRecognizer,
                 cfg: OcrConfig, *, direction: Literal["ltr", "rtl"], engine: str) -> tuple[list[Region], dict[str, float]]

class OcrStage:
    name: ClassVar[str] = "ocr"
    version: ClassVar[int] = 1
    gpu_group: ClassVar[str | None] = VISION_GROUP
```
- `read_regions(strip [3,H,W], regions, ...)`:
  1. No regions → `([], metrics)` without any model call.
  2. `tiles = keep_tiles(plan_tiles(W, H, cfg.tile_px, cfg.overlap), [(r.bbox.y0, r.bbox.y1) for r in regions])`.
  3. For every tile (in order): `raw = detector.detect([strip[:, t.y0:t.y1, t.x0:t.x1] for t in batch])` (batches of 4 tiles) and shift every line box by `(t.x0, t.y0)`.
  4. `lines = merge_lines(all, nms_iou=cfg.nms_iou)`; `by_region, orphans = assign_lines(regions, lines, min_ioa=cfg.assign_min_ioa, pad_px=cfg.region_pad_px, direction=direction)`.
  5. Crops: for every assigned line in region order then line order, the padded box (`line_pad_px` on each side, floor/ceil, clamped to the strip, at least 1×1) cut from the strip
     (`strip[:, y0:y1, x0:x1]` view); `readings = recognizer.read(crops)` mapped back to `(region_id, line_index)`.
  6. `build_ocr_regions(..., engine=engine, lang=cfg.lang, strip_width=W, strip_height=H)`.
  7. Metrics (all `float`): `tiles`, `lines`, `orphan_lines`, `regions`, `regions_empty` (no line survived), `regions_low_conf` (non-empty with `confidence < cfg.low_conf`).
- `OcrStage`: `inputs(ctx)` = `[ingest.json, slices.json, regions.json, *list_images(raw_dir)]`; `outputs` = `["ocr.json"]`; `config_subset(cfg)` = `{**cfg.ocr.model_dump(), "reading_direction": cfg.detect.reading_direction}`.
  `run(ctx, models)`: `models["line_detector"]`, `models["recognizer"]`; missing `regions.json` → `FileNotFoundError("regions.json missing — run the detect stage first")`; load ingest, strip (`load_strip`),
  `RegionsArtifact.load(regions.json)`; `read_regions(..., engine=cfg.ocr.rec_repo.split("/")[-1])`; save `RegionsArtifact(regions=...)` to `ocr.json`; return the metrics.

## Part 5 — VRAM group and CLI
- `gpu/groups.py`: the `vision` loader returns `{"detector": Detector.load(cfg.detect, device), "line_detector": LineDetector.load(cfg.ocr, device), "recognizer": LineRecognizer.load(cfg.ocr, device)}`
  (imports stay inside the loader) and `est_gib` becomes `3.0`.
- `omniscan ocr SERIES [--chapter/-c CHAPTER]... [--force]`: same options and help text style as `detect`; runs `[IngestStage(), SliceStage(), DetectStage(), OcrStage()]` through `_run_stages`.

## Acceptance tests (CPU unless marked `gpu`; fakes only — no downloads)
**Lines:**
1. `polygon_box([[10,5],[110,7],[108,25],[9,22]]) == (9, 5, 110, 25)`.
2. `merge_lines` (with `nms_iou=0.5`): two boxes of the same line from overlapping tiles `(100,100,400,130)@0.9` and `(102,101,401,131)@0.8` → only the first; a truncated `(100,100,250,130)@0.95` next to the whole
   `(100,100,400,130)@0.7` → only the whole one (IoU 0.5 is kept by NMS but the contain rule removes the smaller); two different lines one above the other are both kept; output sorted by `(y0, x0)`.
3. `assign_lines` with region `r0001` text box `(100,100,400,200)` (padded by `pad_px=8` to `(92,92,408,208)`), `min_ioa=0.5`:
   - line `(120,110,380,140)` → assigned to `r0001` (ioa 1.0);
   - line `(120,190,380,240)` → **orphan**: only y 190..208 of its 50 rows lie inside the padded box, ioa = 18/50 = 0.36 < 0.5;
   - with a second region `r0002` text box `(110,140,390,190)` (padded `(102,132,398,198)`), line `(120,150,380,180)` lies wholly inside both (ioa 1.0 each) → it goes to `r0002`, the region with the smaller area;
   - a line far from every region → orphan; a region without lines still has an entry `[]` in `by_region`;
   - two lines in one region come back ordered top to bottom; a same-row pair `(100,100,190,130)` / `(210,102,300,132)` is ordered left-to-right for `"ltr"` and right-to-left for `"rtl"`; `orphans` keep the input order.
**Assemble:**
4. Region with two lines and readings `("안녕", 0.99)`, `("하세요", 0.90)` → `text == "안녕\n하세요"`, `confidence == 0.9`, `lines[i].engine`, boxes rounded outwards (`(9.2, 4.6, 110.1, 24.9)` → `BBox(9,4,111,25)`), clamped to the strip, `lang`, `orientation == "h"`; every other field equal to the input's; the input regions are not mutated.
5. A line whose reading is `"  "` is dropped; a region with no usable line → `text == ""`, `lines == []`, `confidence == 0.0`; output order equals input order.
**Model wrappers (fake model + fake processor):**
6. `LineDetector.detect`: the fake processor records the tensor it receives (uint8 `[3,h,w]` on `device`) and returns `target_sizes`; the fake `post_process_object_detection` returns two `[4,2]` polygons per tile; the result is `LineBox`es in tile pixels clipped to the tile, one list per tile in order;
   a polygon that clips to under 3 px is dropped; `detect([])` returns `[]`; the detection kwargs (`threshold`, `box_threshold`, `unclip_ratio`, `min_size`, `target_sizes == [[h, w]]`) are passed through; the model is called inside `inference_mode`.
7. `LineDetector.load` (monkeypatched `from_pretrained`, CPU): `.eval()`, dtype stays fp32, `revision` passed through.
8. `LineRecognizer.read`: 5 crops of widths `[50, 200, 20, 120, 80]` with `batch_size=2` → 3 processor calls whose widths are non-decreasing across chunks (sorted by width), results returned in the ORIGINAL order (the fake maps a crop's width to a text so you can check); empty → `[]` and no call; scores are `float`.
**Pipeline (fake wrappers, CPU strip tensor):**
9. Tiles that touch no region are never sent to the detector; a chapter with 3 regions in 2 distant places sends only the tiles overlapping them (assert the crops the fake receives).
10. Line boxes returned in tile pixels are shifted to strip pixels (second tile `y0` added); the same line seen by two overlapping tiles yields ONE reading request.
11. End-to-end with fakes: 2 regions, 3 lines → `ocr` regions with the texts of the fake recognizer, `confidence` = min score, metrics `lines == 3`, `orphan_lines`, `regions_empty`, `regions_low_conf` computed as defined; no regions → `[]` and no model calls; crops handed to the recognizer are the padded, clamped strip views (assert shapes).
**Stage (real `load_strip`, CPU config as in `tests/unit/test_stages_ingest_slice.py`, fakes in `models`):**
12. Writes `ocr.json` (loadable `RegionsArtifact`), manifest `ocr` `done`, second run `skipped`; changing `ocr.rec_repo` or `detect.reading_direction` re-runs it; changing `regions.json` re-runs it; missing `regions.json` → recorded failure with the documented message.
**CLI / groups / docs:**
13. `omniscan ocr S` (monkeypatched `build_vram_manager` returning a manager whose `vision` loader yields fakes) runs ingest, slice, detect, ocr per chapter, prints one line per stage, exits 0; `--force`, `--chapter` work; `test_gpu_groups.py` updated (the loader returns the three models, `est_gib` 3.0); `omniscan slice` still builds no manager.
14. `tests/unit/test_docs.py` and the whole suite stay green.
**GPU (`@pytest.mark.gpu`; needs the models, downloaded on first use):**
15. `LineDetector.load(OcrConfig(), device)` and `LineRecognizer.load(...)` are fp32 (`param.dtype == torch.float32`); `read_regions` on `make_korean_page(seed=1, n_bubbles=4, n_free=1)` (uint8 `[3,H,W]` tensor on the device; the regions
    are `to_regions_artifact(page)` with `lines=[]`, `text=""` and `confidence=0` — i.e. what detection would give): ≥ 90 % of the regions non-empty, and the mean character error rate (Levenshtein distance / length, both texts with all whitespace removed) over regions is ≤ 0.15.
    Print (do not assert) the same numbers for `font="Gaegu-Regular.ttf"` and the wall time.
16. Detector robustness: `read_regions` on the same page with `n_bubbles=0, n_free=0` regions returns `[]`; on a page whose regions list is the truth but the page image is `clean` (no text), every region comes back empty (`regions_empty == regions`).

## Out of scope
The second-opinion reader (PaddleOCR-VL), Chinese/Japanese models, vertical text, adopting orphan lines as new regions, re-detecting on upscaled crops, changing `OcrConfig`, fp16, GPU connected-component code (measured slower than OpenCV), any web UI change.

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_ocr_lines.py tests/unit/test_ocr_assemble.py tests/unit/test_ocr_model.py tests/unit/test_ocr_pipeline.py tests/unit/test_ocr_stage.py tests/unit/test_gpu_groups.py tests/unit/test_cli.py tests/unit/test_docs.py -q
uv run pytest -q
uv run ruff format . && uv run ruff check .
uv run pyright
```
The full run includes the `gpu` tests; report the measured CER and the seconds of test 15.

## Report
Write `docs/reports/C4a.md` (Changes, Tests, Deviations, Questions) and commit `C4a: OCR stage and omniscan ocr`. If anything above is unclear: stop, write the question
under Questions, commit what you have, and end.

# O1b — OCR engines: model ids for `ppocr`, and the `manga_ocr` crop reader

**Owner:** GLM builder · **Branch:** `O1b` · **Worktree:** `V:\OmniScan-wt\O1b` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read `docs/PRODUCT_SPEC.md` section 1, `config/models.toml` (the OCR entries: fields `role`, `family`, `size_class`, `langs`, `recommended_for`, `upstream_repo`, `upstream_revision`, format `hf`), `src/omniscan/core/config.py` `OcrConfig` (**director-owned, do not edit**: the fields `engine`, `det_model`, `rec_model`, `crop_pad_px`, `crop_batch_size`, `vl_max_new_tokens` already exist), all of `src/omniscan/ocr/` (`model.py`, `pipeline.py`, `assemble.py`, `lines.py`, `stage.py`), `src/omniscan/gpu/groups.py`, `src/omniscan/models/resolve.py` (`local_model_source`), `src/omniscan/models/catalog.py` (`load_catalog`, `ModelEntry`), `tests/unit/test_ocr_model.py`, `test_ocr_pipeline.py`, `test_ocr_stage.py`, `test_gpu_groups.py` (test style, fakes for models).

## Why
Owner wish: the user chooses the OCR engine and model — PP-OCR in every size and version (v5, v6), `manga-ocr` for Japanese, PaddleOCR-VL — and the defaults come from measurements. The catalog already lists every model (card O1a). This card makes the pipeline **use** them: (1) `ocr.det_model` / `ocr.rec_model` select PP-OCR models by catalog id (v5/v6, any size); (2) a second kind of engine that **reads whole region crops** instead of detecting text lines first — `manga_ocr` now, PaddleOCR-VL in card O1d — behind the same stage, artifacts and downstream code. Measured facts (director's spikes, RX 9070 XT, fp32, transformers 5.17):
- PP-OCRv6 and v5 models load through the same `AutoModelForObjectDetection` / `AutoModelForTextRecognition` code as today — only the repo differs.
- `jzhang533/manga-ocr-base-2025` (`ocr-rec-manga-ocr-2025`, 30 M parameters) is a `VisionEncoderDecoderModel` (ViT encoder + BERT decoder). `AutoTokenizer` **fails** on it in transformers 5 (it wants `fugashi`), but the model only needs to *decode*: `vocab.txt` (6144 lines, ids = line numbers, `[PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4`) plus the WordPiece rule "`##` prefix = continuation" is enough. `AutoImageProcessor` works (`ViTImageProcessor`, 224×224, mean/std 0.5). `model.generate(pixel_values)` with its own generation config (beam search 4, max_length 300) read 89 Japanese bubble/free-text crops in 35 s and returned correct text, e.g. `あれ、しまったまた窓を開けたまま寝ちゃったみたい...`. **No `fugashi`, no new dependency.**
Everything below must work with the model folder installed by `omniscan models download <id>` (folder `<models_dir>/<id>`, files as in the catalog) **or** the Hugging Face hub as a fallback, exactly like the existing loaders.

## Files you may create / modify
- `src/omniscan/ocr/engines.py` (create: model-id resolution + engine registry)
- `src/omniscan/ocr/crop_readers.py` (create: `MangaOcrReader` and the pure helpers)
- `src/omniscan/ocr/pipeline.py` (modify: add `read_region_crops`; `read_regions` stays unchanged)
- `src/omniscan/ocr/model.py` (modify: `LineDetector.load` / `LineRecognizer.load` use `model_source`; behaviour with `det_model`/`rec_model` = None stays **identical**)
- `src/omniscan/ocr/stage.py` (modify: choose the path by `cfg.ocr.engine`)
- `src/omniscan/gpu/groups.py` (modify: the vision group loads what the engine needs)
- `tests/unit/test_ocr_engines.py`, `tests/unit/test_ocr_crop_readers.py` (create); extend `test_ocr_pipeline.py`, `test_ocr_stage.py`, `test_gpu_groups.py`, `test_ocr_model.py`
- `tests/unit/test_ocr_manga_gpu.py` (create; marked `gpu`, skipped when the model is not installed)
- `docs/USER_GUIDE.md` (an "OCR engines and models" section; `tests/unit/test_docs.py` validates fenced commands), `docs/reports/O1b.md`
Anything else is off-limits (especially `src/omniscan/core/**`, `config/models.toml`, `detect/`, `translate/`).

## Interfaces (exact)
```python
# ocr/engines.py
@dataclass(frozen=True, slots=True)
class ModelSource:
    repo: str                    # Hugging Face repo id of the catalog entry
    revision: str | None         # pinned commit (catalog upstream_revision) or None
    local: str | None            # installed folder (as str) or None when not installed

def model_source(model_id: str, models_dir: Path | None, catalog: Sequence[ModelEntry] | None = None) -> ModelSource: ...
def load_kwargs(source: ModelSource, *, models_dir: Path | None) -> tuple[str, dict[str, Any]]: ...   # (path_or_repo, from_pretrained kwargs)
DEFAULT_REC_MODEL: dict[str, str] = {"manga_ocr": "ocr-rec-manga-ocr-2025"}   # paddleocr_vl is added by card O1d
def engine_rec_model(cfg: OcrConfig) -> str | None: ...       # cfg.rec_model or DEFAULT_REC_MODEL.get(cfg.engine)

# ocr/crop_readers.py
class TextReader(Protocol):
    def read(self, crops: Sequence[torch.Tensor]) -> list[tuple[str, float]]: ...     # uint8 [3, h, w] crops -> (text, score); LineRecognizer already satisfies it

def decode_wordpieces(ids: Sequence[int], vocab: Sequence[str]) -> str: ...
def clean_manga_text(text: str) -> str: ...
class MangaOcrReader:
    def __init__(self, model: Any, processor: Any, vocab: Sequence[str], device: torch.device, *, batch_size: int) -> None: ...
    @classmethod
    def load(cls, cfg: OcrConfig, device: torch.device, models_dir: Path | None = None) -> MangaOcrReader: ...
    def read(self, crops: Sequence[torch.Tensor]) -> list[tuple[str, float]]: ...

# ocr/pipeline.py
def read_region_crops(strip: torch.Tensor, regions: Sequence[Region], reader: TextReader, cfg: OcrConfig, *, engine: str) -> tuple[list[Region], dict[str, float]]: ...
```

## Definitions (exact)
- **`model_source`**: look `model_id` up in `catalog` (default `load_catalog()`); unknown id → `ValueError(f"unknown OCR model {model_id!r}")`; entry of another format than `hf`/`zip`, or without `upstream_repo` → `ValueError(f"{model_id} is not a Hugging Face model")`. `repo`/`revision` = `upstream_repo`/`upstream_revision`; `local` = `local_model_source(entry.upstream_repo, models_dir, catalog)` when `models_dir` is not None else None.
- **`load_kwargs(source, models_dir=...)`**: installed (`source.local` not None) → `(source.local, {"local_files_only": True})` and `log.info("loading %s from %s", source.repo, source.local)`; else `(source.repo, {"revision": source.revision} if source.revision else {})` and — when `models_dir` is not None — the same `log.warning(...)` text the existing loaders emit ("model … is not installed in …; using the Hugging Face hub/cache. Run "omniscan models download <id>" …", now naming the catalog id).
- **`LineDetector.load` / `LineRecognizer.load`**: when `cfg.det_model` / `cfg.rec_model` is set, build the source with `model_source(...)` and load through `load_kwargs`; when it is `None` keep today's exact code path (`cfg.det_repo` / `rec_repo` + revisions, `local_model_source(repo, ...)`). A `rec_model` whose catalog `role` is not `recognizer` (or `det_model` whose role is not `text_line_detector`) → `ValueError` naming the id and the expected role.
- **`decode_wordpieces(ids, vocab)`**: for every id in order: skip ids whose token is one of `[PAD] [UNK] [CLS] [SEP] [MASK]` and tokens of the form `<unusedN>`; **stop** at the first `[SEP]` (id 3); a token starting with `##` contributes its remainder, every other token contributes itself; concatenate **without spaces**. Ids outside `range(len(vocab))` → `IndexError`. Golden: vocab `["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]","あ","い","##う","、"]`, ids `[2,5,6,7,8,3,5]` → `"あいう、"`; ids `[2,3]` → `""`.
- **`clean_manga_text(text)`** (a port of the upstream post-processing without `jaconv`): (1) remove all whitespace (`"".join(text.split())`); (2) replace `…` with `...`; (3) collapse every run of two or more `.` or `・` characters into the same number of `.` (`re.sub("[・.]{2,}", lambda m: "." * len(m.group()), text)`); (4) convert half-width ASCII letters, digits and the punctuation `! " # $ % & ' ( ) * + , - / : ; < = > ? @ [ \ ] ^ _ { | } ~` to full width by adding `0xFEE0` to the code point — **but keep `.` and `~` as they are**. Golden: `"あれ、 しまった…"` → `"あれ、しまった..."`; `"・・・強い風の音..."` → `"...強い風の音..."`; `"コモナ!"` → `"コモナ！"`; `"ABC 123"` → `"ＡＢＣ１２３"`; `"…"` → `"..."`.
- **`MangaOcrReader.load`**: `rec_model = engine_rec_model(cfg)` (if `cfg.engine != "manga_ocr"` and `cfg.rec_model is None` → `ValueError("manga_ocr needs ocr.rec_model")`); `source = model_source(rec_model, models_dir)`; `path_or_repo, kwargs = load_kwargs(...)`; `torch.cuda.set_device(device)` first when `device.type == "cuda"` (MIOpen note in `LineRecognizer.load`); `VisionEncoderDecoderModel.from_pretrained(path_or_repo, **kwargs)` → `.to(device).eval()` in **fp32**; `AutoImageProcessor.from_pretrained(...)`; `vocab.txt` from the installed folder or, for the hub fallback, `huggingface_hub.hf_hub_download(repo, "vocab.txt", revision=...)`, read as UTF-8, one token per line (strip only the trailing `\n`). `batch_size = cfg.crop_batch_size`.
- **`MangaOcrReader.read(crops)`**: `[]` for no crops. Crops are processed in chunks of `batch_size` **in the given order** (the ViT processor resizes every crop to 224×224, so no width sorting is needed). For a chunk: convert the crops for the processor (uint8 tensors are accepted by the processor; keep them on the GPU if it works, otherwise convert with `torchvision.transforms.functional.to_pil_image` — crops are small, one host round trip per crop is acceptable here), `pixel_values` → device, then `model.generate(pixel_values=..., output_scores=True, return_dict_in_generate=True)` (the model's own generation config: beam 4, max_length 300 — do not override) inside `torch.inference_mode()`. Text = `clean_manga_text(decode_wordpieces(sequence.tolist(), vocab))`. **Score** = `exp(mean(log-probabilities of the generated tokens))`: use `model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False)` for beam search (fall back to `1.0` with a `log.debug` when it raises), average over the non-padding generated positions, `math.exp`, clamp to `[0, 1]`, round to 4 decimals; an empty text gets score `0.0`.
- **`read_region_crops(strip, regions, reader, cfg, *, engine)`**: for every region in order: crop box = the region's `bbox` padded by `cfg.crop_pad_px` and clamped to the strip, at least 1×1 (reuse `_crop_box`); the crop is a **view** of the strip (`strip[:, y0:y1, x0:x1]`, never a copy); all crops go to `reader.read(...)` in one call (the reader batches). Then build the regions with the existing `build_ocr_regions(regions, by_region, readings, engine=engine, lang=cfg.lang, strip_width, strip_height)` where `by_region = {region.id: [LineBox(box=(bbox.x0, bbox.y0, bbox.x1, bbox.y1) as floats, score=1.0)]}` (one "line" per region, key `(region.id, 0)` in `readings`). Metrics dict: `{"tiles": 0.0, "lines": <len(regions)>, "orphan_lines": 0.0, "regions": <len(regions)>, "regions_empty": ..., "regions_low_conf": ...}` computed like `read_regions` does.
- **`OcrStage.run`**: `cfg.ocr.engine == "ppocr"` → today's call (`read_regions` with `models["line_detector"]`, `models["recognizer"]`; `engine=` label = the catalog id when `rec_model` is set, else the repo name as today); any other engine → `read_region_crops(strip, regions.regions, models["reader"], cfg.ocr, engine=<rec model id>)`. The drop rule (`drop_conf`) and `ocr.json` writing are unchanged.
- **`gpu/groups.py`**: the vision group loads `LineDetector` + `LineRecognizer` for `ppocr` (as today) and `{"reader": MangaOcrReader.load(...)}` for `manga_ocr` (in addition to the comic detector, which is always loaded); an engine without a loader yet (`paddleocr_vl`) → `ValueError("OCR engine 'paddleocr_vl' is not available yet")`. Existing tests for the ppocr group must pass unchanged.

## Acceptance tests (CPU, no downloads; fakes and monkeypatching as in the existing OCR tests)
1. **`model_source`**: with a hand-made catalog (an installed hf entry — create the folder and its `.installed.json` like `tests/unit/test_models_resolve.py` does — and a missing one): `local` set / None; unknown id, wrong format (an `ollama` entry) → the exact `ValueError`s; `models_dir=None` → `local is None`.
2. **`load_kwargs`**: installed → `(folder, {"local_files_only": True})` + info log (caplog); missing → `(repo, {"revision": ...})` + the warning naming the id; no revision → `{}`.
3. **`LineDetector.load` / `LineRecognizer.load`** (monkeypatch `transformers.AutoModelForObjectDetection` etc. with recorders as `test_ocr_model.py` does): `det_model`/`rec_model` set → the recorder got the catalog repo/revision (or the installed folder with `local_files_only`); wrong role → `ValueError`; both `None` → the recorder got exactly the arguments it got before this card (regression).
4. **`decode_wordpieces`**: the two goldens, `##` at the start, skipping `<unused3>` and `[MASK]`, stopping at `[SEP]`, an out-of-range id → `IndexError`. **`clean_manga_text`**: the five goldens, empty string, a string of only spaces, mixed full/half width, `.`/`~` kept, tab and newline removed.
5. **`MangaOcrReader.read`** with a **fake model/processor** (a tiny `torch.nn.Module` whose `generate` returns a fixed `sequences` tensor and scores, a processor returning a `pixel_values` tensor): order preserved across chunks (`batch_size=2`, 5 crops), texts decoded through the vocab and cleaned, score = the exp of the mean log-probability you computed by hand for the fake scores, `[]` for no crops, empty decode → `("", 0.0)`, `compute_transition_scores` raising → score `1.0`, a crop that is 1×1 does not crash.
6. **`read_region_crops`**: fake reader recording the crops it receives; three regions → three crops whose shapes equal the padded/clamped boxes (assert `crop.data_ptr()` is inside the strip's storage, i.e. views), one region touching the strip corner is clamped, readings are mapped back to the right regions in input order, an empty text drops the line (`region.lines == []`, confidence 0.0), `OcrLine.engine == engine`, `bbox` equals the region's bbox, metrics dict values as defined (including `regions_low_conf` for a score of 0.5 with `low_conf=0.85`).
7. **Stage**: with `engine="manga_ocr"` and a fake `models={"reader": FakeReader}` the stage writes an `ocr.json` whose regions carry the fake texts and `engine == "ocr-rec-manga-ocr-2025"`; `drop_conf` drops a low-score region; with `engine="ppocr"` the existing tests still pass; `config_subset` changes with `engine` and `rec_model`.
8. **VRAM group**: `build_vram_manager(cfg)` with monkeypatched loaders — `ppocr` loads the detector + recogniser (+ comic detector), `manga_ocr` loads the comic detector + `MangaOcrReader` (assert with recorders that `LineDetector.load` is **not** called), `paddleocr_vl` → the `ValueError`.
9. **GPU smoke** (`@pytest.mark.gpu`, skipped with a clear reason when `local_model_source("jzhang533/manga-ocr-base-2025", cfg.paths.models_dir)` is None): render two crops of Japanese text with Pillow using any installed CJK-capable font from `fonts/` (skip if none has Japanese glyphs — check with `font.getmask`/`getbbox` on `あ`), read them with the real `MangaOcrReader` on `resolve_device(...)` and assert the returned text contains at least half of the characters of the rendered string.
10. `uv run --frozen pytest -q -m "not gpu"` stays green (all existing OCR/GPU-group/CLI tests included).

## Out of scope
PaddleOCR-VL (card O1d), any change to the detector/translation/typeset code, ONNX/llama.cpp engines, choosing defaults per language (the qualification suite O1c decides; `recommended_for` in the catalog is data), a CLI flag (`series.toml` / env vars already reach `cfg.ocr`), fp16/bf16.

## Commands to run before finishing
```bash
uv run --frozen pytest tests/unit/test_ocr_engines.py tests/unit/test_ocr_crop_readers.py tests/unit/test_ocr_pipeline.py tests/unit/test_ocr_stage.py tests/unit/test_ocr_model.py tests/unit/test_gpu_groups.py tests/unit/test_docs.py -q
uv run --frozen pytest -q -m "not gpu"
uv run --frozen ruff format . && uv run --frozen ruff check .
uv run --frozen pyright
```

## Report
`docs/reports/O1b.md`: Changes, Tests (commands + results), Deviations, Questions. **Commit early** (`O1b: WIP engines + crop reader`) once `engines.py`, `crop_readers.py` and their tests pass; the final commit is `O1b: OCR model ids and the manga_ocr engine`. You have 150 tool calls in total. If anything is unclear: stop, write the question under Questions, commit, and end.

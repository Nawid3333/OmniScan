# OmniScan architecture & contracts

Read this before implementing any stage. Contracts live in `src/omniscan/core/` (owned by Claude; builders don't edit).

## Layout on disk
```
<library_root>/<Series>/Chapter N/*.jpg     raws (read-only; written only by `omniscan import`)
<library_root>/<Series>/_reference_en/...   already-translated chapters (reference mode)
<work_root>/<Series>/series.db              glossary, story memory, run registry
<work_root>/<Series>/Chapter N/             manifest.json + JSON/.npz artifacts (no intermediate images)
<output_root>/<Series>/Chapter N/0001.jpg   final English slices
<output_root>/<Series>/_filtered/Chapter N/ promo pages/slices moved out (never deleted)
```
Use `omniscan.core.paths.SeriesPaths` / `ChapterPaths`; never build these paths by hand.
Chapter order = `chapter_number()` then natural sort. Image order inside a chapter = natural sort (`list_images`).

## Coordinate system
All pixel coordinates in artifacts are **strip space**: the chapter's raw images stitched vertically after scaling
to the common strip width. Integers, half-open ranges `[x0, x1)`, `[y0, y1)`. A slice is just a y-range of the strip.
`SourceFile.y0/y1` map each raw file into the strip; `SourceFile.scale = strip_width / original_width`.

## Artifacts (`core/schemas.py`, all pydantic v2, `extra="forbid"`, `schema_version` on top-level models)
| File | Model | Written by |
|---|---|---|
| `ingest.json` | `IngestArtifact` (files → strip rows) | ingest |
| `slices.json` | `SlicesArtifact` (bands, slices, blank/forced/filtered flags) | slicer (+ promo filter flags) |
| `filter.json` | `FilterArtifact` | promo filter |
| `regions.json` | `RegionsArtifact` (text empty) | detect |
| `ocr.json` | `RegionsArtifact` (text filled) | ocr |
| `translations/<run_id>.json` | `CandidateRun` | each translation run |
| `final.json` | `FinalArtifact` | judge |
| `inpaint.json` + `patches.npz` | `InpaintArtifact` + npz (cleaned crops and masks per region) | inpaint |
| `layout.json` | `LayoutArtifact` | typeset |
| `export.json` | `ExportArtifact` (files written to `output_root/<Series>/<Chapter>/`) | export |
| `manifest.json` | `Manifest` of `StageRecord`s | stage runner |
Save/load only through `Artifact.save()` (atomic tmp+rename) and `Model.load(path)`.

## Stages (`core/stage.py`)
A stage is a class with `name`, `version`, `gpu_group` class vars and four methods:
`inputs(ctx) -> list[Path]`, `outputs(ctx) -> list[str]`, `config_subset(cfg) -> Mapping`, `run(ctx, models) -> metrics`.

The runner (`run_stage` / `run_chapter` / `run_series`) skips a stage when the manifest shows the same `version`,
the same **content hash of `inputs()`**, the same **hash of `config_subset()`**, status `done`, and all `outputs()` exist.
So: list every file your output depends on in `inputs()` (upstream artifacts included), and only the config keys
that change your output in `config_subset()`. Bump `version` when the algorithm changes.

In-memory hand-off within one chapter pass: `ctx.lazy("strip", factory)` creates a value once (e.g. the decoded
strip tensor on the GPU) and later stages reuse it; `ctx.put/drop`. The runner clears the memo after each chapter.

## GPU (`gpu/vram.py`)
`VramManager.register(group, loader, est_gib)` + `acquire(group)`. Exactly one group is resident; acquiring the
resident group is free (so a 200-chapter pass loads models once). Acquiring a torch group first evicts local Ollama
models (`keep_alive: 0`, only those with `size_vram > 0`); acquiring `OLLAMA_GROUP` frees all torch memory.
Budget: `cfg.gpu.vram_budget_gib` via `set_per_process_memory_fraction`. The runner records `peak_vram_gib` per stage.

Rules for GPU code: keep tensors on the GPU between steps; no Python loops over rows/pixels; batch work; use pinned
host buffers + `non_blocking=True` for transfers (measured: pinned H2D 49 GB/s vs pageable 12 GB/s).

## JPEG codec (`gpu/codec/base.py`)
`JpegCodec` protocol: `info`, `decode -> uint8 [3,H,W]`, `decode_into(datas, strip, y_offsets)` (stitch for free),
`encode`. Backends: `rocjpeg` (auto if VCN reachable — not in WSL today), `hybrid` (C2: CPU Huffman + GPU IDCT),
`turbo` (baseline for the benchmark). Selection logic arrives with card C2.

## Config (`core/config.py`)
`get_config()` merges defaults → `config/default.toml` → `~/.config/omniscan/config.toml` → env
`OMNISCAN_<SECTION>__<KEY>`. Secrets (`OLLAMA_API_KEY`) come only
from env or `~/.config/omniscan/secrets.env` via `get_secrets()`; never log them.

## Render pass (inpaint → typeset → export)
Stage outputs stay JSON / `.npz`; the pixels of the final chapter are produced only by `export`, which decodes the strip once and edits it in place.
- **inpaint** (`gpu_group` for LaMa later; v1 is flat fill): inputs `ocr.json` (+ `ingest.json`, `slices.json`, raw images); writes `inpaint.json` (`InpaintArtifact`) and
  `patches.npz` with two arrays per region that had text lines: `"<region_id>.pixels"` (uint8 `[h,w,3]`, the strip crop with the text removed) and `"<region_id>.mask"` (bool `[h,w]`),
  both exactly the size of `InpaintItem.box`. Regions whose flat fill failed are marked `needs_lama=True` (a later pass overwrites them).
- **typeset** (no GPU): inputs `ocr.json`, `final.json`, `inpaint.json`; for every region with a final English text it computes the target box (inscribed rectangle of the bubble
  polygon, or the padded text box for free text), fits the text with `typeset/fit.py` and writes `layout.json` (`LayoutArtifact`). Text colour is black on light fills and white on dark ones
  (decided from `InpaintItem.fill` luminance unless the region carries `text_color`).
- **export** (`gpu_group` none, uses the codec's device): inputs all of the above + `slices.json`, raw images; decodes the strip on the GPU (`load_strip`), replaces the **masked**
  pixels of every patch, rasterises every `LayoutItem` into a small RGBA patch (glyphs are drawn with FreeType/PIL on the CPU — the one unavoidable CPU step, question F9 — and
  composited on the GPU), cuts the strip into the non-filtered slices and encodes them to `output_root/<Series>/<Chapter>/0001.jpg …`. Unmasked pixels of the raw pages are never touched.
- **inpaint_lama** (later card, `gpu_group` "inpaint"): handles the `needs_lama` items; writes `inpaint_lama.json` (`InpaintArtifact`, method `"lama"`) and `patches_lama.npz` (same layout as `patches.npz`);
  export applies `patches.npz` first and `patches_lama.npz` after it, so a LaMa patch overrides the flat-fill placeholder of the same region.
- Config sections: `[inpaint]`, `[typeset]`, `[export]` (see `config/default.toml`). `export.json` lists the written slices (name, size, bytes) and is the stage's manifest output.

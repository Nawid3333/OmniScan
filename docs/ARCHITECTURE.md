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
| `ocr.json` | `RegionsArtifact` (text filled, hand edits applied) | ocr (+ editing tools) |
| `ocr_auto.json` | `RegionsArtifact` (the OCR's own reading, before hand edits) | ocr |
| `translations/<run_id>.json` | `CandidateRun` | each translation run |
| `final.json` | `FinalArtifact` (hand-written lines applied) | judge (+ editing tools) |
| `final_auto.json` | `FinalArtifact` (the judge's own lines, before hand edits) | judge |
| `edits.json` | `ChapterEdits` (hand edits: regions, English lines, lettering, checked lines) | editing tools only (`edits/store.py`) |
| `edits_history.json` | `EditsHistory` (earlier states of `edits.json`, for undo and redo) | editing tools only (`edits/store.py`) |
| `cleanup.json` + `cleanup.npz` | `CleanupArtifact` + npz (hand-painted cleanup patches: masks, pixels) | editing tools only (`cleanup/store.py`; a "lama" stroke is rebuilt by `cleanup/lama_now.py` through `inpaint/lama_pipeline.lama_regions`, the model loaded on demand from the VRAM manager's inpaint group); applied last by export |
| `inpaint.json` + `patches.npz` | `InpaintArtifact` + npz (cleaned crops and masks per region) | inpaint |
| `layout.json` | `LayoutArtifact` (hand lettering applied) | typeset |
| `export.json` | `ExportArtifact` (files written to `output_root/<Series>/<Chapter>/`) | export |
| `qa.json` | `QaArtifact` (regions whose original text is still readable on the exported pages) | `qa` stage (`omniscan qa`, not part of `omniscan run`) |
| `manifest.json` | `Manifest` of `StageRecord`s | stage runner |
| `<series>/memory.json` | `SeriesMemory` (learned rules + translation memory of the whole series) | `learn/memory.py`, rebuilt from every `edits.json` when one changes; only rule switches are set by hand |
| contribution archive (`.zip`, anywhere) | `Contribution` in `contribution.json` + `pages/<chapter id>/<page>.jpg` (hand corrections and checked lines with the pages they are on) | `share/contribution.py` (`omniscan contribute export`); never read by the pipeline |
Save/load only through `Artifact.save()` (atomic tmp+rename) and `Model.load(path)`.

**Hand edits** (`edits/`): `edits.json` is user-owned — only the editing tools write it (web Studio, via
`edits/store.py`). The `ocr` and `judge` stages write their own output to `ocr_auto.json`/`final_auto.json`
and `ocr.json`/`final.json` as that output with `edits.json` applied (`edits/apply.py`, pure functions); an
editing tool rebuilds the same two files from the `_auto` ones with the same functions. An edit is matched to
its region by id and box overlap (IoU >= 0.5 with the box it was made on), so it survives a re-run that
renumbers regions. `edits.json` is deliberately *not* an input of `ocr` (a text fix must not re-run the OCR
models); the tools apply it themselves. It *is* an input of `typeset`, which is cheap: its hand lettering
(`layout`) is applied there (`typeset/overrides.py`), and `typeset/chapter.py::chapter_layout` is the one
function both the stage and the studio's live preview (`typeset/page_preview.py`) letter a chapter with.
Every change of `edits.json` goes through `edits/store.py::save_edits`, which pushes the state it replaces
onto `edits_history.json` (at most `HISTORY_DEPTH` steps); `edit_group` makes several operations one step (an
import, a desktop save), and `undo`/`redo` swap states and rebuild `ocr.json`/`final.json` like any edit.
`edits/replace.py` is find & replace over English lines or source texts: a pure rule (`FindReplace`), a
per-chapter `plan`, and `apply_changes` through the same store operations (one `edit_group` per chapter).
`edits/session.py::StudioSession` is the desktop Translator Studio's view of one chapter: it holds the page's
changes in memory and saves them through the same `edits/store.py` operations (the web Studio and `omniscan edit`
call those directly).

**Per-line status** (`edits/store.py::line_statuses`): a `LineCheck` in `edits.json` records a proofreader
approving a line — the region's source text and English at that moment, matched to its region like the other
edits (`apply.match_checks`). A line is `checked` while both still equal the check, else `edited` when it carries
a region edit or a hand-written line, else `todo`; a stale check stays recorded (an undo can make it hold again).

**Learning** (`learn/`): `learn/harvest.py` compares every edit with the pipeline text it records
(`RegionEdit.auto_text`, `TranslationEdit.auto_text`, set when the edit is first made) — OCR fixes,
deletions, watermark/sfx labels, English lines, rewritten words. `learn/memory.py` turns a series'
corrections into `memory.json`: word rules need `learn.min_count` matching corrections and none that kept
the old word; the translation memory keeps the latest English per source line. `learn/apply.py` applies it:
the `ocr` stage runs `apply_to_regions` on its reading (so `ocr_auto.json` holds the lessons), the
`translate` stage and the studio's on-demand translation pass `TranslationHints` to `run_profile` (an exact
memory line is used as the candidate; similar lines and preferred words go into the chat_json prompt).
`memory.json` is not a stage input file (every edit rewrites it); instead the two stages declare what they take
from it through the optional `input_extra(ctx)` hook (`core/stage.py`, hashed with the input files): the
`ocr` stage its active lessons (`learn/apply.py::ocr_lessons`), the `translate` stage the remembered English of
this chapter's lines (`remembered_lines`). A switched-off rule re-runs the OCR; a new hand translation re-runs
only the chapters holding that line, and an exact memory line wins over a reused candidate.

**Page images** (`translate/images.py`): a `chat_json` profile with `images = true` attaches the page tiles its
request's regions sit on (`PageImages`: read from the raw pages with Pillow; each slice scaled so the strip width
fits `image_side` and cut into tiles at most `image_side` px tall (`tiles_of`), so a webtoon slice keeps its
detail; base64 JPEG in the Ollama message's `images`). Each region carries `image` and `box` in its tile's pixels
(the tile holding its vertical centre), `run.py::_chunks` cuts requests at `images_per_request` images, and a
repair round sends only the missing regions' tiles. The stage builds one `PageImages` for all its profiles.
`translation_key` adds the image size and the tile's identity (its rows and the sha256 of the raw pages under
it, from ingest.json) only for a line actually sent with its image (`PageImages.sent`), so profiles without
images keep their keys, a replaced page re-translates the lines it shows, and a line sent text-only is sent
again once its page can be read. The stage hash sees the image settings only when they are on
(`stage_profile`), and with images the stage's inputs add ingest.json, slices.json and the raw pages. Local
models get a token per 28 px patch of each image added to `num_ctx`, which never shrinks between image requests
of one model (no reload per request). Unreadable pages degrade to a text-only request (logged once).

**Speakers and voices** (`translate/voices.py`): `Region.speaker` is set by hand (a `RegionEdit.speaker`, applied
like every region edit); a series' hand-written `voices.toml` (library dir, next to `series.toml`) describes how
each character talks. The chat_json prompt carries each region's speaker and the voices of the characters who
speak or are named in the request; `translation_key` includes a region's speaker and voice only when it has a
speaker, so unassigned lines keep their keys. `voices.toml` is an input of the translate stage.

**Contributions** (`share/contribution.py`, X4 in `docs/ROADMAP.md`): what a user shares so the models and defaults
improve, for now as a local zip archive. `build` walks a series' chapters: each current region is paired with the
pipeline's output (`ocr_auto.json` / `final_auto.json`, or the reading an edit recorded in `auto_text`) and its
hand edits (matched like the stages re-apply them), deleted pipeline regions — and detections a hand-drawn box
replaced (`replaced_by`) — are added back as `deleted`, a line whose proofreader's check still holds
(`edits.store.line_statuses`) is marked `checked` (verified data: the machine was right), and only pages holding a
correction or a checked line are kept, in page pixels (strip resolution). `write_archive` re-encodes those pages from the raw files (Pillow, like the PSD export:
no metadata) and gives every entry the same fixed timestamp; `contribution.json` holds no date. Names never leave
the machine: the series and chapters are HMAC-SHA256 ids keyed with a random per-install salt
(`<work_root>/contribution-salt`), fonts are file names. `[share] enabled` (`ShareConfig`) gates `build`: false in
config.toml opts out every series, false in a series.toml that series; no upload exists yet.

**Consistency** (`qa/consistency.py`): a series-wide proofreading report, never a stage and never written to
disk: `series_lines` reads every chapter's current regions and English lines, `divergences` groups repeated
source lines with different English, `term_misses` checks locked glossary terms with `glossary/match.py`
(served by `omniscan consistency` and `GET /api/series/{series}/consistency`).

**Reading one region again** (`ocr/on_demand.py`): the Studio's *Read again* and `omniscan edit ocr` cut the
region's box plus a 24 px margin from the raw pages (`cleanup/strip.py`, Pillow) and pass it to the `ocr`
stage's own `read_regions` (ppocr: lines found inside the crop) or `read_region_crops` with the region's box
in crop pixels; the models come from a VRAM manager's vision group under the GPU lock, for that one read
(`pipeline/on_demand.group_models`, the one loader of every on-demand action: this, finding missed text and
LaMa on a selection). The
reading is only returned; keeping it is a hand edit (`update_region(text=…)`). The module imports torch only
inside `read_region`, so the web app and the CLI stay torch-free at import.

**Finding missed text on one page** (`detect/on_demand.py`): the Studio's *Find missed text* and
`omniscan edit find` crop one raw page's strip rows, run the `detect` stage's own tiling, merging and
`build_regions` over it (optionally at another detector threshold, restored afterwards), drop every box a
current region covers (the box mostly inside the region, or the region mostly inside the box), and read the
rest with `ocr/on_demand.read_in` from the same crop, all under one vision-group load. The finds are returned
as suggestions; adding one is `edits.store.add_region`, so `regions.json` / `ocr_auto.json` and the manifest
never change.

**Interchange** (`interchange/`): other tools' files in and out of a chapter, never a stage. Out: LabelPlus
files (`labelplus.py`), layered PSD pages (`psd.py`) and BallonsTranslator projects (`ballons.py`), built from
the chapter's artifacts on the CPU. In: LabelPlus labels (a point each) go to the region they point into
(`labelplus.py::match_labels`); BallonsTranslator and manga-image-translator text blocks (`ballons.py`, `mit.py`,
a box each in page pixels) go to the region they overlap most (`blocks.py::match_blocks`). Everything imported
is recorded through `edits/store.py`, like any hand edit. Between OmniScan users, a chapter travels as a project archive
(`project.py`): `raw/`, `work/` (the whole chapter work dir), `series/` (series.toml, voices.toml) and optionally
`output/`, listed with sha256 in a `ChapterProject` (`project.json`). Stage input hashes cover file names and
contents, not locations, so an unpacked chapter's stages stay done. `unpack` checks the listing (only those
parts, plain names, no duplicates up to case, a size cap), stages into hidden sibling folders, verifies every
file's size and sha256, and only then moves the folders into place.

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
`[importer] downloader` (`ImporterConfig`) is the command `omniscan import --from-url` runs (`importer/from_url.py`):
manhwa-manga-downloader stays a subprocess plug-in, read only through its versioned `--json` result object
(`schema` 1; a missing field reads as 1, any other value is refused).

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

## Performance tests (`tests/perf`, `pytest --perf`)
`tests/perf/test_pipeline_perf.py` runs the real pipeline (all ten stages) on the E1 golden test's synthetic
pages, twice in one run, and measures the second chapter, so model loading and first-use warm-up do not count.
Each stage's seconds are compared with this device's baseline in `tests/perf/baselines.json` (keyed by
`cuda:<device name>`): a stage fails when it is slower than its baseline by more than 30 % plus 0.25 s
(`tests/perf/budget.py`). A device without a baseline only prints its timings. `pytest --perf --perf-update`
records the current timings as the device's baseline; commit the updated `baselines.json` with the change that
explains the new numbers. The perf tests are skipped unless `--perf` is given (and need the GPU and the cached
model weights, like E1), so the CI suite never runs them.


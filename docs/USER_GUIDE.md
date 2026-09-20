# OmniScan user guide

This guide covers everything OmniScan can do today. Commands and flags are checked against the CLI by
`tests/unit/test_docs.py`, so this document cannot silently drift from the program.

## Folder layout

OmniScan keeps three roots (see `[paths]` in [Configuration](#configuration)):

| Path | Written by | Contents |
|---|---|---|
| `library_root/<Series>/<Chapter N>/` | `omniscan import` | raw chapter images (read-only for the pipeline) |
| `library_root/<Series>/_reference_en/` | reference mode (not implemented yet) | already-translated chapters used as style reference |
| `work_root/<Series>/series.db` | glossary seeding (no command creates it yet) | SQLite glossary working copy |
| `work_root/queue.db` | the job queue | every queued/running/finished job |
| `work_root/<Series>/glossary.yaml` | `omniscan glossary export` | hand-editable glossary review file |
| `work_root/<Series>/watermarks.json` | `omniscan watermark add` / `remove` | fixed-position watermark regions |
| `work_root/<Series>/<Chapter N>/ingest.json` | `omniscan ingest` | strip layout: files, widths, y-ranges |
| `work_root/<Series>/<Chapter N>/slices.json` | `omniscan slice` | bands and slices in strip space |
| `work_root/<Series>/<Chapter N>/regions.json` | `omniscan detect` | bubble and text regions with ids and reading order |
| `work_root/<Series>/<Chapter N>/inpaint.json` | `omniscan inpaint` | per-region clean record: method, fill colour, needs-lama flag |
| `work_root/<Series>/<Chapter N>/patches.npz` | `omniscan inpaint` | cleaned crops and text masks per region (export applies them) |
| `work_root/<Series>/<Chapter N>/layout.json` | `omniscan typeset` | per-region font, size, wrapped lines, box and colour |
| `work_root/<Series>/<Chapter N>/ocr.json` | `omniscan ocr` | the regions with their text lines, text and confidence |
| `work_root/<Series>/<Chapter N>/export.json` | `omniscan export` | the written slices: quality, subsampling, file list |
| `work_root/<Series>/<Chapter N>/inpaint_lama.json` | `omniscan inpaint --lama` | LaMa clean record for the regions the flat fill could not clean |
| `work_root/<Series>/<Chapter N>/patches_lama.npz` | `omniscan inpaint --lama` | LaMa-cleaned crops and masks (export applies them after `patches.npz`) |
| `work_root/<Series>/<Chapter N>/filter.json` | `omniscan filter run` / `restore` | promo-filter decisions |
| `work_root/<Series>/<Chapter N>/manifest.json` | the stage runner | per-stage status, inputs and config hashes |
| `work_root/<Series>/<Chapter N>/converted/` | `omniscan ingest` | JPEG cache for raws that were not JPEG |
| `output_root/<Series>/<Chapter N>/` | `omniscan export` | final English slices (`0001.jpg` …) |
| `output_root/<Series>/_filtered/<Chapter N>/` | `omniscan filter run` | byte-copies of promo-filtered raw files (never deleted) |
| `output_root/<Series>/_packaged/` | `omniscan pack` | finished `.cbz` / `.pdf` files |

`<Chapter N>` means whatever the folder is actually called — chapter names are parsed with
`chapter_number()` and sorted in reading order. The promo filter reads its example images from
`paths.promo_examples` (default `~/omniscan/promo_examples`) in two non-recursive subfolders:
`global/` (shared across series) and `<Series>/` (series-specific).

## Configuration

Load order (later wins): built-in defaults (`src/omniscan/core/config.py`) → `config/default.toml`
(in the repo) → `~/.config/omniscan/config.toml` (per machine) → environment variables
`OMNISCAN_<SECTION>__<KEY>` (e.g. `OMNISCAN_GPU__VRAM_BUDGET_GIB=13`).

```bash
OMNISCAN_PATHS__LIBRARY_ROOT=/data/omniscan/library OMNISCAN_GPU__CODEC=turbo uv run omniscan slice DemoSeries
```

Every key in `config/default.toml`:

| Key | Meaning | Effect today |
|---|---|---|
| `paths.library_root` | where raw chapters live | yes |
| `paths.work_root` | where artifacts and per-series dbs live | yes |
| `paths.output_root` | where final English chapters live | yes |
| `paths.promo_examples` | where promo-filter example images live | yes (`filter run`) |
| `paths.models_dir` | local cache for model weights (managed by `omniscan models`; catalog in `config/models.toml`) | yes (LaMa weights for `omniscan inpaint --lama`; the vision models land there too, card U2b) |
| `gpu.device` | torch device: `auto` (default: strongest discrete GPU, else Apple MPS, else CPU), `cpu`, `mps`, `cuda:N` | yes (a named GPU that is unreachable falls back to CPU) |
| `gpu.vram_budget_gib` | VRAM budget in GiB | yes (the comic detector loads inside this budget) |
| `gpu.codec` | `auto` / `rocjpeg` / `hybrid` / `turbo` | only `auto`/`turbo` work — both run the CPU `turbo` codec; `rocjpeg`/`hybrid` are not implemented yet |
| `slicer.band_min_px` | smallest strip band kept, in px | yes |
| `slicer.target_height` | preferred slice height, in px | yes |
| `slicer.min_height` / `slicer.max_height` | acceptable slice height range, in px | yes |
| `slicer.hard_max_height` | hard ceiling, in px (forces a cut) | yes |
| `slicer.uniform_tol` | tolerance for treating neighbouring rows as uniform gutter, in px | yes |
| `slicer.max_drift` | allowed panel-edge drift across a cut | yes |
| `detect.repo` | Hugging Face repo of the comic detector | yes (`omniscan detect`) |
| `detect.threshold` | minimum detector score kept before merging | yes |
| `detect.tile_px` | tile side in strip pixels (capped at the strip width) | yes |
| `detect.overlap` | tile overlap as a fraction of the tile side | yes |
| `detect.batch_size` | tiles per forward pass | yes |
| `detect.nms_iou` | same-class IoU above which the lower-scored box is dropped | yes |
| `detect.contain_thr` | a tile-edge box mostly inside a same-class box is dropped at/above this containment | yes |
| `detect.edge_penalty` | score penalty for boxes cut by an internal tile edge | yes |
| `detect.merge_bubble_text` | several text boxes inside one bubble become one region | yes |
| `detect.reading_direction` | order of regions within a row: `ltr`, or `rtl` for manga | yes |
| `inpaint.pad_px` | patch = union of a region's line boxes grown by this, in px | yes |
| `inpaint.mask_dilate_px` | line boxes are grown by this to form the text mask, in px | yes |
| `inpaint.flat_tol` | 90th-percentile colour deviation of the ring under which a flat fill is accepted | yes |
| `inpaint.min_ring_px` | fewer ring pixels than this → no flat fill | yes |
| `typeset.min_px` | smallest font size tried, in px | yes (`omniscan typeset`) |
| `typeset.max_px` | largest font size tried, in px | yes (`omniscan typeset`) |
| `typeset.line_spacing` | line pitch as a multiple of the font size | yes (`omniscan typeset`) |
| `typeset.margin_px` | inset of the bubble's inscribed rectangle, in px | yes (`omniscan typeset`) |
| `typeset.free_grow` | free text / SFX boxes are grown by this fraction on every side | yes (`omniscan typeset`) |
| `typeset.stroke_free_px` | outline width of free-standing text, in px | yes (`omniscan typeset`) |
| `typeset.stroke_sfx_px` | outline width of sound effects, in px | yes (`omniscan typeset`) |
| `ocr.det_repo` | Hugging Face repo of the PP-OCRv5 text-line detector | yes (`omniscan ocr`) |
| `ocr.det_revision` | pin the detector to a specific Hugging Face revision | yes (empty = latest) |
| `ocr.rec_repo` | Hugging Face repo of the recognition model; also recorded per line as its engine | yes (`omniscan ocr`) |
| `ocr.rec_revision` | pin the recognizer to a specific Hugging Face revision | yes (empty = latest) |
| `ocr.tile_px` | tile side in strip pixels (capped at the strip width) | yes |
| `ocr.overlap` | tile overlap as a fraction of the tile side | yes |
| `ocr.det_threshold` | minimum detector score kept before post-processing | yes |
| `ocr.box_threshold` | minimum box score kept by the detector post-processor | yes |
| `ocr.unclip_ratio` | how far detected line boxes are unclipped outward | yes |
| `ocr.min_size` | smallest line box kept, in px | yes |
| `ocr.rec_batch_size` | line crops per recognition forward pass | yes |
| `ocr.line_pad_px` | padding around a line before it is cropped for reading | yes |
| `ocr.assign_min_ioa` | a line joins the region containing at least this fraction of it | yes |
| `ocr.region_pad_px` | region box padding used when assigning lines to regions | yes |
| `ocr.nms_iou` | same-line IoU above which the lower-scored duplicate is dropped | yes |
| `ocr.low_conf` | regions whose confidence is below this are counted as low-confidence in the metrics | yes |
| `ocr.drop_conf` | regions with an OCR confidence below this (or with no readable text) are dropped from `ocr.json`; they are detector false positives whose pixels stay untouched | yes |
| `ocr.lang` | language code recorded on every OCR'd region | yes |
| `export.jpeg_quality` | JPEG quality of the exported slices | yes (`omniscan export`) |
| `export.subsampling` | chroma subsampling of the exported slices: `444`, `422` or `420` | yes (`omniscan export`) |
| `inpaint.lama_url` | where the LaMa TorchScript weights are downloaded from | yes (`omniscan inpaint --lama`) |
| `inpaint.lama_sha256` | expected sha256 of the LaMa weights, checked after every download | yes (`omniscan inpaint --lama`) |
| `inpaint.lama_file` | file name of the weights inside `<models_dir>/lama/` | yes (`omniscan inpaint --lama`) |
| `inpaint.lama_window` | one fixed window side in px (every new window shape costs a 10–25 s warm-up) | yes (`omniscan inpaint --lama`) |
| `inpaint.lama_dilate_px` | extra growth of the flat-fill mask for LaMa, in px | yes (`omniscan inpaint --lama`) |
| `inpaint.lama_context_px` | a region needs at least this much context inside the window on every side, else it is skipped | yes (`omniscan inpaint --lama`) |
| `ollama.local_url` | local Ollama base URL | yes (`doctor`) |
| `ollama.cloud_url` | Ollama cloud base URL | yes (`doctor`) |
| `ollama.request_timeout_s` | per-request timeout | used by the LLM client module (no pipeline stage yet) |
| `relay.url` | relay deployment URL | no consumer yet |

Secrets never go in TOML. Put them in `~/.config/omniscan/secrets.env` (or export them); all three are
optional today and `omniscan doctor` reports which are missing. Never commit this file.

```text
OLLAMA_API_KEY=...            # Ollama cloud (`translate` cloud profiles; doctor reports if unset)
EXTRACTPICS_API_KEY=...
OMNISCAN_RELAY_CLIENT_TOKEN=...
```

## Commands

Stubs (`acquire`,
`reference`) are registered but not usable; each prints `not implemented yet` and exits 2. Every other
command below is fully working. The examples assume you generated the demo chapter with
`uv run python scripts/make_demo_chapter.py`.

### `omniscan doctor`

Check this machine is ready for OmniScan.

| Option | Meaning |
|---|---|
| `--json` | emit a JSON array instead of a table |

Checks Python 3.14, the ROCm torch build and GPU, `rocminfo`/gfx1201, the rocJPEG decoder, the local
Ollama server, the required Ollama models, the Ollama cloud key (only when one is set), the secrets
file, the pipeline roots, the required model-catalog downloads and the configured codec. Any `FAIL`
row makes the command exit 1; `WARN` rows do not. The `models` row warns with the total size to
download while required models are missing (until then the pipeline loads them from the Hugging Face
hub/cache). Writes nothing.

```bash
uv run omniscan doctor
```

### `omniscan hardware`

Report what this machine offers: OS and architecture, the CPU (name, physical/logical cores), RAM,
the torch build (`cuda`/`rocm`/`mps`/`cpu`), the device `gpu.device = "auto"` would pick, one line
per torch-visible GPU (best first: discrete before integrated, then more VRAM) with its backend and
device string, the ONNX Runtime providers when onnxruntime is installed, and the free disk space at
`paths.models_dir`. Integrated GPUs are marked `(integrated)`. The same snapshot as JSON is what the
future settings screen reads, and `omniscan models list` assesses each catalog model against it.

| Option | Meaning |
|---|---|
| `--json` | emit the hardware snapshot as one JSON object |

Never fails when torch is missing or broken: it then reports no GPUs and `best device: cpu`. Writes
nothing.

```bash
uv run omniscan hardware
uv run omniscan hardware --json
```

### `omniscan import`

Import raw chapter images from a local folder into the library. The source folder can be one chapter
folder (all images), a folder of chapter folders (each child names a chapter), or a flat dump of
images whose filenames carry an explicit chapter marker (`Ch1`, `Chapter_02`, `ep3`, …). Series and
chapter are guessed from the folder names unless you override them.

| Argument/option | Meaning |
|---|---|
| `source` | directory to read (required) |
| `--series <str>` | override the destination series name |
| `--chapter <str>` | override the destination chapter name |
| `--move` | move instead of copy; delete the source after import |
| `--dry-run` | print the plan without writing anything |

Reads the source folder; writes `library_root/<series>/<chapter>/`. Exit 2 when the source cannot be
planned unambiguously (empty folder, mixed subfolders and loose images, no parseable chapter names).

```bash
uv run omniscan import ~/Downloads/DemoSeries --dry-run
uv run omniscan import ~/Downloads/DemoSeries
```

### `omniscan ingest`

Normalise raw chapter images to JPEG and record the strip layout (`ingest.json`). The raws are
stitched vertically into a common strip width; non-JPEG raws get a JPEG copy under `converted/`.

| Argument/option | Meaning |
|---|---|
| `series` | series name (required) |
| `--chapter`, `-c <str>` | chapter folder name; repeatable. Default: all |
| `--force` | re-run even if up to date |

Reads `library_root/<series>/<chapter>/`; writes `work_root/<series>/<chapter>/` (`ingest.json`,
`manifest.json`, `converted/`). Exit 2 if the series has no chapters; exit 1 if any chapter fails.

```bash
uv run omniscan ingest DemoSeries
```

### `omniscan slice`

Cut chapter strips into slices (`slices.json`). Runs `ingest` first if needed.

| Argument/option | Meaning |
|---|---|
| `series` | series name (required) |
| `--chapter`, `-c <str>` | chapter folder name; repeatable. Default: all |
| `--force` | re-run even if up to date |

Reads `ingest.json` and the raw images; writes `slices.json` + `manifest.json` in the chapter work
dir. Exit codes as for `ingest`.

```bash
uv run omniscan slice DemoSeries
```

### `omniscan detect`

Detect speech bubbles and text in chapter strips (`regions.json`). Runs `ingest` and `slice` first if
needed. The RT-DETR-v2 comic detector (`detect.repo`, downloaded from Hugging Face on first use) runs
on the configured GPU in fp32 over overlapping square tiles of the strip; tiles of blank and filtered
slices are skipped, detections are merged across tiles, and every kept text box becomes a region —
`bubble_text` when the detector also saw its bubble, `free_text` for text outside bubbles — with
chapter-stable ids and a reading order that follows `detect.reading_direction`.

| Argument/option | Meaning |
|---|---|
| `series` | series name (required) |
| `--chapter`, `-c <str>` | chapter folder name; repeatable. Default: all |
| `--force` | re-run even if up to date |

Reads `ingest.json`, `slices.json` and the raw images; writes `regions.json` + `manifest.json` in the
chapter work dir. Exit codes as for `ingest`.

```bash
uv run omniscan detect DemoSeries
```

### `omniscan ocr`

Read the text of the detected regions (`ocr.json`). Runs `ingest`, `slice` and `detect` first if
needed. Two PP-OCRv5 models (`ocr.det_repo`, `ocr.rec_repo`, downloaded from Hugging Face on first
use) run on the configured GPU in fp32: the text-line detector runs over overlapping square tiles of
the strip — only tiles that touch a region — the line boxes are merged across tiles (lower-scored
duplicates above `ocr.nms_iou` are dropped) and assigned to the region containing most of each line,
then every assigned line is cropped from the strip (padded by `ocr.line_pad_px`) and read by the
recognition model in batches of `ocr.rec_batch_size`. Each region ends up with its lines (reading
order, per-line text/score/engine), its text joined with newlines, a confidence (the minimum line
score) and `ocr.lang` as its language.

| Argument/option | Meaning |
|---|---|
| `series` | series name (required) |
| `--chapter`, `-c <str>` | chapter folder name; repeatable. Default: all |
| `--force` | re-run even if up to date |

Reads `ingest.json`, `slices.json`, `regions.json` and the raw images; writes `ocr.json` +
`manifest.json` in the chapter work dir. Exit codes as for `ingest`.

```bash
uv run omniscan ocr DemoSeries
```

### `omniscan inpaint`

Remove the source text from every OCR region and record the result (`inpaint.json` + `patches.npz`).
For every region with text lines a mask covers its text; where the surroundings of the text are one
flat colour (a white or dark bubble interior) the masked pixels are replaced by exactly that colour.
Regions on textured art cannot be cleaned this way — they are recorded with a `needs_lama` flag, and
SFX regions always are.

With `--lama` a second stage (`inpaint_lama`) cleans exactly those regions with the LaMa model
(TorchScript `big-lama.pt`, Apache-2.0): for every such region a fixed 512×512 window of the strip
around it is inpainted on the GPU in fp32 and stored as a patch in `patches_lama.npz` (+
`inpaint_lama.json`); export applies `patches.npz` first and `patches_lama.npz` after it. The weights
(205 MB) are downloaded from `inpaint.lama_url` into `paths.models_dir/lama/` on first use and their
sha256 is checked; regions wider or taller than
`inpaint.lama_window - 2 * inpaint.lama_context_px` are skipped. The pixels themselves stay out of
the JSON: the `.npz` files hold one cleaned crop plus its mask per region, and the later export stage
applies them. Like `translate`, this needs `ocr.json`, which no stage produces yet.

| Argument/option | Meaning |
|---|---|
| `series` | series name (required) |
| `--chapter`, `-c <str>` | chapter folder name; repeatable. Default: all |
| `--force` | re-run even if up to date |
| `--lama` | Also clean regions on textured art with LaMa (downloads 205 MB on first use) |

Reads `ingest.json`, `slices.json`, `ocr.json` and the raw images; writes `inpaint.json` and
`patches.npz` — with `--lama` also `inpaint_lama.json` and `patches_lama.npz` — in the chapter work
dir. Exit codes as for `ingest`.

```bash
uv run omniscan inpaint DemoSeries
uv run omniscan inpaint DemoSeries --lama
```

### `omniscan export`

Write the finished English slices of a chapter to `output_root/<series>/<chapter>/` and record them in
`export.json`. The chapter strip is re-decoded from the raws, every cleaned crop from `patches.npz` is
pasted back through its text mask (crops from `patches_lama.npz` last, when the LaMa pass exists — they
override the flat fill), every region in `layout.json` is rendered with its font and blended over the
strip, and the slices that survived the promo filter are re-encoded as `0001.jpg`, `0002.jpg`, …
Unmasked pixels are never modified, and a re-export deletes only stale `NNNN.jpg` files from the output
folder — anything else you keep there stays.

| Argument/option | Meaning |
|---|---|
| `series` | series name (required) |
| `--chapter`, `-c <str>` | chapter folder name; repeatable. Default: all |
| `--force` | re-run even if up to date |

Reads `ingest.json`, `slices.json`, `inpaint.json`, `patches.npz`, `layout.json` and the raw images
(plus `inpaint_lama.json` + `patches_lama.npz` when they exist); writes `export.json` in the chapter
work dir and the slice JPEGs into `output_root/<series>/<chapter>/`, encoded with the `[export]`
quality and subsampling. It does not run the earlier stages for you — a missing input fails the chapter
with the stage to run named. Exit codes as for `ingest`.

```bash
uv run omniscan export DemoSeries
```

### `omniscan run`

Run the whole pipeline over a series: the ten stages in three passes (vision: `ingest` → `ocr`, text:
`translate` → `judge`, render: `inpaint` → `export`), so each model group is loaded once per pass
instead of once per chapter. Chapters whose stage fails are dropped from the later passes; everything
is resumable through the chapter manifests, so a re-run only executes what changed.

| Argument/option | Meaning |
|---|---|
| `series` | series name (required) |
| `--chapter`, `-c <str>` | chapter folder name; repeatable. Default: all |
| `--stage`, `-s <name>` | stage name; repeatable. Default: all ten (`ingest`, `slice`, `detect`, `ocr`, `translate`, `judge`, `inpaint`, `inpaint_lama`, `typeset`, `export`) |
| `--no-lama` | skip the LaMa inpaint stage (`inpaint_lama`) |
| `--force` | re-run stages even if up to date |

The text pass talks to Ollama (local or cloud models); the vision and LaMa models are loaded through
the VRAM manager, which frees the previous group first, so the three passes never fight over 16 GB.
One line is printed per stage outcome, then a summary line. Exit 2 for an unknown series or stage
name; exit 1 when any chapter failed; exit 3 on an Ollama rate limit (partial results are kept —
re-run later).

```bash
uv run omniscan run DemoSeries
uv run omniscan run DemoSeries -c "Chapter 1" -s ingest -s slice
uv run omniscan run DemoSeries --no-lama --force
```

### `omniscan filter run`

Filter promo files/slices of a chapter against the user's promo examples.

| Argument/option | Meaning |
|---|---|
| `series`, `chapter` | required |
| `--threshold <float>` | similarity threshold for a filtered verdict. Default: 0.9 |

Needs `ingest.json` and `slices.json` (exit 2 with a message if either is missing). Every raw file and
every slice is d-hashed and matched against `promo_examples/global` and `promo_examples/<Series>`
images; decisions go to `filter.json`, and files/slices that are judged promos are byte-copied into
`output_root/<series>/_filtered/<chapter>/` (nothing is deleted). Exit 0 even when nothing is
filtered; with no examples configured everything is kept.

```bash
uv run omniscan filter run DemoSeries "Chapter 1"
```

### `omniscan filter restore`

Restore a previously filtered file or slice (metadata override; nothing is deleted).

| Argument | Meaning |
|---|---|
| `series`, `chapter` | required |
| `target` | `file` or `slice` |
| `index` | the index of the decision in `filter.json` (`file` index = source-file index, `slice` index = slice index) |

Appends a manual `restored` decision to `filter.json`. Requires `filter.json` from a previous
`filter run`.

```bash
uv run omniscan filter restore DemoSeries "Chapter 1" slice 0
```

### `omniscan glossary list`

Print the glossary of a series as a table (source, target, type, status, count).

| Argument/option | Meaning |
|---|---|
| `series` | required |
| `--status <proposed\|locked\|rejected>` | only entries with this status |

Requires `work_root/<series>/series.db` (exit 2 with a message otherwise). Nothing writes that db
today, so this is only usable once the glossary-seeding stage lands.

```bash
uv run omniscan glossary list DemoSeries
```

### `omniscan glossary export`

Write the full glossary of a series to its `glossary.yaml` (hand-editable).

| Argument | Meaning |
|---|---|
| `series` | required |

Same `series.db` requirement as `list`; writes `work_root/<series>/glossary.yaml`.

```bash
uv run omniscan glossary export DemoSeries
```

### `omniscan glossary import`

Import glossary entries from the series' `glossary.yaml` into its db.

| Argument/option | Meaning |
|---|---|
| `series` | required |
| `--mode <merge\|replace>` | merge updates by source, replace rebuilds. Default: merge |

Same `series.db` requirement; exits 2 if `glossary.yaml` does not exist.

```bash
uv run omniscan glossary import DemoSeries --mode merge
```

### `omniscan translate`

Run translation profiles over a chapter's OCR text and write one candidate run per profile to
`work_root/<series>/<chapter>/translations/<profile>.json`. Needs `ocr.json` from `omniscan ocr`, so
run that first.

| Argument/option | Meaning |
|---|---|
| `series` | required |
| `--chapter`, `-c <str>` | chapter folder name; repeatable. Default: all |
| `--profile`, `-p <name>` | profile name; repeatable. Default: every enabled profile |
| `--force` | re-run even if the run file already exists |

Profiles live in `config/translation_profiles.toml`, overridden by
`~/.config/omniscan/translation_profiles.toml` (a profile there replaces the same-named one here).
One profile is one model plus a prompt style: `chat_json` (many regions per request, glossary in the
prompt, JSON answer) or `translategemma` (one request per region, locked glossary terms
pre-substituted). Interrupted or rate-limited runs keep a dot-prefixed partial file that the next
invocation resumes; a finished run deletes it. Exit 2 for unknown profiles or chapters with no
`ocr.json`-less series; a chapter without `ocr.json` fails that chapter but the rest still run (exit
1). An Ollama rate limit stops everything immediately with exit 3 and keeps the partial results.

```bash
uv run omniscan translate DemoSeries
uv run omniscan translate DemoSeries --profile gemma4-12b-local --chapter "Chapter 1" --force
```

### `omniscan judge`

Turn a chapter's candidate translation runs into one final English line per translatable region,
written to `work_root/<series>/<chapter>/final.json`. Needs `ocr.json` and at least one run under
`translations/` — until the OCR stage lands there is nothing to judge.

| Argument/option | Meaning |
|---|---|
| `series` | required |
| `--chapter`, `-c <str>` | chapter folder name; repeatable. Default: all |
| `--run`, `-r <str>` | candidate run id to judge; repeatable. Default: every run |
| `--force` | re-run even if `final.json` already exists |

Judge settings live in `config/judge.toml`, overridden by `~/.config/omniscan/judge.toml` (a later
file overrides only the keys it sets): the judge model and endpoint, `prefer` (runs to favour when
candidates agree, in order), `chunk_regions` (regions per request), `agree_threshold`,
`always_judge`, `max_repair_rounds` and `temperature`. To save tokens the judge model is asked only
about regions where the candidates disagree or a locked glossary term is broken; everywhere else the
best candidate is picked deterministically, preferring the `prefer` runs. An answer that still
breaks a locked term gets one repair round; whatever the model fails to resolve falls back to the
first clean candidate and is flagged `judge_failed` (plus `glossary_violation` when a locked term is
still missing), so `final.json` is always complete. Chapters without `ocr.json` or without runs fail
that chapter but the rest still run (exit 1). Exit 2 for no chapters, an unknown `--run` or an
invalid judge config; an Ollama rate limit stops everything immediately with exit 3.

```bash
uv run omniscan judge DemoSeries
uv run omniscan judge DemoSeries --chapter "Chapter 1" --run gemma4-31b-cloud --force
```

### `omniscan typeset`

Fit every final English line into its region's target box and record font role, size, wrapped lines
and colour (`layout.json`). Pure planning — nothing is drawn here; a later stage renders and
composites.

| Argument/option | Meaning |
|---|---|
| `series` | series name (required) |
| `--chapter`, `-c <str>` | chapter folder name; repeatable. Default: all |
| `--force` | re-run even if up to date |

Needs `ocr.json`, `final.json` and `inpaint.json` in the chapter work dir; a chapter missing one of
them fails (exit 1). Per region, in reading order: bubble text is fitted into the bubble's inscribed
ellipse box (the original text box wins when it is strictly larger), free text and SFX into their text
box grown by `typeset.free_grow`. The font role follows the region kind (bubble text → dialogue,
free text → free, SFX → sfx) and with it the default font. The colour is the region's `text_color`
when it carries one, else black on light bubble fills and white on dark ones (decided from the
inpaint fill's luminance); free text and SFX are white with an outline
(`typeset.stroke_free_px` / `typeset.stroke_sfx_px`). Watermark regions are never laid out. Writes
`layout.json` + `manifest.json` in the chapter work dir. Exit codes as for `ingest`.

```bash
uv run omniscan typeset DemoSeries
```

### `omniscan eval`

Score finished chapters against ground truth instead of by eye: how many text boxes detection+OCR
found, how many truth characters they cover, how accurately they were read (CER), how faithfully the
raw OCR matches the source-language text (page chrF), and — when `final.json` exists — how close the
English is to the official translation (chrF). The ground truth comes from the `translated-check`
folder (next to the library root): `<translated-check>/<series>/<chapter>/truth/<lang>/EnnPpp.svg`,
the Inkscape text layers of the Pepper&Carrot CC BY 4.0 test data (written by
`scripts/fetch_pepper_carrot.py`). Each `flowRoot` becomes one truth box in strip space (exact box
from its `flowRegion`), and each `<text>` element becomes one too, matched to the OCR regions of
`ocr.json` (watermarks excluded) by which box contains the region's centre, the smallest such box
winning.

`<text>` elements carry no box geometry, so their boxes are **approximate**: each line's box is
estimated from its position, `font-size` and `text-anchor` (a wide CJK glyph counts as one em,
everything else as half). The header line reports how many truth boxes are approximate when there
are any; their boxes are good enough for scoring, not for pixel-exact masks.

| Argument/option | Meaning |
|---|---|
| `series` | series name (required) |
| `--chapter`, `-c <str>` | chapter folder name; repeatable. Default: all |
| `--lang <str>` | truth language folder under `truth/`. Default: `kr` |
| `--json` | print one JSON object per chapter instead of the text block |

Needs `ingest.json` and `ocr.json` in the chapter work dir (missing → exit 1 after the other
chapters; `final.json` is optional). A chapter without `truth/<lang>` fails with exit 2. Writes
`eval.json` into the chapter work dir and prints, per chapter: detection recall (per box), the
character-weighted recall `chars` (share of truth characters in detected boxes — fairer than box
recall when hundreds of tiny SFX boxes dominate), precision, CER macro/micro over the detected
boxes, the page-level OCR chrF (raw OCR text against the source-language truth, one score per page
that has truth boxes), and — with `final.json` — the translation chrF over the scored pages, plus
up to five missed boxes (longest text first) and the five worst-read boxes. A chapter with no
usable truth boxes prints `n/a` instead of the detection and OCR lines. Boxes whose text is
punctuation only are ignored; truth boxes outside their page are dropped and counted.

```bash
uv run omniscan eval PepperCarrotKR -c "Episode 06"
uv run omniscan eval PepperCarrotKR --json
```

### `omniscan watermark add`

Record a fixed-position watermark region for a series (fractions of every raw page).

| Argument/option | Meaning |
|---|---|
| `series` | required |
| `--x0 <0..1>` / `--y0 <0..1>` | left/top edge as a fraction of page width/height |
| `--x1 <0..1>` / `--y1 <0..1>` | right/bottom edge as a fraction of page width/height |
| `--note <str>` | free-text reminder of what the region holds |

Writes `work_root/<series>/watermarks.json`. Exit 2 on invalid coordinates.

```bash
uv run omniscan watermark add DemoSeries --x0 0.0 --y0 0.0 --x1 1.0 --y1 0.05 --note "site banner"
```

### `omniscan watermark list` / `omniscan watermark remove`

| Command | Arguments | Effect |
|---|---|---|
| `watermark list <series>` | — | print the regions as a table |
| `watermark remove <series> <index>` | region index | remove it; remaining indices keep their values. Exit 2 on an unknown index |

Both read/write `work_root/<series>/watermarks.json`.

```bash
uv run omniscan watermark list DemoSeries
uv run omniscan watermark remove DemoSeries 0
```

### `omniscan pack`

Package finished chapters (`output_root/<series>/<chapter>/*.jpg`) into CBZ and/or PDF files.

| Argument/option | Meaning |
|---|---|
| `series` | required |
| `--chapter`, `-c <str>` | chapter folder name; repeatable. Default: all |
| `--format`, `-f <cbz\|pdf>` | cbz or pdf; repeatable. Default: cbz |
| `--out <path>` | output folder. Default: `<output_root>/<series>/_packaged` |

Reads the chapter's output images; writes one archive per chapter and format. Exit 2 when there is
nothing to pack. `omniscan export` is what fills the output folder it packs.

```bash
uv run omniscan pack DemoSeries
uv run omniscan pack DemoSeries --chapter "Chapter 1" --format cbz --format pdf
```

### `omniscan serve`

Run the web debug tool's API (pair with `npm run dev` in `webui/` for the UI).

| Option | Meaning |
|---|---|
| `--host <str>` | bind address. Default: 127.0.0.1 |
| `--port <int>` | port. Default: 8000 |
| `--reload` | auto-reload on code changes |

Serves existing artifacts read-only: `/api/series`, `/api/series/{s}/chapters`,
`/api/series/{s}/chapters/{c}/ingest`, `/api/series/{s}/chapters/{c}/slices`,
`/api/series/{s}/chapters/{c}/ocr` (the OCR stage's `ocr.json`),
`/api/series/{s}/chapters/{c}/translations` (run ids) and `/api/series/{s}/chapters/{c}/translations/{run_id}`
(each translation run's JSON), `/api/series/{s}/chapters/{c}/final` (the judge's `final.json`),
`/api/series/{s}/glossary` (the series glossary),
`/api/series/{s}/chapters/{c}/glossary-hits` (glossary hits per OCR region),
`/api/series/{s}/chapters/{c}/pages/{index}` (raw page bytes),
`/api/series/{s}/chapters/{c}/output` (finished output image names) and
`/api/series/{s}/chapters/{c}/output/{name}` (finished output image bytes).

The one write endpoint is `/api/series/{s}/chapters/{c}/filter/restore` (POST): it appends a
`restored` decision to the chapter's `filter.json`, exactly what `omniscan filter restore` does —
metadata only, it never touches files under `_filtered/`. There is also
`/api/series/{s}/filtered`, the read-only listing behind the Filtered view. Everything else is
GET-only and never writes.

```bash
uv run omniscan serve
```

### `omniscan models`

Manage the model catalog (`config/models.toml`): every model OmniScan can download, with its
purpose, size and licence — the three Hugging Face vision models and the LaMa file (mirrored
unchanged as assets of this repo's GitHub release `models-v1`, with automatic fallback to the
upstream source when the mirror is unreachable), the full PP-OCR v5/v6 / PaddleOCR-VL / manga-ocr
OCR catalog (downloaded straight from Hugging Face at a pinned commit, per-file sha256-verified),
and the Ollama LLMs. Weights land in `paths.models_dir` (`<models_dir>/<id>/` for the zips and hf
models, `<models_dir>/lama/big-lama.pt` for LaMa); every download is sha256-verified. Each entry
also carries its `role` (detector / text_line_detector / recognizer / vlm_ocr / inpaint / llm) and
`langs` tags — see [docs/MODELS.md](MODELS.md) for the per-model notices.

| Subcommand | Effect |
|---|---|
| `list [--json] [--role R] [--lang C]` | print one row per model (`id`, `kind`, `role`, `size`, `required`/`optional`, `status`, `langs`, `description`) with a last column `fit` — how the model would run on this machine (`ok`/`slow`/`warn`/`incompatible`, see below) — plus a footer with the total size of the missing required models. `--role` filters to one role (`recognizer`, `text_line_detector`, …), `--lang` to models tagged with a language code (`ko`, `ja`, `th`, …). `--json` emits a machine-readable object (the stable interface a future settings screen will use) that carries a `compatibility` object per model (`level`, `device`, `messages`), the `hardware` snapshot of `omniscan hardware`, and additionally `family`, `size_class`, `langs`, `recommended_for` and `notes`; the Ollama statuses show `unknown` when the daemon is unreachable |
| `download <id>... [--required] [--force]` | download the named models; `--required` also downloads every required model that is not installed. A non-`ok` model prints its compatibility messages as a warning first; an `incompatible` model is refused (exit 1, nothing downloaded) unless `--force` is given. Progress is printed at most once per 5 % step; a failing model is reported on stderr and the others are still tried (exit 1). Unknown ids exit 2 |
| `remove <id>...` | delete installed models (their folder, the LaMa file, or the Ollama daemon's copy). Prints `removed` or `nothing to remove` per model; unknown ids exit 2 |
| `verify [<id>...] [--deep]` | recompute the statuses (missing/installed/corrupt/cloud/unknown); default: every non-llm model. `--deep` additionally re-hashes every installed file (zip archives and per-file for the hf models) instead of comparing sizes. Exit 1 when one is `corrupt` |

Statuses: `installed` (present and verified), `missing`, `corrupt` (size or sha256 mismatch — delete
and re-download), `cloud` (Ollama Cloud model, nothing on disk), `unknown` (Ollama unreachable). For
the hf models the installed folder also carries a `.installed.json` marker with the pinned revision
and the verified per-file sizes; changing the catalog pin marks the copy `corrupt` until it is
re-downloaded.

The `fit` column is the per-model compatibility against the hardware snapshot (`omniscan hardware`):
`ok` (a GPU with enough VRAM, or fast on the CPU), `slow` (runs, but noticeably slower than on a
GPU), `warn` (CPU only and slow), `incompatible` (no supported backend, or the RAM/VRAM it needs is
not there). The reasons are the `messages` in the JSON and the indented lines under the table; the
requirements themselves are the requirement fields in `config/models.toml`
(see [docs/MODELS.md](MODELS.md)).

Statuses: `installed` (present and verified), `missing`, `corrupt` (size or sha256 mismatch — delete
and re-download), `cloud` (Ollama Cloud model, nothing on disk), `unknown` (Ollama unreachable).

How the pipeline finds models: the `detect` and `ocr` stages load each model from
`<models_dir>/<id>/` when its catalog entry is installed (`from_pretrained` with
`local_files_only=True`, so an installed app also works offline); otherwise they fall back to the
Hugging Face hub/cache with the config's pinned revision and log a warning that `omniscan models
download --required` would install it. LaMa always loads `<models_dir>/lama/big-lama.pt`, the path
its download already uses.

```bash
uv run omniscan models list
uv run omniscan models list --role recognizer
uv run omniscan models list --lang ko
uv run omniscan models list --json
uv run omniscan models download --required
uv run omniscan models download ocr-rec-ppocrv6-tiny llm-gemma4-12b
uv run omniscan models remove llm-gemma4-12b
uv run omniscan models verify --deep
```

### `omniscan queue`

A persistent job queue for running pipeline stages over series without babysitting one command per
series. You add jobs, then a worker drains them one at a time. Everything lives in one SQLite file,
`work_root/queue.db`; jobs survive restarts, and a worker that died mid-job recovers automatically
(the interrupted job is queued again with its attempt refunded). Exactly one worker per `queue.db` is
supported.

A job is "run these stages over this series (optionally only some chapters)". Any of the ten pipeline
stage names is accepted (`ingest`, `slice`, `detect`, `ocr`, `translate`, `judge`, `inpaint`,
`inpaint_lama`, `typeset`, `export`); the stages run exactly as listed, in pipeline order and passes
(like `omniscan run`), so a fresh chapter needs `ingest` listed before `slice`.

| Subcommand | Effect |
|---|---|
| `add <series>` | queue a job. Options: `--stage`, `-s <name>` (repeatable, default `ingest` + `slice`; any of the ten stage names, run exactly as listed), `--chapter`, `-c <name>` (repeatable, default all), `--priority`, `-p <int>` (higher runs first, default 0), `--max-attempts <int>` (default 2), `--force` |
| `list [--status <status>]` | print the jobs, one line each; filter by `queued` / `running` / `paused` / `done` / `failed` / `cancelled` |
| `run [--webhook <url>] [--max-jobs <n>]` | drain the queue as the worker. `--webhook` POSTs every event as JSON (also settable with the `OMNISCAN_NOTIFY_WEBHOOK` env var); `--max-jobs` stops after n jobs |
| `pause <id>` / `resume <id>` | pause a queued job / put a paused job back into the queue |
| `cancel <id>` | cancel a queued or paused job (a running job cannot be cancelled) |
| `retry <id>` | reset a failed or cancelled job to queued with a fresh attempt counter |
| `clear` | delete done and cancelled jobs (failed jobs stay so you can retry them) |

The `run` worker prints one line per executed job (`job 1 done`, `job 1 failed: <error>`) and a
summary, exits 1 if any job failed, and prints `queue is empty` when there was nothing to do. Failures
after a retryable error are retried up to `--max-attempts` times automatically; unknown stage names and
bad job ids exit 2.

```bash
uv run omniscan queue add DemoSeries
uv run omniscan queue add DemoSeries -s ingest -s slice -p 5 -c "Chapter 1"
uv run omniscan queue list
uv run omniscan queue run
```

```text
queued job 1: DemoSeries stages=slice chapters=all priority=0
queued job 2: DemoSeries stages=ingest,slice chapters=Chapter 1 priority=5
   1  queued    pri=0 try=0/2  DemoSeries  slice  all
   2  queued    pri=5 try=0/2  DemoSeries  ingest,slice  Chapter 1
job 2 done
job 1 done
done=2 failed=0 retried=0
```

### `omniscan update`

OmniScan can update itself from the project's GitHub Releases: `check` asks GitHub which app release
is newest (releases that are not app versions — like the `models-v1` model mirror — are ignored), and
`download` fetches the build for this platform and verifies it against the release's `SHA256SUMS`
before staging it. Nothing replaces the running program yet; the staged folder is what a
platform-specific installer will consume.

| Subcommand | Effect |
|---|---|
| `check [--channel stable\|beta] [--repo OWNER/NAME] [--json]` | print `update available: vA.B.C (current vX.Y.Z)` plus the release notes (first 20 lines), or `up to date (vX.Y.Z)`; `--json` prints the same facts as JSON (`current`, `latest`, `update_available`, `tag`, `notes`, `asset`) |
| `download [--channel ...] [--repo ...]` | check, then download and verify into `updates/` next to the work root (`~/omniscan/updates` by default), under a folder per release tag, printing progress every 10 % and the staged path; prints `already up to date` (exit 0) when nothing is newer |

Errors answer on stderr and exit 1. A build is only staged when its SHA-256 matches `SHA256SUMS`; a
staged file with the matching checksum is reused. Set `GITHUB_TOKEN` in the environment to
authenticate GitHub requests (needed while the repository is private); the token is never printed.

```bash
uv run omniscan update check
uv run omniscan update check --channel beta --json
uv run omniscan update download
```

```text
update: update available: v1.2.3 (current v0.1.0)
update:   Bugfixes for the slicer and a faster OCR pass.
update: staged ~/omniscan/updates/v1.2.3/omniscan-windows-x64.zip (9c1b…)
```

## Web viewer

Start the API and the UI in two terminals:

```bash
uv run omniscan serve
```

```bash
cd webui
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://localhost:8000`, so the defaults of both commands work
together. Open the local URL Vite prints and pick a series and chapter.

Four chapter views (Slicer, OCR, Translation, Reader) need a chapter; the **Filtered** view only needs
a series and shows its every chapter that has filtered items.

The Slicer view stacks the raw pages and overlays:

- the detected **bands** as translucent green rectangles,
- the **cut lines** between slices (navy; a **forced cut** is red),
- the **original raw-file boundaries** as dashed gray lines.

The **OCR view** (switch with the `OCR` button above the picker) draws every OCR region from the
chapter's `ocr.json` on the same stacked pages. Boxes are colored by kind (bubble text blue, free text
orange, SFX purple, watermark gray); the label is `reading order · confidence`. A **dashed** outline marks a
low-confidence region (confidence below 0.5) and a **yellow** outer box plus a `≠` marks a region where the
second-opinion reading (`ocr_alt`) differs from the main text. Checkboxes hide/show kinds, and clicking a
region opens a side panel with its full detail (id, language, orientation, per-line engine/score/text).
This view only has data once a chapter has an `ocr.json` from `omniscan ocr`; until then it shows
`no ocr.json yet`.

The **Translation view** (switch with the `Translation` button) is one review table row per OCR region in
reading order: the source text with glossary terms highlighted, every candidate translation run
(`translations/<run_id>.json`) side by side, and the judge's final line (`final.json`) with its decision
badge, per-flag chips and the rationale behind a `why` disclosure. The glossary column lists the terms
matched in the region's source text; a **locked** term whose English target is missing from the final line
is shown with `✗` and gives the row a red background, while rows with final flags, runs disagreeing with
each other, or regions no run produced text for get an amber one. A checkbox filters the table down to just
the rows needing attention. All inputs beyond `ocr.json` are optional — a missing `final.json`, run file or
`series.db` simply shows less.

The **Reader view** (switch with the `Reader` button) shows a chapter's finished English output images
(`output_root/<series>/<chapter>/`) as one continuous vertical strip, the way a reader would see the
released chapter. `final only` shows the output images alone; `raw | final` puts them side by side with
the chapter's raw pages (kept files only) with linked scrolling for comparison. A width slider sets the
reading column width, fitted to the viewport. The view shows `no output yet` until `omniscan export`
has written the chapter's slices.

The **Filtered view** (switch with the `Filtered` button, shown as soon as a series is picked) lists
everything the promo filter ever marked `filtered`, per chapter: one row per filtered file or slice
with its score, matched example and method, a thumbnail for file-level items (the raw page is still in
the library), a `filtered` (red) or `restored` (green) badge, and a **Restore** button on items still
filtered. Restore appends a `restored` decision to the chapter's `filter.json` — the same thing
`omniscan filter restore` does; files already copied into `_filtered/` are never touched, and a
restored item can be re-filtered only by re-running `omniscan filter run`. The row turns green without
a refetch when the restore succeeds; a failure shows the API's error next to the button.

The whole tool is read-only except the Filtered view's Restore button: the API only answers GET
requests plus that one POST, which appends to `filter.json` and never writes anything else.

## Resuming and re-running

`ingest`, `slice`, `detect` and `ocr` are stage runs recorded in the chapter's `manifest.json`: stage
version, content hash of the inputs, hash of the relevant config subset, status and outputs. A second
run with unchanged inputs and config is a no-op — it prints `skipped (0.00s)` per stage and exits 0.
Use `--force` to re-run even when up to date. `slice` always runs `ingest` first when the ingest is
missing or stale, so `slice` alone is enough for a fresh chapter (`detect` pulls in both the same
way, and `ocr` pulls in all three).

## Troubleshooting

- **`doctor` rows.** `FAIL` means the machine is not ready — the command exits 1 and the pipeline
  would not run (e.g. torch is not a ROCm build, or the local Ollama is unreachable). `WARN` is
  informational and does not affect the exit code: the `rocjpeg` WARN is expected (the hardware JPEG
  decoder is not usable in WSL — no `/dev/dri` — and is not present on Windows, so the CPU `turbo`
  codec runs), as are the secrets/paths "not set / will be created on first use" rows. The `rocm`
  row (`rocminfo`) only applies to Linux/WSL; on Windows it reports "not applicable".
- **`torch_gpu` FAIL: "no usable discrete GPU".** `gpu.device = "auto"` skips integrated GPUs (on
  Windows they are listed first and crash on the first kernel). Name a device explicitly
  (`gpu.device = "cuda:1"`) only if `auto` picks the wrong one.
- **rocJPEG probe (Linux/WSL).** `uv run python scripts/rocjpeg_probe.py` exits 0 only when the
  hardware JPEG decoder is usable; in WSL2 it fails today. Re-run it after every AMD driver / ROCm
  update — a driver that exposes VCN would let the real decoder replace the CPU codec.
- **`ollama_local` FAIL (unreachable).** Start Ollama (it listens on `localhost:11434`). When
  running inside WSL2 instead of Windows, WSL reaches the Windows Ollama only with mirrored
  networking (`networkingMode=mirrored` in `~/.wslconfig`, then `wsl --shutdown`).
- **`ollama_models` WARN (missing: …).** The named models are not pulled yet; install them in the
  Windows-side Ollama.
- **VA-API noise before the `doctor` table.** Lines like `InitVAAPI(drm_node) returned …` are the
  rocJPEG probe failing inside WSL — expected, and already reported as the `rocjpeg` WARN row.
- **A NumPy "array is not writable" warning during `slice`.** Harmless; the tensor is read-only by
  design.
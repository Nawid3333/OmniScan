# OmniScan user guide

This guide covers everything OmniScan can do today. Commands and flags are checked against the CLI by
`tests/unit/test_docs.py`, so this document cannot silently drift from the program.

## Folder layout

OmniScan keeps three roots (see `[paths]` in [Configuration](#configuration)):

| Path | Written by | Contents |
|---|---|---|
| `library_root/<Series>/<Chapter N>/` | `omniscan import` | raw chapter images (read-only for the pipeline) |
| `library_root/<Series>/_reference_en/` | `omniscan import --series "<Series>/_reference_en"` | official English chapters `omniscan reference` learns the glossary from |
| `work_root/<Series>/series.db` | `omniscan reference`, `omniscan glossary import` | SQLite glossary working copy |
| `work_root/queue.db` | the job queue | every queued/running/finished job |
| `work_root/<Series>/glossary.yaml` | `omniscan glossary export`, `omniscan reference` | hand-editable glossary review file |
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
| `work_root/<Series>/<Chapter N>/filter.json` | `omniscan filter restore` / `force` / web Restore | manual promo-filter overrides (read as an input by `ingest` and `slice`) |
| `work_root/<Series>/<Chapter N>/manifest.json` | the stage runner | per-stage status, inputs and config hashes |
| `work_root/<Series>/<Chapter N>/converted/` | `omniscan ingest` | JPEG cache for raws that were not JPEG |
| `output_root/<Series>/<Chapter N>/` | `omniscan export` | final English slices (`0001.jpg` …) |
| `output_root/<Series>/_filtered/<Chapter N>/` | `omniscan filter run` (ingest/slice) | byte-copies of promo-filtered raw files and filtered slices (never deleted) |
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
| `inpaint.glyph_mask` | remove only the lettering's own pixels (+ outline, anti-aliasing) instead of whole line boxes | yes |
| `inpaint.glyph_grow`, `inpaint.glyph_grow_sfx` | the letters are grown by at most this many stroke widths (text / sound effects) | yes |
| `inpaint.glyph_grow_min_px`, `inpaint.glyph_grow_max_px`, `inpaint.glyph_grow_max_sfx_px` | limits of that growth, in px | yes |
| `inpaint.glyph_ring_px` | band around the glyphs that must be flat for a flat fill of just the glyphs, in px | yes |
| `inpaint.remove_watermarks` | erase watermarks (ad/site text, stored fixed-position zones) from the page; they are never lettered either way | yes |
| `typeset.min_px` | smallest font size tried, in px | yes (`omniscan typeset`) |
| `typeset.max_px` | largest dialogue size, in px | yes (`omniscan typeset`) |
| `typeset.line_spacing` | line pitch as a multiple of the font size | yes (`omniscan typeset`) |
| `typeset.margin_px` | smallest gap between lettering and the balloon's edge, in px | yes (`omniscan typeset`) |
| `typeset.bubble_padding` | share of a balloon's width / height kept free on each side | yes (`omniscan typeset`) |
| `typeset.free_grow` | free text / SFX boxes are grown by this fraction on every side | yes (`omniscan typeset`) |
| `typeset.stroke_free_px` | outline width of free-standing text, in px | yes (`omniscan typeset`) |
| `typeset.stroke_sfx_px` | thinnest outline of sound effects, in px (large effects scale it up) | yes (`omniscan typeset`) |
| `typeset.style` | lettering preset: `auto` (manga for Japanese sources, webtoon otherwise), `webtoon`, `manga` | yes (`omniscan typeset`) |
| `typeset.uppercase` | letter dialogue in capitals: `auto` (the style's convention), `always`, `never` | yes (`omniscan typeset`) |
| `typeset.font_dialogue`, `font_thought`, `font_shout`, `font_narration`, `font_free`, `font_sfx` | your own font for a role: a file in the fonts folder or an absolute path (empty = the preset) | yes (`omniscan typeset`) |
| `typeset.size_spread`, `typeset.shout_spread` | dialogue / shouts at most this × the chapter's typical balloon size (0 = no limit) | yes (`omniscan typeset`) |
| `typeset.hyphenate` | hyphenate a word that fits no line even at `min_px` instead of overflowing | yes (`omniscan typeset`) |
| `typeset.sfx_max_px` | largest sound-effect lettering, in px | yes (`omniscan typeset`) |
| `sfx.detect` | turn free text that reads as onomatopoeia (`config/sfx_text.toml`) into sound effects | yes (`omniscan ocr`) |
| `sfx.sweep` | also look over the whole page (CRAFT) for the effects the detector missed; needs `sfx.detect` | yes (`omniscan ocr`) |
| `sfx.sweep_min_px` | swept lettering with smaller letters is left alone (signs, background text), in px | yes (`omniscan ocr`) |
| `sfx.max_chars`, `sfx.lexicon_size_ratio` | longest effect (letters) and how large vs the dialogue a lexicon match must be lettered | yes (`omniscan ocr`) |
| `sfx.size_ratio`, `sfx.size_max_chars` | `size_ratio` > 0: very short lettering this much larger than the dialogue is an effect too (off by default: big signs pass) | yes (`omniscan ocr`) |
| `sfx.mode` | `replace` (erase and redraw in English), `subtitle` (keep the art, small translation below), `keep` | yes (`omniscan inpaint`, `typeset`) |
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

Secrets never go in TOML. Put them in `~/.config/omniscan/secrets.env` (or export it); it is
optional today and `omniscan doctor` reports when it is missing. Never commit this file.

```text
OLLAMA_API_KEY=...            # Ollama cloud (`translate` cloud profiles; doctor reports if unset)
```

## Commands

Every command below is fully working. The examples assume you generated the demo chapter with
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

Import raw chapter images from a local folder or a `.zip`/`.cbz` archive into the library. The
source can be one chapter folder (all images), a folder of chapter folders (each child names a
chapter), or a flat dump of images whose filenames carry an explicit chapter marker (`Ch1`,
`Chapter_02`, `ep3`, …). Archives are extracted to a temporary directory and planned with the same
three shapes; a lone top-level wrapper folder is descended into, and the series name defaults to
the archive's own name. Series and chapter are guessed from the folder names unless you override
them.

| Argument/option | Meaning |
|---|---|
| `source` | folder, or `.zip`/`.cbz` archive (required) |
| `--series <str>` | override the destination series name |
| `--chapter <str>` | override the destination chapter name |
| `--move` | move instead of copy; delete the source after import |
| `--dry-run` | print the plan without writing anything |

Files whose name does not end in `.jpg`/`.jpeg` are converted to plain baseline JPEG (quality 95)
on the way in — alpha is flattened over white and EXIF rotation is applied, exactly like the ingest
stage, so the pipeline never re-converts them. `--dry-run` reports how many that is
(`import:   N file(s) will be converted to JPEG`) and the commit summary adds
`, N file(s) converted to JPEG` when it is more than zero. The conversion runs on the CPU
(libjpeg-turbo); no hardware JPEG decoder is wired up.

Reads the source; writes `library_root/<series>/<chapter>/`. Exit 2 when the source cannot be
planned unambiguously (empty folder or archive, mixed subfolders and loose images, no parseable
chapter names, a corrupt or encrypted archive) or when a destination file exists with different
content.

```bash
uv run omniscan import ~/Downloads/DemoSeries --dry-run
uv run omniscan import ~/Downloads/DemoSeries.zip
```

The standalone GUI import page (`ImportView`, demoed by `scripts/gui_import_demo.py`) wraps the
same pipeline with a plan preview you can fix before committing: rename the series, move or
reorder pages, rename/merge/split chapters, see exactly which files will be converted, then
import with copy or move while a progress bar tracks it.

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

### `omniscan slice-compare`

Run every slicer strategy over one chapter's strip and print where each would cut, so you can pick
the strategy for a series ([Slicer strategies](#slicer-strategies)). The strip is built exactly as
`omniscan slice` builds it (ingest runs first when its artifact is missing); nothing is written to
the work directory.

| Argument/option | Meaning |
|---|---|
| `series` | series name (required) |
| `--chapter`, `-c <str>` | chapter folder name. Default: the first chapter |
| `--json` | emit the per-strategy summaries as one JSON object |

Prints one row per strategy (`strategy slices min median max forced blank`). With `--json` it
prints `{"series", "chapter", "strip_height", "strategies": [...]}` with one summary object per
strategy. An unknown series or chapter prints a message on stderr and exits 1.

```bash
uv run omniscan slice-compare DemoSeries
uv run omniscan slice-compare DemoSeries -c "Chapter 1" --json
```

### Slicer strategies

Four ways to cut a chapter strip, selected per series with `strategy` in the
`library_root/<series>/series.toml` `[slicer]` section (default `smart`). All of them produce the
same `slices.json` artifact (tiling the strip exactly, with blank and forced-cut flags):

| Strategy | Behaviour |
|---|---|
| `smart` | detect uniform gutter bands (`slicer.uniform_tol`, `band_min_px`, `max_drift`), then plan cuts between `min_height` / `target_height` / `max_height` by dynamic programming; forced cuts land on the least-detailed row |
| `page` | one slice per raw page, boundaries exactly at the file edges; falls back to `smart` (recorded as `params.fallback`) when the ingest has no page layout |
| `fixed` | cut every `slicer.target_height` rows; a tail shorter than `slicer.min_height` merges into the previous slice |
| `simple_gutter` | cut at the centre of each gutter run at least `slicer.gutter_min_rows` rows tall (`slicer.gutter_variance` decides what a gutter row is), the run nearest `target_height` inside every `[min_height, max_height]` window; a forced `target_height` cut when no gutter qualifies |

`omniscan slice-compare` shows what each would do to one chapter before you commit the series to
one of them in `series.toml`.

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

### OCR engines and models

`ocr.engine` chooses how the regions are read. Every model comes from the catalog
(`config/models.toml`, listed by `omniscan models list`) and is installed with
`omniscan models download <id>`; an engine's model that is not installed is pulled from its pinned
Hugging Face revision on first use.

| Engine | Reads | Models (`[ocr]` in `series.toml` or `OMNISCAN_OCR__*`) | Status |
|---|---|---|---|
| `ppocr` (default) | detected text lines, as described above | `ocr.det_model` + `ocr.rec_model`, catalog ids (PP-OCRv5/v6, any size; unset = `ocr.det_repo`/`ocr.rec_repo`) | available |
| `manga_ocr` | every region as one whole crop | `ocr.rec_model`, catalog id (default `ocr-rec-manga-ocr-2025`, Japanese) | available |
| `paddleocr_vl` | every region as one whole crop | `ocr.rec_model`, catalog id (default `ocr-vl-1.6`) — accuracy mode: reads all CJK scripts and Latin, incl. stylised lettering, but is slow (~3.6 s per region on the reference GPU) and big (3.4 GiB fp32; the vision group is budgeted at 6.6 GiB) | available |

With `ppocr`, `ocr.det_model` / `ocr.rec_model` name catalog entries and win over the plain
`ocr.det_repo` / `ocr.rec_repo` (a wrong role — a recogniser id as `det_model` — is rejected); each
line records the model id as its `engine`. With a crop-reading engine no line detection runs: every
region is padded by `ocr.crop_pad_px` and read as one crop (`manga_ocr` in batches of
`ocr.crop_batch_size`; `paddleocr_vl` one crop at a time, so `crop_batch_size` is ignored), and the
region's text is that one reading (the score is the model's confidence). `paddleocr_vl` also honours
`ocr.vl_max_new_tokens`, the longest text it may generate per region. Changing `ocr.engine`,
`ocr.det_model` or `ocr.rec_model` re-runs the `ocr` stage.

```bash
OMNISCAN_OCR__ENGINE=manga_ocr uv run omniscan ocr DemoSeries
OMNISCAN_OCR__ENGINE=paddleocr_vl uv run omniscan ocr DemoSeries
OMNISCAN_OCR__DET_MODEL=ocr-det-ppocrv6-medium OMNISCAN_OCR__REC_MODEL=ocr-rec-ppocrv6-medium uv run omniscan ocr DemoSeries
```

Which engine and model is best per language is measured, not assumed — `recommended_for` in the
catalog and the qualification suite (card O1c) are the source of truth.

### `omniscan inpaint`

Remove the source text from every OCR region and record the result (`inpaint.json` + `patches.npz`).
Where the surroundings of a region's line boxes are one flat colour (a white or dark bubble interior)
the boxes are filled with exactly that colour. Otherwise only the lettering itself is removed
(`inpaint.glyph_mask`): its ink is found from the pixels, together with whatever the ink encloses and
the outline around the letters, and when the band around those glyphs is flat they alone are filled.
The rest — lettering on textured art — is recorded with a `needs_lama` flag and its glyph mask, so
LaMa only rebuilds the letters, not the art around them. Sound effects are erased only with
`sfx.mode = "replace"` (the default). Watermarks are erased too (`inpaint.remove_watermarks`, on by
default): ad/site text like any lettering, a stored fixed-position zone as a whole box (flat-filled on a
plain margin, rebuilt by LaMa over art).

With `--lama` a second stage (`inpaint_lama`) cleans exactly those regions with the LaMa model
(TorchScript `big-lama.pt`, Apache-2.0): for every such region a fixed 512×512 window of the strip
around it is inpainted on the GPU in fp32 and stored as a patch in `patches_lama.npz` (+
`inpaint_lama.json`); export applies `patches.npz` first and `patches_lama.npz` after it. The weights
(205 MB) are downloaded from `inpaint.lama_url` into `paths.models_dir/lama/` on first use and their
sha256 is checked; regions wider or taller than
`inpaint.lama_window - 2 * inpaint.lama_context_px` are tiled. No lettering serves as context: each
window masks every region still to be cleaned that reaches into it, and finished regions (flat or LaMa)
are written back before the next one runs. The pixels themselves stay out of
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
| `--step` | step-by-step mode: preview one chapter after every stage and decide before continuing |
| `--preview-chapter <name>` | which chapter step mode previews. Default: the first chapter. Needs `--step` |

The text pass talks to Ollama (local or cloud models); the vision and LaMa models are loaded through
the VRAM manager, which frees the previous group first, so the three passes never fight over 16 GB.
One line is printed per stage outcome, then a summary line. Exit 2 for an unknown series or stage
name (or `--preview-chapter` without `--step`); exit 1 when any chapter failed — including a failed
preview chapter in step mode (nothing else runs then); exit 3 on an Ollama rate limit (partial
results are kept — re-run later); exit 4 when you answered `n` at a step-mode prompt.

```bash
uv run omniscan run DemoSeries
uv run omniscan run DemoSeries -c "Chapter 1" -s ingest -s slice
uv run omniscan run DemoSeries --no-lama --force
```

With `--step` the pipeline spends the preview effort on one chapter before hours of GPU and LLM time
go into a whole series: the preview chapter first runs through all three passes, pausing after every
stage. Each pause prints a `Preview` block — the stage's position in the run, its status, and what it
produced for that one chapter (file and slice counts, detected regions, the OCR text, translation
candidates, final lines, exported files — a missing artifact is shown as such) plus the error when
the stage failed — and then asks `Continue? [y]es / [n]o stop / [a]ll (finish without asking)`:
`y` (or just enter) runs the next stage, `n` stops the run right there (exit 4), and `a` answers
every remaining question silently. Once the preview chapter has been through every stage, the
remaining chapters run automatically without prompts. Up-to-date (skipped) stages still pause.

```bash
uv run omniscan run DemoSeries --step
uv run omniscan run DemoSeries --step --preview-chapter "Chapter 3" --no-lama
```

```text
Preview [5/10] Chapter 3 ocr: done
    12 region(s) with text, 2 low-confidence (< 0.85)
    "그럼 우리는 어떻게…"
    "당신이 정말로 그럴 생각이…"
Continue? [y]es / [n]o stop / [a]ll (finish without asking) [y]:
```

### `omniscan filter run`

Run the promo filter over a series: ingest and slice re-run for the chosen chapters and everything
matching a promo example is filtered out of the strip.

| Argument/option | Meaning |
|---|---|
| `series` | required |
| `-c/--chapter <name>` | chapter folder; repeatable. Default: every chapter |
| `--json` | print one JSON object (`chapters`, `totals`) instead of text lines |

Every raw file whose dHash is within `filter.threshold` (default 0.9, `[filter]` in the config) of a
`promo_examples/global` or `promo_examples/<Series>` image is left out of the strip at ingest time,
and every slice whose dHash matches is flagged after slicing. Filtered files are byte-copied into
`output_root/<series>/_filtered/<chapter>/` (nothing is deleted) and recorded in `ingest.json` /
`slices.json`, which the printed counts summarise. With no examples configured everything is kept,
and a chapter whose every image matches fails with an error.

```bash
uv run omniscan filter run DemoSeries
uv run omniscan filter run DemoSeries --chapter "Chapter 1" --json
```

### `omniscan filter restore`

Restore a filtered file or slice (metadata override; nothing is deleted).

| Argument | Meaning |
|---|---|
| `series`, `chapter` | required |
| `target` | `file` or `slice` |
| `index` | `file` index = the raw file's position in the chapter folder (its `SourceFile.index`); `slice` index = the slice's index in `slices.json` |

Appends a manual `restored` entry to the chapter's `filter.json` (creating the file when absent) and
prints that the decision applies on the next run: the next `omniscan filter run`, `ingest` or `slice`
keeps the restored file or slice no matter what the examples say.

```bash
uv run omniscan filter restore DemoSeries "Chapter 1" slice 0
```

### `omniscan filter force`

Force-filter a file or slice, even when no example matches it.

| Argument | Meaning |
|---|---|
| `series`, `chapter`, `target`, `index` | as for `filter restore` |

Appends a manual `filtered` entry to `filter.json`; it applies on the next run even when
`filter.enabled` is false or the examples folder is empty.

```bash
uv run omniscan filter force DemoSeries "Chapter 1" file 2
```

### `omniscan filter add`

Copy an image into the promo-examples folder for `filter run` to match against.

| Argument/option | Meaning |
|---|---|
| `series`, `path` | required; `path` is a JPEG/PNG file on disk |
| `--global` | add to the shared `global` examples instead of this series' folder |
| `--name <name>` | destination file name. Default: the source's name |

Refuses to overwrite an existing example and rejects anything that is not a readable JPEG/PNG.

```bash
uv run omniscan filter add DemoSeries ~/Downloads/end_card.jpg --name end_card.jpg
```

### Watermark text patterns

Aggregator-injected ads are often plain text on the page rather than a repeated banner, so the promo
filter above cannot catch them. The OCR stage therefore reclassifies a detected region as a
`watermark` when its read text contains one of the patterns in `config/watermark_text.toml`
(plus your own `~/.config/omniscan/watermark_text.toml`; both files' lists are combined, edits take
effect on the next OCR run). Watermarked regions are excluded from translation, scoring and
evaluation like the fixed-position ones, and the inpaint stage erases them from the page
(`inpaint.remove_watermarks = false` keeps them). The shipped list also catches stamped web addresses
("http", "www."); the sound-effect sweep (below) checks the text it finds in the art against the same
patterns, so a site stamp the detector missed is erased as well. To teach the filter a new site, append
its specific brand or site name to the user file (a pattern is matched as a case-insensitive substring
of the OCR'd text):

```toml
[watermark_text]
patterns = [
    "구글검색",
    "먹튀위키",
]
```

### `omniscan glossary list`

Print the glossary of a series as a table (source, target, type, status, count).

| Argument/option | Meaning |
|---|---|
| `series` | required |
| `--status <proposed\|locked\|rejected>` | only entries with this status |

Requires `work_root/<series>/series.db` (exit 2 with a message otherwise) — `omniscan reference` is
what creates it.

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

### `omniscan glossary propose`

Propose glossary terms from a series' own OCR text when no official English release exists to learn
from (there is no `reference` mode without one). The chat model reads each chapter's Korean lines and
returns (source, target) pairs with its own best English rendering; a term that recurs in at least
`--min-chapters` distinct chapters is written to the glossary **as `proposed`, `origin="llm"` — never
auto-locked**, because there is no official translation to agree with. Review them with
`omniscan glossary list --status proposed` and edit them by hand (or via `glossary.yaml`) as usual.

| Argument/option | Meaning |
|---|---|
| `series` | series name (required) |
| `--chapter`, `-c <str>` | chapter folder name; repeatable. Default: all |
| `--model <name>` | chat model used for term proposal. Default: `gemma4:31b-cloud` |
| `--min-chapters <int>` | distinct chapters a term must recur in to be written at all. Default: 2 |
| `--dry-run` | scan and extract, but write nothing to the glossary |
| `--json` | emit one JSON object instead of text lines |

Like `reference`, this never silently overwrites entries: a proposal that disagrees with a `locked`
or `rejected` entry is flagged as a conflict and the entry is left untouched, and a row an earlier
machine pass proposed is updated in place. This command is CPU-only (one chat request per chapter);
it does not touch the GPU.

```bash
uv run omniscan glossary propose DemoSeries
uv run omniscan glossary propose DemoSeries --dry-run --min-chapters 1
```

### `omniscan reference`

Bootstrap a series' glossary from its official English release (decision D7): match the raw chapters
against the already-imported reference chapters, OCR both sides, and extract the (source, target)
term pairs the official translation actually uses. A pair that recurs identically in at least three
distinct reference chapters is written locked (`status="locked"`, `origin="reference"`); everything
else is written `proposed` for human review — nothing is dropped.

The reference chapters live under `library_root/<Series>/_reference_en/` (one folder per chapter,
anything `omniscan import` accepts). Import them with the pseudo series name — its `_`-prefix keeps
them out of the raw chapter list:

```bash
uv run omniscan import ~/Downloads/OfficialEN.zip --series "DemoSeries/_reference_en"
uv run omniscan reference DemoSeries
```

| Argument/option | Meaning |
|---|---|
| `series` | series name (required) |
| `--min-locks <int>` | distinct reference chapters a (source, target) pair must recur in to be locked. Default: 3 |
| `--model <name>` | chat model used for term extraction. Default: `gemma4:31b-cloud` |
| `--dry-run` | match, OCR and extract, but write nothing to the glossary |
| `--force` | re-run the OCR stages even if up to date |

Chapters are matched by page art (the `omniscan match chapters` matcher — folder names never decide),
so chapter numbering and folder names may differ between the two sides. Matched pairs run the
`ingest`/`slice`/`detect`/`ocr` stages on both sides (the same stages, artifacts and manifests the
pipeline uses — the reference side's land under `work_root/<Series>/_reference_en/`, so a re-run
skips what is up to date), the OCR'd regions are paired page by page in reading order, and one chat
request per matched chapter asks for its terms. The merge lands in `work_root/<series>/series.db`
and is re-exported to `glossary.yaml`.

Existing entries are never silently overwritten. Agreement with a locked entry only raises its
`count` (a re-run is idempotent); disagreement is flagged as a conflict and the entry is left
untouched — same for entries you `rejected` and for `origin="user"` rows. A row a previous machine
pass proposed is updated in place, its previous target recorded in the notes, and runner-up English
spellings the model saw are kept there too.

The summary prints the matched / raw-only / reference-only chapter counts, the paired-line and
skipped-page counts, and the locked/proposed counts, followed by one line per unmatched chapter, OCR
failure, conflict and rejected entry still being extracted:

```text
reference: 3 matched chapter pair(s), 1 raw-only, 0 reference-only, 9 paired line(s), 0 page pair(s) skipped
reference:   raw chapter without reference match: Chapter 4
reference: 2 term(s) locked, 1 proposed
reference:   conflict: 민준: store has 'Minjun' (locked), reference extraction says 'Min-jun' — entry left untouched
```

Chapters without a reference match are normal, not an error. Exit 2 when the series has no chapters
or nothing is imported under `_reference_en` yet; exit 1 when any chapter's OCR failed. Like every
GPU command it holds the exclusive GPU lock while the models run.

```bash
uv run omniscan reference DemoSeries --dry-run
uv run omniscan reference DemoSeries --min-locks 2 --model translategemma:12b --force
```

### `omniscan story summarize`

Store a short English summary of what happens in each chapter, so the translate and judge stages of
**later** chapters know what already happened — names, relationships and plot state stop drifting
from chapter to chapter. Summaries live in the `chapter_summary` table of the series' `series.db`.

This is a manual, explicit command (an unconditional extra LLM call per chapter would change the
pipeline's cost profile — it is deliberately not part of `omniscan run`). Reading a summary back
into a later chapter's prompts is free and automatic: once a chapter has a stored summary, the
translate/judge prompts of every later chapter carry the last three prior chapters' summaries as
"Story so far". Summaries are computed from a chapter's `final.json` (the judged English text), so
run `omniscan translate` + `omniscan judge` for a chapter first.

| Argument/option | Meaning |
|---|---|
| `series` | series name (required) |
| `--chapter`, `-c <str>` | chapter folder name; repeatable. Default: all |
| `--model <name>` | chat model used for summarisation. Default: `gemma4:31b-cloud` |
| `--force` | re-summarize chapters that already have a summary |
| `--json` | emit one JSON object instead of text lines |

Chapters without `final.json` yet, and chapters whose reply is unusable, are skipped and reported —
re-run later. Already-summarized chapters are skipped unless `--force`. This command is CPU-only;
it does not touch the GPU.

```bash
uv run omniscan story summarize DemoSeries
uv run omniscan story summarize DemoSeries -c "Chapter 3" --force
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

### `omniscan usage`

Summarise the LLM usage the text stages already recorded, per chapter: one row per candidate run
under `translations/` (its profile and model) plus one `judge` row per chapter's `final.json`, with
per-series and grand totals in prompt/completion tokens, requests and seconds. A read-only report —
nothing is written, and no prices are applied (cost estimation is a separate decision). Artifacts
written before usage recording existed count as zeros, and corrupt files are skipped.

| Argument/option | Meaning |
|---|---|
| `series` | series name; omit for every series under `work_root` |
| `--json` | emit one JSON object (`rows`, `totals`) instead of the table |

An unknown series exits 2; a series with no recorded usage prints only the zero `TOTAL` line.

```bash
uv run omniscan usage SoloLeveling
uv run omniscan usage --json
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

Needs `ocr.json`, `final.json` and `inpaint.json` in the chapter work dir (and reads
`inpaint_lama.json` when it exists, to know which sound effects were erased); a chapter missing one
of them fails (exit 1). Per region, in reading order: bubble text is shaped to an ellipse inside the
bubble (see "Lettering and sound effects" below), free text is fitted into its text box grown by
`typeset.free_grow`, sound effects follow `sfx.mode`. The colour is the region's measured
`text_color` when it carries one, else black on light bubble fills and white on dark ones; free text
gets an outline (`typeset.stroke_free_px`) in its measured outline colour, or black/white by contrast.
Watermark regions are never laid out. Writes `layout.json` + `manifest.json` in the chapter work dir.
Exit codes as for `ingest`.

```bash
uv run omniscan typeset DemoSeries
```

### Lettering and sound effects

**Balloons.** Lettering follows the balloon the way a professional letterer sets it: every line's width
is the balloon's width at that height (an ellipse inside the bubble, `typeset.bubble_padding` kept free),
so lines are short at the top and bottom and longest in the middle; every line count is tried, and
among the layouts that let the text grow (nearly) largest the one that breaks best wins — after a
sentence or a clause, never after "the"/"a"/"to", no little word stranded on its own line. A word that
fits no line even at `typeset.min_px` is hyphenated. Sizes are kept consistent across the chapter:
dialogue is at most `typeset.size_spread` × the chapter's typical balloon size (shouted lines —
capitals, or ending in "!!" — `typeset.shout_spread`).

**Styles and fonts.** `typeset.style` picks a preset: `webtoon` (Mali, mixed case — the clean look of
official English webtoons) or `manga` (Kalam in capitals — the hand-lettered look official English manga
get from CC Wild Words); `auto` uses `manga` for Japanese sources. Any role can use your own licensed
font:

```toml
[typeset]
style = "manga"
font_dialogue = "C:/Users/me/Fonts/CCWildWords-Regular.ttf"
```

**Sound effects.** The text detector does not report sound effects on real pages, so after OCR a
free-text region becomes a sound effect when its letters are onomatopoeia from `config/sfx_text.toml`
(ko/ja/zh; repeats and stretched letters count: 쾅쾅쾅, 쿠구구궁, ドドド; add your own words in
`~/.config/omniscan/sfx_text.toml`). Most effects drawn into the art are not detected at all, so with
`sfx.sweep` (on by default) the OCR stage also runs CRAFT — the scene-text detector EasyOCR uses (MIT,
downloaded on first use, `sfx-detector-craft` in `omniscan models list`) — over every page: lettering
outside the detected regions is cut out, turned level when tilted, read twice by the chapter's OCR
engine (the crop, and the letters alone in black on white without the art around them), and kept as a
sound effect when it is a lexicon word within a misread stroke or two (광! for 쾅!, 번찍 for 번쩍; the
lexicon's spelling is used when only one word is that close, the raw reading kept in `ocr_alt`).
Anything else it finds stays untouched — an unknown word may be part of the art. The OCR stage measures each effect's fill and outline colour, tilt
and stroke weight; the translator is asked for the English effect an official release would letter;
and with `sfx.mode = "replace"` the English effect is drawn over the erased original in that style —
same colours and outline, same tilt, a heavy / bold-brush / light-marker face matching the original's
weight, stacked letter by letter when the original ran down a tall column. Where the original stays on
the page (`sfx.mode = "subtitle"`, or it could not be erased) a small outlined translation is set just
below it; `sfx.mode = "keep"` leaves effects untranslated.

To look at the result without a chapter: `uv run python scripts/lettering_demo.py` letters synthetic
pages over drawn art with this exact code (LaMa included) into `data/lettering_demo/`, and
`uv run python scripts/sfx_sweep_check.py` counts how many of ten drawn-in effects the sweep finds with
this machine's models (`--fonts <folder>` letters them in extra faces, `--out` saves the pages).

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

### `omniscan match chapters`

Align two independently-sourced chapter sets of the same series before comparing them page by page:
a raw-language set and an official English release do not necessarily line up 1:1 — one side may
carry an extra prologue or ad/insert chapter, chapter numbering may differ, folder names always can,
and either side may simply have more chapters. The command computes which chapter of the second
directory corresponds to which chapter of the first, with a confidence signal, and flags every
chapter that has no confident counterpart (never forcing a wrong pairing just to fill a slot).

Matching runs on page art alone: every page gets a dHash (the promo filter's perceptual hash, so it
is language-independent and resolution-independent), the pages of every candidate chapter pair are
aligned with a global sequence alignment (tolerant of an extra ad page on either side), and the
chapters of the two sets are aligned the same way, using each pair's page alignment as its evidence.
Folder names are used only to put chapters in reading order, never to match them.

```bash
uv run omniscan match chapters data/raws/PepperCarrotKR data/translated-check/PepperCarrotKR
uv run omniscan match chapters data/raws/PepperCarrotKR data/translated-check/PepperCarrotKR --out pepper_match.json --json
```

| Argument/option | Meaning |
|---|---|
| `dir_a`, `dir_b` | the two chapter-set directories (one subfolder per chapter); either order works |
| `--out <path>` | mapping artifact path. Default: `chapter-match.json` in the current folder |
| `--force` | overwrite an existing mapping (the file is meant to be hand-edited, so overwriting is refused without it) |
| `--page-similarity <0..1>` | dHash similarity at/above which two pages may align. Default: `0.75` |
| `--page-gap <float>` | penalty for leaving a page unmatched inside a chapter pair. Default: `0.25` |
| `--chapter-gap <float>` | penalty for leaving a chapter unmatched. Default: `0.6` |
| `--min-quality <0..1>` | page-coverage quality at/above which a chapter pair may match. Default: `0.3` |
| `--review-quality <0..1>` | matched chapters below this quality are listed for a second look. Default: `0.5` |
| `--json` | print the full mapping JSON to stdout instead of the text summary |

The output is one JSON file, hand-editable on purpose: `matched` (each with the `a`/`b` folder
names, `quality` — the aligned pages' similarity sum divided by the larger page count, `pages_a`,
`pages_b`, the aligned `pages` as `(page_a, page_b, similarity)` triples, `runner_up` (the strongest
alternative partner on either side, when one was even allowed), `review` true/false) plus
`unmatched_a` / `unmatched_b` folder-name lists and the `thresholds` the run used. Fix a wrong
match by editing the `a`/`b` names or moving chapters between `matched` and the unmatched lists;
keep `quality` and `pages` as they are or delete them, but keep the field names.

The text summary prints matched/unmatched counts, every unmatched chapter (eyeball these before
trusting the mapping), and the weakest matches — lowest quality first — each with its `runner_up`
when a close alternative existed. Missing or chapter-less directories exit 2; an existing mapping
file is never overwritten without `--force`.

### `omniscan match duplicates`

Flag accidental duplicate imports **within one chapter set**: two chapter folders under one root
that hold (nearly) the same pages — an accidental double download, a reposted chapter re-imported
under a different name, a copy-paste mistake. This is a different question from `match chapters`
(aligning two *independent* sets 1:1): here every chapter is compared against every other chapter of
the *same* directory, and one chapter can appear in several flagged pairs (three accidental copies
of a chapter pair up as three pairs). Pairs score the same page-coverage quality as the matcher, and
a pair is flagged once it reaches `--min-quality` — whose default of `0.9` is deliberately much
higher than the matcher's `0.3`: ordinary different chapters of one series share almost no page art,
while "literally the same pages imported twice" scores near 1.0, so a high bar keeps the report free
of near-misses.

```bash
uv run omniscan match duplicates data/raws/PepperCarrotKR
uv run omniscan match duplicates data/raws/PepperCarrotKR --out dups.json --json
```

| Argument/option | Meaning |
|---|---|
| `root` | the chapter-set directory to scan (one subfolder per chapter) |
| `--out <path>` | report artifact path. Default: `chapter-duplicates.json` in the current folder |
| `--force` | overwrite an existing report (the file is meant to be hand-edited, so overwriting is refused without it) |
| `--page-similarity <0..1>` | dHash similarity at/above which two pages may align. Default: `0.75` |
| `--page-gap <float>` | penalty for leaving a page unmatched inside a chapter pair. Default: `0.25` |
| `--min-quality <0..1>` | page-coverage quality at/above which a chapter pair is a duplicate. Default: `0.9` |
| `--json` | print the full report JSON to stdout instead of the text summary |

The report is hand-editable JSON like the chapter mapping: `duplicates` (each with the `a`/`b`
folder names — `a` the natural-sort-earlier of the two — and `quality`, sorted by descending
quality) plus the `thresholds` the run used. Nothing is deleted or merged automatically; a flagged
pair is a hint for you to remove one copy by hand. A chapter folder with no page images can never
appear in a pair; a directory with no chapter folders at all exits 2.

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

Stored watermark regions affect the chapters on the next `detect` run of the series: every stored
zone becomes a watermark region on every page (erased whole by the inpaint stage, see
`inpaint.remove_watermarks`), and any detected region whose box mostly (>= 50 % of its own area) falls
inside a stored watermark zone is excluded from translation, scoring and evaluation the same way tier 1
(text-pattern) and tier 2 (image-hash) matches are. Existing chapters re-detect automatically, because the detect stage now depends on
`watermarks.json`.

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
`/api/series/{s}/chapters/{c}/inpaint` (the inpaint stage's `inpaint.json`) and
`/api/series/{s}/chapters/{c}/inpaint/patches/{region_id}.png` (one region's patch as an RGBA PNG, alpha = its text mask),
`/api/series/{s}/chapters/{c}/layout` (the typeset stage's `layout.json`),
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

### `omniscan library`

Covers and metadata: every series can get a cover image and basic metadata from the free structured
metadata APIs — no LLM, no scraping. `cover` searches **AniList**, **MangaDex** and **Jikan
(MyAnimeList)** in that order; a failing or rate-limited provider is skipped and reported, it never
aborts the search. The candidates are scored against the query and the best one's cover is saved to
`<series>/_meta/cover.jpg|png|webp` next to a `series.json` that records where everything came from.
Your own image always wins: `--file` sets it instead and marks the source `user`.

| Subcommand | Effect |
|---|---|
| `cover SERIES [--title TITLE] [--provider anilist\|mangadex\|jikan\|all] [--pick N] [--file PATH] [--size large\|small] [--json]` | print the numbered candidate list (`#`, provider, title, year, country, score) plus any provider errors, then download the best match's cover (a match must score ≥ 0.75); `--title` overrides the search text (default: the series name), `--pick N` chooses from the list, `--file` sets your own image with no network call, `--size small` fetches the smaller cover variant. Without a confident match: `no confident match — choose one with --pick N`, exit 1, nothing written |
| `info SERIES [--json]` | print the stored `series.json` fields (title, provider, year, country, status, credit, cover path), or `no metadata for SERIES` (exit 1) |

The series folder does not need to exist yet — `_meta` is created under `paths.library_root`. The
client honours each provider's rate limit (AniList 30/min, MangaDex 5/s, Jikan 3/s): a `429` waits
out its `Retry-After` (capped at 60 s) and server errors are retried with a short backoff. Covers
are copyrighted artwork of the series: they are stored in your library for your own display, with
the source recorded in `series.json` (`cover_source`, `credit`), and are never re-distributed.

```bash
uv run omniscan library cover "Solo Leveling"
uv run omniscan library cover "Solo Leveling" --pick 3
uv run omniscan library cover "Solo Leveling" --file ~/Pictures/my-cover.png
uv run omniscan library info "Solo Leveling"
```

```text
1. anilist    Solo Leveling  2018  KR  1.000
2. anilist    The Privilege of the Second Life is Power Leveling  2024  KR  0.592
3. anilist    Solo Leveling: Ragnarok  2024  KR  0.850
jikan: HTTP 504
cover saved: ~/omniscan/library/Solo Leveling/_meta/cover.jpg (anilist)
```

## Desktop app

`omniscan gui` opens a native desktop window over the same library and config the CLI uses. The GUI
is an optional extra (PySide6); install it once with `uv sync --extra gui`, then:

```bash
uv run omniscan gui          # or: uv run python -m omniscan.gui
```

Without the extra installed the command prints one line naming the extra and exits 2.

The window has six pages in the left sidebar (also `Ctrl+1`…`Ctrl+6`). The status bar shows the
configured GPU device and the job state; window size and the last open page are remembered across
restarts.

**Library** lists every series under `paths.library_root` with its chapter count. Picking a series
fills the chapter table with one row per chapter and one column per pipeline stage (`ingest` …
`export`), colored by state from the chapter's manifest: green `done` (recorded with its outputs on
disk), yellow `stale` (recorded but its output is missing, or an earlier stage re-ran after it —
the stage would re-run), red `failed`, gray `not run`. Double-click a chapter (or pick a series and
switch pages) to open it in the Reader. Refresh re-reads the library.

**Reader** is the side-by-side raw | output compare view: prev/next chapter buttons, a chapter
switcher, zoom −/+ (`Fit width` resets), a `Sides` selector (`Both` / `Raw only` / `Output only`),
`Jump to slice…` to scroll both panes to one output slice, and `Linked scrolling` so the two panes
follow each other while comparing. A chapter with no output yet shows the caption
`Output (not translated yet)` and an empty right pane.

**Run** starts pipeline runs. Pick the series and chapters (or `All chapters`), then a mode:

- **Full** — every stage over every selected chapter (the CLI `run` without `--stages`).
- **Subset** — tick exactly the stages to run (the ten checkboxes; LaMa inpainting and
  `Force re-run` are checkboxes too). This is the CLI's `--stages`.
- **Step** — runs the preview chapter and pauses after each of its stages: the preview panel shows
  what the stage produced and waits for `Continue` (next stage) or `Abort` (end the run). The same
  gate as the CLI's `run --step`.
- **Auto** — reserved for the upcoming automatic mode; today it runs exactly like Full.

One run at a time; the form is disabled while one is running and the status bar shows `job: running`.
The progress bar counts stages across chapters, the log under it prints one line per finished stage,
and `Cancel` stops after the current stage (the summary line ends with `aborted=stopped`). A failed
chapter or a broken setup (e.g. no Ollama daemon) is reported inline. The run page takes the
exclusive GPU lock for real-GPU work, so a GUI run and a CLI run queue up instead of colliding.

**Models** is the `omniscan models` screen: the catalog with role/language filters, each model's fit
on this machine, download/remove buttons and `Download required models`. Hardware detection runs in
the background (it imports torch); the header shows the same snapshot `omniscan hardware` prints.

**Settings** edits the config in place, with validation:

- **Global** — paths, GPU device (`auto`, `cpu`, `mps`, `cuda[:N]`; editable), warm-up, codec, OCR
  engine and models, translation settings, slicer strategy, the filter switch and threshold. Every
  change is written to the user `config.toml` when it validates; a refused value shows the reason
  inline and the editor reverts. `Reset` drops the override so the built-in default applies again.
- **Per-series** — the same override mechanism as `series.toml` in the series' library folder: pick
  a section and key, enter a value (bools/numbers/lists as JSON), press `Set`; existing overrides
  are listed with `Remove` buttons.
- **Translation** — the translation profiles from `config/translation_profiles.toml` plus the user's
  `translation_profiles.toml`; ticking a profile enables it (written to the user file), and a
  translate run runs exactly the enabled profiles.
- **Hardware** — the machine snapshot from `hw detect` plus one row per catalog model that does not
  fit (level, device, why). Detection runs when you open the tab (it imports torch) or on
  `Re-detect`.

A successful settings write reloads the config into every page (the Library re-scans, the Run page
re-lists series). The window never touches `secrets.env`; translation runs read it exactly as the
CLI does.

**Import** is the same import the CLI's `omniscan import` does, with an editable preview first:
browse or drag-and-drop a folder or `.zip`/`.cbz` archive, pick a series name, then adjust the
detected chapter grouping — move pages between chapters, reorder, rename, merge or split chapters —
before committing. Non-JPEG pages are flagged for conversion (quality 95, on the CPU) in the
preview. `Move instead of copy` deletes the source pages as they're imported (disabled for
archives, since there both extraction and the archive itself would need separate handling).

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

Six chapter views (Slicer, OCR, Translation, Reader, Inpaint, Layout) need a chapter; the **Filtered**
view only needs a series and shows its every chapter that has filtered items.

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

The **Inpaint view** (switch with the `Inpaint` button) checks the inpaint stage's work on the stacked
raw pages. A `raw ↔ clean` slider reveals the clean layer: each `inpaint.json` item's stored patch from
`patches.npz`, positioned exactly where export applies it — a `flat` item (and any item whose patch was
not stored) shows its solid fill colour instead, and a `none` item is not cleaned at all. A `show patch
outlines` checkbox (on by default) draws every item's box, colored by method (flat blue, lama purple,
none gray; dashed = nothing was cleaned, a `· needs lama` label flags items the flat fill could not
clean). Clicking a box opens a side panel with the method, `needs_lama`, `mask_px` and the fill colour
swatch, plus the patch itself enlarged. The view shows `no inpaint.json yet` until `omniscan inpaint`
has run.

The **Layout view** (switch with the `Layout` button) draws every `layout.json` item on the stacked raw
pages, colored by font role (dialogue blue, thought teal, shout red, narration purple, free orange, sfx
gray; dashed = the typesetter predicted `overflow`). Checkboxes hide/show roles, and clicking an item
opens a side panel with the font, size, alignment, the wrapped lines, colour and stroke swatches, and
`overflow` highlighted red when true. The view shows `no layout.json yet` until `omniscan typeset`
has run.

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
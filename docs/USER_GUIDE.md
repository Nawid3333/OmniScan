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
| `work_root/<Series>/<Chapter N>/filter.json` | `omniscan filter run` / `restore` | promo-filter decisions |
| `work_root/<Series>/<Chapter N>/manifest.json` | the stage runner | per-stage status, inputs and config hashes |
| `work_root/<Series>/<Chapter N>/converted/` | `omniscan ingest` | JPEG cache for raws that were not JPEG |
| `output_root/<Series>/<Chapter N>/` | `omniscan export` (not implemented yet) | final English slices |
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
| `paths.models_dir` | local cache for model weights | no consumer yet |
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

Stubs (`acquire`, `ocr`, `judge`, `inpaint`, `typeset`, `export`, `run`,
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
file, the pipeline roots and the configured codec. Any `FAIL` row makes the command exit 1; `WARN`
rows do not. Writes nothing.

```bash
uv run omniscan doctor
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
`work_root/<series>/<chapter>/translations/<profile>.json`. Needs `ocr.json`, which no stage produces
yet, so there is nothing to translate until the OCR stage lands.

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
nothing to pack. Today the output root only fills up once the export stage exists, so packing needs
finished images placed (or copied) there by hand.

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

### `omniscan queue`

A persistent job queue for running pipeline stages over series without babysitting one command per
series. You add jobs, then a worker drains them one at a time. Everything lives in one SQLite file,
`work_root/queue.db`; jobs survive restarts, and a worker that died mid-job recovers automatically
(the interrupted job is queued again with its attempt refunded). Exactly one worker per `queue.db` is
supported.

A job is "run these stages over this series (optionally only some chapters)". Jobs for stages that
are not implemented yet are accepted but fail permanently with a clear message, so adding the stage
later only means adding it to the executor's stage table.

| Subcommand | Effect |
|---|---|
| `add <series>` | queue a job. Options: `--stage`, `-s <name>` (repeatable, default `slice`), `--chapter`, `-c <name>` (repeatable, default all), `--priority`, `-p <int>` (higher runs first, default 0), `--max-attempts <int>` (default 2), `--force` |
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
No OCR stage exists yet, so this view only has data once a chapter has an `ocr.json`; until then it shows
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
reading column width, fitted to the viewport. The export stage does not exist yet, so the view shows
`no output yet` until finished images are placed in the chapter's output folder by hand.

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

`ingest`, `slice` and `detect` are stage runs recorded in the chapter's `manifest.json`: stage
version, content hash of the inputs, hash of the relevant config subset, status and outputs. A second
run with unchanged inputs and config is a no-op — it prints `skipped (0.00s)` per stage and exits 0.
Use `--force` to re-run even when up to date. `slice` always runs `ingest` first when the ingest is
missing or stale, so `slice` alone is enough for a fresh chapter (`detect` pulls in both the same
way).

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
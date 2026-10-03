# OmniScan

> **Restarted on 2026-09-27** with a bigger goal: every OS and GPU vendor, a manual Translator Studio and opt-in
> shared data. See [`docs/ROADMAP.md`](docs/ROADMAP.md); how to pick up the work: [`docs/HANDOFF.md`](docs/HANDOFF.md).

GPU end-to-end manhwa/manga translator (Korean / Chinese / Japanese → English), with a native desktop app.
Runs on Windows, macOS and Linux, on NVIDIA, AMD, Intel and Apple GPUs (or the CPU). Where it is going: [`docs/ROADMAP.md`](docs/ROADMAP.md).
Chapters are imported from a local folder, normalised and cut into reading slices on the GPU, and — as
the remaining stages land — detected, OCR'd, translated through Ollama, judged, inpainted, typeset and
packaged into CBZ/PDF.

## Status

| Stage | Status |
|---|---|
| import | working — `omniscan import` (a folder, a .zip/.cbz, or a reader URL through manhwa-manga-downloader: `--from-url`) |
| ingest | working — `omniscan ingest` |
| slice | working — `omniscan slice` |
| promo filter | module only — `omniscan filter run` / `filter restore` |
| watermark regions | working — `omniscan watermark add` / `list` / `remove`; watermarks (stored zones, ad/site text) are erased by `inpaint` |
| glossary | working — `omniscan reference` (bootstrap from imported official `_reference_en` chapters) + `omniscan glossary list` / `export` / `import` / `propose` (from raw OCR text, no reference needed) |
| story memory | working — `omniscan story summarize` (per-chapter summaries, fed automatically into later chapters' translate/judge prompts) |
| chapter matching | working — `omniscan match chapters DIR_A DIR_B` (perceptual-hash alignment between two independently-sourced chapter sets) |
| detect | working — `omniscan detect` (regions.json; no text yet) |
| ocr | working — `omniscan ocr` (ocr.json; sound effects found by lexicon and a whole-page CRAFT sweep) |
| translate | working — `omniscan translate` (candidate runs only; needs ocr.json from `omniscan ocr`); pipeline re-runs translate and judge only the regions that changed |
| judge | working — `omniscan judge` (final.json) |
| eval | working — `omniscan eval` (scores ocr.json/final.json against ground-truth SVG text layers) |
| inpaint | working — `omniscan inpaint` (flat fills, glyph-precise masks; `--lama` rebuilds the art under lettering) |
| typeset | working — `omniscan typeset` (balloon-shaped lettering, webtoon/manga presets, style-matched SFX) |
| export | working — `omniscan export` (needs inpaint.json, patches.npz, layout.json) |
| run | working — `omniscan run` (ingest → export, three passes) |
| pack | working — `omniscan pack` (CBZ/PDF of finished output) |
| chapter projects | working — `omniscan project pack` / `unpack` / `show`: a chapter's raw pages and all its work (stage outputs, hand edits with undo history, cleanup) in one `.omniscan` file, to pass between group members; unpacking checks every file and never overwrites without `--force` |
| job queue | working — `omniscan queue add` / `list` / `run` / `pause` / `resume` / `cancel` / `retry` / `clear` |
| models | working — `omniscan models list` / `download` / `remove` / `verify` |
| hardware | working — `omniscan hardware [--json] [--simulate <profile>]` (eleven machine profiles stand in for GPUs we do not own) |
| tune | working — `omniscan tune [--apply]` picks the device, memory budget, OCR engine, batch sizes and LaMa default for this machine (the Settings page's *Optimise for this PC*) |
| studio (manual editing) | working — the web UI's Studio view: draw/move/resize/delete text boxes, fix OCR text, write English lines, translate one region or a page on demand, clean by hand with a brush (inpaint, fill, clone, restore), hand-set the lettering (font, size, colours, outline, angle, box, line breaks) with a live preview of the finished page, choose where the output images split (Slicer view); edits survive every re-run (`edits.json`, `cleanup.json`); the same edits from the command line with `omniscan edit`; LabelPlus files in and out (`omniscan labelplus`), layered PSD pages (`omniscan psd export`), BallonsTranslator projects in and out (`omniscan ballons`), manga-image-translator text files in (`omniscan mit import`); undo/redo of every hand edit, find & replace across a series, a consistency report (`omniscan consistency`), and a per-line status for proofreading (todo / edited / checked; `omniscan edit check`) |
| learning | working — each series learns from your Studio corrections: repeated OCR fixes, deletions and watermark/sound-effect labels apply to later chapters, your English lines become a translation memory and your rewritten names/terms the model's preferred wording (`memory.json`; Learned view, `omniscan learn`) |
| contributing | working (local files only) — `omniscan contribute export` writes a series' hand corrections with the pages they were made on to a zip archive (no names, paths or image metadata); on by default, `[share] enabled = false` opts out; no upload yet |
| web viewer | working — `omniscan serve` (the built UI at the API's address; Slicer, OCR, Translation, Reader and Filtered views — read-only except the Filtered view's Restore button; the OCR/Translation views need an `ocr.json`) |
| desktop app | working — `omniscan gui` (PySide6: Library, Reader, Run, Models, Settings, Import, Studio — the translator's workbench: move/resize/draw boxes on the strip, fix OCR and English in a table with multi-selection, translate or re-read lines on demand, lettering styles for many balloons at once, proofreading status, undo/redo, page navigation and a live preview of the finished page; Queue — batch jobs over series and chapters, the same queue as `omniscan queue`) |
| update | working — `omniscan update check` / `download` |

"working" = usable from the CLI today. "module only" = the code and its own sub-commands exist, but no
pipeline stage consumes it yet.

## Download

The `build` workflow packages the desktop app and the command line for Windows, macOS and Linux
(`omniscan-<os>-<arch>.zip`: unzip, run `OmniScan`; no Python needed). See "Installing the packaged app"
in [docs/USER_GUIDE.md](docs/USER_GUIDE.md). The packaged build runs on any PC on the CPU; **Settings →
Hardware → Optimise for this PC** (or `omniscan tune --apply`) fits the settings to the machine, and the
GPU builds of PyTorch still come from the developer install below.

## Requirements

Windows, macOS or Linux. Real-hardware numbers are measured on the reference machine (Windows 11, AMD RX 9070 XT,
ROCm 10); CI runs the test suite on all three OSes.

- Ollama installed and running at `localhost:11434`.
- Python 3.14, managed by [uv](https://docs.astral.sh/uv/). PyTorch comes from exactly one backend extra — never
  `pip install torch` by hand:

  | Your GPU | Extra |
  |---|---|
  | AMD RX 9070 series (gfx1201), Windows | `rocm-gfx1201` |
  | NVIDIA (Windows, Linux) | `cuda` |
  | Intel Arc / Core Ultra graphics (Windows, Linux) | `xpu` |
  | Apple Silicon (macOS) | `mps` |
  | none of these | `cpu` (works, slowly) |

  `gpu.device = "auto"` then picks the strongest discrete GPU the installed build can reach.
- Node 24 to build the web UI once (`npm install && npm run build` in `webui/`).

## Quickstart

```bash
uv sync --extra cuda --extra gui   # or rocm-gfx1201 / xpu / mps / cpu, see the table above
uv run omniscan doctor
uv run python scripts/make_demo_chapter.py
uv run omniscan slice DemoSeries
cd webui && npm install && npm run build && cd ..   # once, and after every web UI update
uv run omniscan serve --open
```

`make_demo_chapter.py` writes a synthetic `DemoSeries/Chapter 1` into the configured library root
(`~/omniscan/library` by default), `slice` normalises and cuts it, and `serve` opens the Studio on it at
`http://127.0.0.1:8000/` (API and UI at one address; `npm run dev` in `webui/` for UI development).

## Configuration

```bash
# precedence: built-in defaults < config/default.toml < ~/.config/omniscan/config.toml < environment
# env pattern: OMNISCAN_<SECTION>__<KEY>, e.g.:
OMNISCAN_PATHS__LIBRARY_ROOT=/data/omniscan/library OMNISCAN_GPU__CODEC=turbo
# secrets live only in ~/.config/omniscan/secrets.env (OLLAMA_API_KEY) — never commit them;
# `omniscan doctor` reports which are unset
```

See the full key reference and per-command details in the user guide:

- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — folder layout, configuration, every working command, the
  web viewer, troubleshooting
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — architecture and contracts
- [docs/PLAN.md](docs/PLAN.md) — the full build plan
- [docs/HANDOFF.md](docs/HANDOFF.md) — start here for a new AI session or contributor (reading order, working agreement, lessons)
- [docs/CHECKPOINT.md](docs/CHECKPOINT.md) — current state, evidence gathered, and how to resume
- [docs/DECISIONS.md](docs/DECISIONS.md) — why the important choices were made, with the evidence
- [docs/OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md) — decisions still waiting on the owner
- [CLAUDE.md](CLAUDE.md) — contributor/agent rules

## Licence

OmniScan is free software under the [GNU General Public License v3.0](LICENSE) (GPL-3.0-only, chosen by the owner on
2026-09-27): every copy and fork stays open source. Code reused from other GPL-3.0 or permissively licensed
translation tools keeps its original copyright notice and is credited in `THIRD_PARTY.md`.

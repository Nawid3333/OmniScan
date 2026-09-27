# OmniScan

> **Development is paused** (since 2026-09-25). Everything is merged and CI is green; to resume, start with the
> "Development paused" section in [`docs/HANDOFF.md`](docs/HANDOFF.md).

GPU end-to-end manhwa/manga translator (Korean / Chinese / Japanese → English). Windows 11 native, AMD ROCm — one machine, one GPU.
Chapters are imported from a local folder, normalised and cut into reading slices on the GPU, and — as
the remaining stages land — detected, OCR'd, translated through Ollama, judged, inpainted, typeset and
packaged into CBZ/PDF.

## Status

| Stage | Status |
|---|---|
| import | working — `omniscan import` |
| ingest | working — `omniscan ingest` |
| slice | working — `omniscan slice` |
| promo filter | module only — `omniscan filter run` / `filter restore` |
| watermark regions | working — `omniscan watermark add` / `list` / `remove`; watermarks (stored zones, ad/site text) are erased by `inpaint` |
| glossary | working — `omniscan reference` (bootstrap from imported official `_reference_en` chapters) + `omniscan glossary list` / `export` / `import` / `propose` (from raw OCR text, no reference needed) |
| story memory | working — `omniscan story summarize` (per-chapter summaries, fed automatically into later chapters' translate/judge prompts) |
| chapter matching | working — `omniscan match chapters DIR_A DIR_B` (perceptual-hash alignment between two independently-sourced chapter sets) |
| detect | working — `omniscan detect` (regions.json; no text yet) |
| ocr | working — `omniscan ocr` (ocr.json; sound effects found by lexicon and a whole-page CRAFT sweep) |
| translate | working — `omniscan translate` (candidate runs only; needs ocr.json from `omniscan ocr`) |
| judge | working — `omniscan judge` (final.json) |
| eval | working — `omniscan eval` (scores ocr.json/final.json against ground-truth SVG text layers) |
| inpaint | working — `omniscan inpaint` (flat fills, glyph-precise masks; `--lama` rebuilds the art under lettering) |
| typeset | working — `omniscan typeset` (balloon-shaped lettering, webtoon/manga presets, style-matched SFX) |
| export | working — `omniscan export` (needs inpaint.json, patches.npz, layout.json) |
| run | working — `omniscan run` (ingest → export, three passes) |
| pack | working — `omniscan pack` (CBZ/PDF of finished output) |
| job queue | working — `omniscan queue add` / `list` / `run` / `pause` / `resume` / `cancel` / `retry` / `clear` |
| models | working — `omniscan models list` / `download` / `remove` / `verify` |
| hardware | working — `omniscan hardware [--json]` |
| studio (manual editing) | working — the web UI's Studio view: draw/move/resize/delete text boxes, fix OCR text, write English lines, translate one region or a page on demand, clean by hand with a brush (inpaint, fill, clone, restore), hand-set the lettering (font, size, colours, outline, angle, box, line breaks) with a live preview of the finished page; edits survive every re-run (`edits.json`, `cleanup.json`) |
| web viewer | working — `omniscan serve` + `npm run dev` (Slicer, OCR, Translation, Reader and Filtered views — read-only except the Filtered view's Restore button; the OCR/Translation views need an `ocr.json`) |
| desktop app | working — `omniscan gui` (PySide6: Library, Reader, Run, Models, Settings, Import) |
| update | working — `omniscan update check` / `download` |

"working" = usable from the CLI today. "module only" = the code and its own sub-commands exist, but no
pipeline stage consumes it yet.

## Requirements

This project targets one machine: Windows 11 native, AMD RX 9070 XT (gfx1201, 16 GB) with AMD's ROCm 10
wheels. It is not built or tested for any other OS or GPU vendor.

- Windows 11. Ollama installed and running at `localhost:11434`.
- Python 3.14, managed by [uv](https://docs.astral.sh/uv/). PyTorch comes from the `rocm-gfx1201` extra —
  never `pip install torch` from PyPI.
- Node 24 for the web UI only (`npm run dev` in `webui/`).

## Quickstart

```bash
uv sync --extra rocm-gfx1201 --extra gui
uv run omniscan doctor
uv run python scripts/make_demo_chapter.py
uv run omniscan slice DemoSeries
uv run omniscan serve
```

Then in a second terminal, start the web UI and open the local URL it prints:

```bash
cd webui
npm install
npm run dev
```

`make_demo_chapter.py` writes a synthetic `DemoSeries/Chapter 1` into the configured library root
(`~/omniscan/library` by default), `slice` normalises and cuts it, and `serve` + the UI show the result.

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
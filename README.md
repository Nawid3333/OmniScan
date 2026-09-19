# OmniScan

GPU end-to-end manhwa/manga translator (Korean / Chinese / Japanese → English) on your own GPU (developed on Windows 11 with AMD ROCm).
Chapters are imported from a local folder, normalised and cut into reading slices on the GPU, and — as
the remaining stages land — detected, OCR'd, translated through Ollama, judged, inpainted, typeset and
packaged into CBZ/PDF.

## Status

| Stage | Status |
|---|---|
| acquire | not implemented |
| import | working — `omniscan import` |
| ingest | working — `omniscan ingest` |
| slice | working — `omniscan slice` |
| promo filter | module only — `omniscan filter run` / `filter restore` |
| watermark regions | module only — `omniscan watermark add` / `list` / `remove` |
| glossary | module only — `omniscan glossary list` / `export` / `import` |
| detect | working — `omniscan detect` (regions.json; no text yet) |
| ocr | working — `omniscan ocr` (ocr.json) |
| translate | working — `omniscan translate` (candidate runs only; needs ocr.json from `omniscan ocr`) |
| judge | working — `omniscan judge` (final.json) |
| eval | working — `omniscan eval` (scores ocr.json/final.json against ground-truth SVG text layers) |
| inpaint | working — `omniscan inpaint` (flat fill; `--lama` for textured art) |
| typeset | working — `omniscan typeset` (layout.json; needs ocr.json, final.json, inpaint.json) |
| export | working — `omniscan export` (needs inpaint.json, patches.npz, layout.json) |
| run | working — `omniscan run` (ingest → export, three passes) |
| pack | working — `omniscan pack` (CBZ/PDF of finished output) |
| job queue | working — `omniscan queue add` / `list` / `run` / `pause` / `resume` / `cancel` / `retry` / `clear` |
| web viewer | working — `omniscan serve` + `npm run dev` (Slicer, OCR, Translation, Reader and Filtered views — read-only except the Filtered view's Restore button; the OCR/Translation views need an `ocr.json`) |

"working" = usable from the CLI today. "module only" = the code and its own sub-commands exist, but no
pipeline stage consumes it yet.

### Not implemented yet

| Command | Behaviour |
|---|---|
| `acquire`, `reference` | registered stubs; they print `not implemented yet` and exit 2 |

## Requirements

- Windows 11 (tested natively; Linux/WSL2 also works). Ollama installed and running at `localhost:11434`.
- A GPU PyTorch supports. Tested: AMD RX 9070 XT (gfx1201, 16 GB) with AMD's ROCm 10 wheels on Windows and Linux/WSL.
  NVIDIA, Apple Silicon (MPS) and CPU-only are expected to work through PyTorch but are untested; `gpu.device = "auto"`
  picks the strongest discrete GPU (it skips integrated GPUs).
- Python 3.14, managed by [uv](https://docs.astral.sh/uv/). PyTorch comes from AMD's ROCm 10 index —
  never `pip install torch` from PyPI.
- Node 24 for the web UI only (`npm run dev` in `webui/`).

## Quickstart

```bash
uv sync
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
# secrets live only in ~/.config/omniscan/secrets.env (OLLAMA_API_KEY, EXTRACTPICS_API_KEY,
# OMNISCAN_RELAY_CLIENT_TOKEN) — never commit them; `omniscan doctor` reports which are unset
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
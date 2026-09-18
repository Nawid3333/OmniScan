# OmniScan

GPU end-to-end manhwa/manga translator (Korean / Chinese / Japanese → English) for AMD ROCm on WSL2.
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
| detect | not implemented |
| ocr | not implemented |
| translate | not implemented |
| judge | not implemented |
| inpaint | not implemented |
| typeset | not implemented |
| export | not implemented |
| pack | working — `omniscan pack` (CBZ/PDF of finished output) |
| job queue | working — `omniscan queue add` / `list` / `run` / `pause` / `resume` / `cancel` / `retry` / `clear` |
| web viewer | working — `omniscan serve` + `npm run dev` (Slicer, OCR, Translation, Reader and Filtered views — read-only except the Filtered view's Restore button; the OCR/Translation views need an `ocr.json`, which no stage produces yet) |

"working" = usable from the CLI today. "module only" = the code and its own sub-commands exist, but no
pipeline stage consumes it yet.

### Not implemented yet

| Command | Behaviour |
|---|---|
| `acquire`, `detect`, `ocr`, `translate`, `judge`, `inpaint`, `typeset`, `export`, `run`, `reference` | registered stubs; they print `not implemented yet` and exit 2 |

## Requirements

- Windows 11 with WSL2 Ubuntu (26.04 tested); mirrored networking so WSL can reach the Windows-side
  Ollama at `localhost:11434`.
- An AMD GPU supported by ROCm 10 (tested: RX 9070 XT, gfx1201, 16 GB VRAM).
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
- [CLAUDE.md](CLAUDE.md) — contributor/agent rules
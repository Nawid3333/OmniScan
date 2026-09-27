# Roadmap — OmniScan as the default manga/manhwa translation tool

Written 2026-09-27, when the owner restarted the project with a bigger goal. This file is the direction; the
contracts stay in `docs/ARCHITECTURE.md`, decisions and their evidence in `docs/DECISIONS.md`.

## The goal (owner, 2026-09-27)
> Make this a program that runs on Mac, Windows and Linux and on Intel, AMD and NVIDIA GPUs. It should be the
> default tool for automated manga and manhwa translation, and also a choice for translators and translation
> groups who do the work by hand, with everything that comes with it. Give them the option to send data to a
> database so the program keeps improving: the automatic side and the manual tool side both win.

This reverses the 2026-09-24 "one machine only" decision. The owner's PC (Windows 11 + RX 9070 XT, ROCm) stays the
reference machine where real-hardware benchmarks are measured; every other platform must install and run.

Three audiences, one program:
1. **Readers / automatic users** drop in a chapter and get an English release (the pipeline that already exists).
2. **Translators and groups** use the same program as their workbench: the machine does the first pass, people
   fix OCR, translation, cleaning and lettering by hand, and export a release.
3. **Everyone, optionally,** shares their corrections so the models, glossaries and defaults get better.

## Decision: our own desktop app, not the browser
OmniScan already has a native desktop app (`omniscan gui`, PySide6/Qt: Library, Reader, Run, Models, Settings,
Import). It becomes **the** product UI. The browser viewer (`omniscan serve` + `webui/`, Svelte) stays as an
optional review/debug tool, but no new feature needs it.

Why the Qt app and not the browser (or Electron/Tauri around the web UI):
- **One install, no Node, no second server.** Today the browser path needs `omniscan serve` plus Node 24 and
  `npm run dev`. That is fine for a developer, not for a translation group.
- **Manual editing needs a real canvas.** Brush masks, box handles, zoom with exact scroll sync, drag-to-move text
  and live re-lettering are what `QGraphicsView` is built for; in a browser they mean re-implementing a paint
  program in JavaScript and round-tripping every stroke over HTTP.
- **Same code path as the CLI.** The GUI already calls a Qt-free service layer (`omniscan/gui/services/*`) that the
  CLI's `--json` commands share; the models run in-process on the user's GPU, no local web server in between.
- **Cross-platform for free.** Qt runs natively on Windows, macOS and Linux; PySide6 is LGPL (no licence fee, the
  app can stay under its own licence); `PySide6-Essentials` keeps the installer small (no bundled Chromium).
- Electron/Tauri would mean shipping a second runtime and keeping Python as a sidecar server anyway.

## Principles that hold for every milestone (owner, 2026-09-27)
- **GPU first, everywhere.** Image work (decode, slicing, masks, compositing), detection/OCR and inpainting run on the
  GPU on every vendor; the LLM runs on the GPU through Ollama (or in the cloud). The CPU is a fallback for machines
  without a supported GPU, never the default, and every CPU exception in the pipeline needs benchmark evidence
  (`docs/DECISIONS.md` lists the two accepted ones: DB post-processing and glyph rasterisation).
- **Fully optimised, and proven by measurement.** Speed claims come from `scripts/measure_run.py` and the
  `docs/benchmarks/` probes on real hardware, not from reasoning.
- **A full test suite that guards correctness and speed:**
  - the CPU suite (`pytest -m "not gpu"`) on all three OSes in CI, on every push;
  - the GPU suite (`pytest -m gpu`) on each vendor's real hardware, today the owner's AMD PC; the `gpu` fixture
    must accept `xpu`/`mps` devices once those machines exist to run it;
  - performance regression tests (a `perf` marker): a synthetic chapter per stage with a stored per-GPU baseline,
    failing when a stage gets slower than its budget, so an optimisation or a regression is visible in numbers;
  - mutation checks (`scripts/mutate.py`) for logic, and the end-to-end synthetic chapter
    (`tests/unit/test_e2e_synthetic.py`) viewed as images after any change to decoding or compositing.

## Milestones
Each one ends in something a user can run. Card IDs follow the existing scheme (`docs/tasks/<ID>.md`).

### X1 — Runs everywhere (this branch)
- Torch backend per machine: `rocm-gfx1201` (AMD RX 9070 series), `cuda` (NVIDIA), `xpu` (Intel Arc / Core Ultra),
  `mps` (Apple Silicon), `cpu` — one `uv sync --extra <name>`, `[tool.uv.conflicts]` refuses two at once.
- `resolve_device("auto")`: discrete CUDA/HIP GPU → discrete Intel XPU → Apple MPS → CPU; `select_device` and the
  VRAM manager work for `cuda` and `xpu` alike; `omniscan doctor` reports any vendor and warns (not fails) on CPU.
- CI on Ubuntu, Windows and macOS (`cpu` + `gui`), plus the owner's `rocm-gfx1201` build on Windows, plus a
  `uv lock --check` job.
- Next inside X1: other AMD cards (RDNA2/RDNA3 on Linux through PyTorch's ROCm index, other `device-gfx*` extras
  from AMD's index on Windows), and real-hardware smoke runs on NVIDIA/Intel/Apple machines when someone has one
  (the `gpu`-marked tests still assume a `cuda` device).

### X2 — Installers and first run
- One installer per OS (PyInstaller first, Nuitka evaluated): Windows `.exe`, macOS `.dmg` (signed later), Linux
  AppImage. The installer stays small; the matching torch runtime and the models download on first run after
  hardware detection (`omniscan hardware` already produces what that screen needs). Answers open questions B4/B11.
- First-run wizard in the desktop app: pick GPU/backend, data folder, download required models, connect Ollama
  (local or cloud) or pick a cloud translation provider.

### X3 — Translator Studio (the manual workbench)
A new "Studio" page in the desktop app, working on the pipeline's own artifacts (regions, OCR, translation,
inpaint patches, layout), so every manual change is just a better version of a stage output:
- Page canvas with the original, the cleaned page and the lettered result, side by side or as layers.
- Region list per page: fix boxes (add, split, merge, delete), reading order, region type (dialogue, SFX, sign).
- Text table: source text (editable OCR), machine candidates, the judge's pick, the final English; glossary hits and
  locked terms highlighted; per-line status (todo / edited / checked).
- Cleaning tools: brush and lasso to extend or shrink the inpaint mask, re-run LaMa on a selection, clone/heal.
- Lettering tools: move/resize text boxes, font/size/stroke/tilt per box, style presets, live re-render.
- Re-run any single stage for one page or one region; history with undo per chapter.
- Roles for groups later: translator, proofreader, cleaner, typesetter, quality check; a chapter moves between them.
- Export: CBZ/PDF/images as today, plus a project file a group can pass around.

### X4 — Shared data that improves the program (opt-in)
- **Nothing leaves the machine unless the user turns it on**, per series or globally, and every upload shows what
  is in it first. Default is off.
- What can be shared: corrected OCR lines with their crop, source → final translation pairs with context,
  glossary entries, region boxes the user fixed, lettering choices. Raw pages are never uploaded by default (they
  are usually someone else's copyright); crops only where the user allows it.
- A local "contribution" format first (JSON + crops, versioned schema in `core/`), exportable as a file; an upload
  service and its database come after, with accounts, licence terms for contributed data (e.g. CC BY or a
  contributor agreement), deletion on request, and moderation.
- How it feeds back: OCR fine-tuning and qualification sets (`eval/`), translation-model evaluation and prompt
  tuning, shared series glossaries, better defaults for detection and lettering. Improvements ship as model or
  config updates through the existing `models-v1`-style mirror and `omniscan update`.
- Needs the owner's decisions before building: hosting, licence of contributed data, whether uploads need an
  account. Tracked in `docs/OPEN_QUESTIONS.md` (section X).

### X5 — Sources and languages
- Use the downloader being built in the `manhwa-manga-downloader` repository as an optional source for
  `omniscan import` (a plug-in, not a hard dependency), keeping the existing legal boundary (no DRM platforms).
- More target languages than English once the Studio exists (the prompts and glossary are already per language).

## Order
X1 now, X3 next (it is what makes OmniScan a translator's tool and it produces the data X4 needs), X2 in parallel
once X1's CI is green, X4 after the owner answers its questions, X5 when the downloader is ready.

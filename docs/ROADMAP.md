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
- **One universal path first, vendor-specific fast paths where they win** (owner, 2026-09-27). Every feature works
  through the shared PyTorch path on every GPU. Where one vendor has something clearly faster (nvJPEG, MIOpen tuning,
  oneDNN, Metal), it is added as an optional backend behind the same interface (like `gpu.codec`), picked
  automatically for that hardware, and kept only when a benchmark shows the gain; the universal path stays the
  fallback and the reference the fast path is tested against.
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

- **Progress and time left everywhere.** Long actions show a progress bar and an estimate learned from this
  machine's measured seconds per page per stage (`omniscan.pipeline.eta`, remembered across runs per device and usage
  level). The Run page has it; model downloads, imports and the Studio's re-runs get the same.
- **Shaders / custom GPU kernels: only where profiling proves torch is the bottleneck.** The image work already runs
  as GPU compute through PyTorch on every vendor. Hand-written graphics shaders would mean four versions (CUDA, ROCm,
  Intel, Metal); the plan instead: GPU JPEG decode where the vendor ships one (nvJPEG via torchvision on NVIDIA,
  question F1 for AMD), Triton kernels (NVIDIA, AMD and Intel from one source) for hot spots a profile finds, and a
  GPU-rendered canvas (Qt RHI) for the Studio's zoom, layers and overlays.

## Milestones
Each one ends in something a user can run. Card IDs follow the existing scheme (`docs/tasks/<ID>.md`).

### X1 — Runs everywhere (this branch)
- Torch backend per machine: `rocm-gfx1201` (AMD RX 9070 series), `cuda` (NVIDIA), `xpu` (Intel Arc / Core Ultra),
  `mps` (Apple Silicon), `cpu` — one `uv sync --extra <name>`, `[tool.uv.conflicts]` refuses two at once.
- `resolve_device("auto")`: discrete CUDA/HIP GPU → discrete Intel XPU → Apple MPS → CPU; `select_device` and the
  VRAM manager work for `cuda` and `xpu` alike; `omniscan doctor` reports any vendor and warns (not fails) on CPU.
- CI on Ubuntu, Windows and macOS (`cpu` + `gui`), plus the owner's `rocm-gfx1201` build on Windows, plus a
  `uv lock --check` job.
- Hardware recognition (`omniscan hardware`, the Models page) plus a usage level the user picks on the Settings page:
  `gpu.usage = full | balanced | background` (`omniscan.hw.usage`). `balanced` leaves a quarter of the CPU and GPU
  memory to other apps; `background` uses a quarter of the CPU threads, half the GPU memory and a lower process
  priority. Next: a GPU duty cycle for `background` (short pauses between pages so games and video stay smooth),
  lower Ollama thread/GPU-layer settings per level, and a tray icon to switch levels while a run is going.
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
Built so far (2026-09-27): the Studio page with region boxes over the raw strip, editable source and English,
removing false boxes, the automatic QA check over the artifacts, `studio.json` edits that survive re-runs, the local
`corrections.jsonl` log and a one-click re-letter (typeset + export). Everything else below is still to do.

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
- Every manual correction is saved locally as a before/after record (the input X4 later shares, and a per-series
  memory the next chapter's translation reuses).
- Automatic QA pass over finished pages: re-read the lettered page to catch leftover source text, overflowing or
  clipped lettering, untranslated lines and typos; flagged pages open straight in the Studio.
- Context for the translator model: the page image and the speaker with it, plus per-character voice profiles
  (how each character talks) kept with the series glossary.
- Interchange with the tools groups already use: PSD export (layers: raw, clean, text) and LabelPlus import/export,
  plus opening other translation tools' project files (e.g. BallonsTranslator, manga-image-translator output) so
  their users switch in one click.
- Look (owner's design language): OLED black by default, a user-chosen accent colour, an optional light theme, and
  quick / standard / pro modes that show more of the app step by step (Settings → Appearance; `omniscan.gui.theme`).

### X4 — Shared data that improves the program (on by default, with an opt-out)
- **Owner's decision (2026-09-27):** full page images plus the corrections are shared **by default**, with a clear
  opt-out (first run and Settings, per series or globally). Users are told plainly what is sent and that it stays
  private. Safeguards: uploads are never published as a public dataset; file names, folder paths and image metadata
  (EXIF etc.) are stripped before upload; takedown and deletion requests are honoured; the consent text is exact;
  upload IP addresses count as personal data under the GDPR (privacy notice, retention limit).
- What is shared: the page images, corrected OCR lines, source → final translation pairs with context, glossary
  entries, region boxes the user fixed, lettering choices (the Studio's `corrections.jsonl` is the local source).
- A local "contribution" format first (JSON + crops, versioned schema in `core/`), exportable as a file; an upload
  service and its database come after, with accounts, licence terms for contributed data (e.g. CC BY or a
  contributor agreement), deletion on request, and moderation.
- How it feeds back: OCR fine-tuning and qualification sets (`eval/`), translation-model evaluation and prompt
  tuning, shared series glossaries, better defaults for detection and lettering. Improvements ship as model or
  config updates through the existing `models-v1`-style mirror and `omniscan update`.
- Still needs the owner's decisions before the upload service is built: hosting, licence of contributed data,
  whether uploads need an account. Tracked in `docs/OPEN_QUESTIONS.md` (X1-X3).

### X6 — A separate reader app (phone, tablet, desktop; later)
- Owner's decision (2026-09-27, "Both"): OmniScan keeps a basic reading mode (the Reader page's `Read` button:
  output only, full screen, keyboard paging, resume where you left off); a separate reader app for phones and
  tablets comes later and opens OmniScan's output as it is.
- The format it reads exists now: every exported chapter folder holds its images plus `omniscan-chapter.json`,
  and each series output folder an `omniscan-series.json` index (`docs/READER_FORMAT.md`). CBZ with
  `ComicInfo.xml` (`omniscan pack`) stays the format for other readers (Tachiyomi/Mihon, Komga, Kavita).
- Not started: the app itself (candidates: Qt for Android/iOS from this code base, or Flutter), sync of reading
  progress, and serving the library over the LAN.

### X5 — Sources and languages
- Use the downloader being built in the `manhwa-manga-downloader` repository as an optional source for
  `omniscan import` (a plug-in, not a hard dependency), keeping the existing legal boundary (no DRM platforms).
- More target languages than English once the Studio exists (the prompts and glossary are already per language).

## Order
Research on what translators and groups need (2026-09-27) ranks, for after X1: the page editor, the QA pass,
saving corrections, image + speaker context for translation, then PSD/LabelPlus interchange.
X1 now, X3 next (it is what makes OmniScan a translator's tool and it produces the data X4 needs), X2 in parallel
once X1's CI is green, X4 after the owner answers its questions, X5 when the downloader is ready.

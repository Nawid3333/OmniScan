# U3c — Desktop app shell: main window, library, reader, run page, `omniscan gui`

**Owner:** GLM builder · **Branch:** `U3c` · **Worktree:** `V:\OmniScan-wt\U3c` · run with `--max-turns 300`
**Start only after U3a is merged** (models view). Read `CLAUDE.md`, `docs/DECISIONS.md` ("Desktop app"), `docs/PRODUCT_SPEC.md` (§ desktop app / reader / pipeline modes), the merged GUI code (`src/omniscan/gui/strip_view.py`, `compare_view.py`, `models_view.py`, `services/`), `tests/gui/`, `scripts/gui_compare_demo.py`, and the contracts you will call: `pipeline/runner.py` (`run_pipeline`, `Gate`, `ReportFn`), `pipeline/preview.py`, `core/config.py` (`load_config`, `set_user_setting`, `set_series_setting`, `series_config`), `hw/detect.py` + `hw/assess.py`, `acquire/` (only to know it exists; the acquire page is card U3d).
This card is a goal, not a spec. You design the classes; the acceptance evidence at the bottom is what counts.

## Goal
`uv run --extra gui omniscan gui` opens ONE native window (PySide6, widgets only, no QtWebEngine) that a translator can use for the daily loop without the terminal:
1. **Library page** — series list (from `paths.library_root`) with chapter counts, and per chapter the state of each of the ten stages (from manifests: done / stale / failed / not run). Select a series → chapters table; double-click a chapter → Reader.
2. **Reader page** — the existing side-by-side raw | output compare view in a page with: chapter switcher (previous/next), zoom, "sync scroll" toggle, jump-to-slice, and a toggle to show only raw or only output. Reuse `compare_view`/`strip_view`; do not fork them.
3. **Run page** — choose series + chapters, a **mode**: *Full* (all stages), *Subset* (check-list of the ten stages, LaMa on/off, force), *Step* (`run_pipeline(mode="step")`: after every stage of the preview chapter the worker emits the `Preview` and blocks on the gate until the user presses Continue/Abort in the GUI), *Auto* (same as Full for now; leave a clearly marked hook, no dead code). Progress per chapter/stage, a log view (stage results/outcomes from `report`), Cancel (uses the `after_stage`/`RunAbortedError` mechanism through `run_pipeline`'s gate/report hooks; if `run_pipeline` lacks a cancel hook, stop and describe the needed core change in the report). One run at a time; the GUI stays responsive (QThread worker; the worker owns the GPU scheduler/client lifetime as the CLI does).
4. **Models page** — the U3a view, unchanged, mounted as a page.
5. **Settings page** — global settings (paths: library/work/output roots and models dir, GPU device incl. "auto" and the warm-up switch, OCR engine + models, translation profile, slicer strategy, filter on/off + threshold) written through `set_user_setting` (validated; show the validation error inline), plus a **per-series overrides** panel using `set_series_setting` (which sections are allowed is in `SERIES_SECTIONS`), plus a **Hardware panel** from `hw.detect`/`assess` (GPU, VRAM, RAM, backend, and the per-model warnings from `assess`).
6. Navigation: a left sidebar (Library, Reader, Run, Models, Settings), status bar (current device, running job), remember window geometry/last page (QSettings). Follows the system light/dark palette; no custom theme engine. Keyboard: Ctrl+1..5 switch pages.
7. `omniscan gui` (Typer command, optional-extra guarded: a clear one-line error and exit code 2 when PySide6 is missing) plus `python -m omniscan.gui`.

## Architecture rules (already decided)
- **Qt-free service layer** in `gui/services/` (library scan, run controller/worker protocol, settings service, hardware service) with plain-Python unit tests that need no Qt; Qt code only in widgets + thin QThread wrappers. Signals carry plain dataclasses.
- Long work never runs on the GUI thread. Blocking calls (`run_pipeline`) run in a worker; the step gate uses a `threading.Event` handshake.
- No new dependencies; `PySide6-Essentials` only; Windows/Linux/macOS safe (no Windows-only APIs).
- Do not edit `src/omniscan/core/**` (director-owned). If a hook is missing, stop and describe it in the report.
- Offscreen Qt has no fonts: tests set `QT_QPA_FONTDIR` as the existing GUI tests do.

## Acceptance evidence (put it in `docs/reports/U3c.md`)
1. `tests/gui/` tests (offscreen, marked as the existing GUI tests are) for every page: constructs, shows sample data from a synthetic library built in code, navigation works, Run page with a **fake** `run_pipeline` (injected) proves progress, cancel and the step gate handshake (Continue, Abort), Settings page writes and rejects invalid values (asserts the TOML on disk), per-series override lands in `series.toml`.
2. Plain unit tests for the service layer (no Qt import).
3. A screenshot script `scripts/gui_screenshots.py --out DIR` that builds a synthetic library (or, with `--series NAME`, uses a real one) and saves one PNG per page (offscreen `QWidget.grab()`), and you ran it. Look at your own screenshots: no clipped text, empty stretched panels or unreadable controls. **Leave the PNGs in `V:\OmniScan\data\screenshots\U3c\`** (gitignored dir; create it) — the director reviews them.
4. A real end-to-end check on this PC: `omniscan gui` opens with the real user config (library `V:\OmniScan\dataaws`, series `PepperCarrotKR`), the Library and Reader pages show "Episode 06" aligned raw|output, and a Run in *Subset* mode with only `typeset`+`export` completes from the GUI. State in the report what you verified and how (offscreen is fine).
5. `uv run --frozen pytest -q -m "not gpu"`, `ruff format . && ruff check .`, `pyright` clean. `docs/USER_GUIDE.md`: a short "Desktop app" section (how to start, the pages).

Commit early (`U3c: WIP shell`), final commit `U3c: desktop app shell`. If something is unclear, write the question in the report, commit and end.

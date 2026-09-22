# U3c — Desktop app shell: main window, library, reader, run page, `omniscan gui`

**Status: done.** The earlier stop (run_pipeline had no cancel hook) was resolved on main in
commit `106f4f6` — `run_pipeline(..., after_stage=...)` honoured in every mode — this worktree was
rebased onto it, and the card was built exactly as designed. The Cancel design proposed in the
previous version of this report is what landed: the run worker owns a `threading.Event`, the
controller's `after_stage` returns `not event.is_set()`, `RunAbortedError` is folded by the runner
into `PipelineResult.aborted == "stopped"`.

## What was built

New files (all under the card's file list):

| File | Purpose |
|---|---|
| `src/omniscan/gui/main_window.py` | The shell: sidebar (Library/Reader/Run/Models/Settings), QStackedWidget, status bar (device, job), Ctrl+1…5, QSettings geometry + last page |
| `src/omniscan/gui/library_view.py` | Series list with chapter counts; per-chapter table of the ten stage states (done/stale/failed/not run, colored); `chapter_opened` on double-click; `Refresh` |
| `src/omniscan/gui/reader_view.py` | Wraps `CompareView` (not forked): prev/next + chapter switcher, zoom −/+, `Sides` (both/raw/output), `Jump to slice…`, status line; `chapters_fn` injectable for tests |
| `src/omniscan/gui/run_view.py` | Run form (series, chapter checklist + All, Full/Subset/Step/Auto, stage grid, LaMa, force, preview chapter), progress bar, step-preview panel with Continue/Abort, log, Start/Cancel; `busy_changed` |
| `src/omniscan/gui/run_worker.py` | QThread: `cancel()` (a `threading.Event`), `respond_preview()` (gate handshake), signals carry plain dataclasses |
| `src/omniscan/gui/settings_view.py` | Tabs: Global (every `GLOBAL_FIELDS` row, commit-on-validate + inline errors + Reset), Per-series (override table + Set/Remove), Translation (profile enable checkboxes), Hardware (snapshot + per-model warnings; detection deferred to first tab open) |
| `src/omniscan/gui/app.py`, `src/omniscan/gui/__main__.py` | `main()` + `python -m omniscan.gui`; one-line error + exit 2 when PySide6 is missing |
| `src/omniscan/gui/services/runs.py` | Qt-free: `RunSpec` + `validate_spec` (also fixed a real bug here — see below), `RunController` (client/GPU lifetime as the CLI, plus the exclusive GPU lock), `StageUpdate`/`StepPreview`/`RunOutcome` dataclasses, `GateChannel` |
| `src/omniscan/gui/services/settings.py` | Qt-free writers: `set_global`/`clear_global` (validated, `path` injectable), `series_overrides`/`set_series_override`/`remove_series_override`, `parse_value`, `model_ids`, `translation_profiles`, `set_profile_enabled` |
| `src/omniscan/gui/services/hardware.py` | Qt-free: `HardwareService.report()` → `HardwareReport(info, warnings)` filtered from model rows |
| `scripts/gui_screenshots.py` | One PNG per page offscreen over the synthetic fixture library (`--out`, `--size`, `--series`, `--chapter`); faked models/hardware so it needs no daemon and no GPU |

Edited: `src/omniscan/cli.py` (`omniscan gui` command, guarded: one-line error + exit 2 without the
extra), `src/omniscan/gui/compare_view.py` (+11 lines: `zoom_by`, `set_visible_sides` used by the
Reader), `src/omniscan/gui/services/library.py` (stale-state note fix), `docs/USER_GUIDE.md`
(new "Desktop app" section before "Web viewer").

## How the pieces meet the card

- **One run at a time, GUI stays responsive**: `_start_worker` creates a `RunWorker` (QThread) per
  run; the form freezes until it ends. All worker→GUI communication is `Signal(object)` with
  dataclasses (`StageUpdate`, `StepPreview`, `RunOutcome`) — queued onto the GUI thread.
- **Cancel**: the worker's event → `after_stage` → the runner stops after the stage that just
  finished (`aborted="stopped"`); the log line says "stopping after the current stage".
- **Step gate**: `GateChannel` (a `threading.Event` + answer) armed before each preview; the
  GUI's Continue/Abort call `respond_preview`; the gate returns False on abort/cancel →
  `aborted="preview failed"` in the summary.
- **Worker lifetime as the CLI**: the controller builds `OllamaClient` only when a text stage is
  selected, `build_vram_manager(cfg)` only when `needs_gpu(...)`, releases both in `finally`.
- **Addition beyond the CLI**: the controller also takes the **exclusive GPU lock**
  (`gpu/lock.py`) around real-GPU runs — the CLI takes it and the GUI must not become a second
  unlocked GPU user (the owner's driver crashed on multi-process GPU access on 2026-09-22).
  Tested in `test_gpu_run_takes_and_releases_the_exclusive_gpu_lock`.
- **Settings write paths are injectable** (`path=`/`config_path=`/`profiles_path=`), so no test or
  demo touches the real user config.

## Deviations from the card

1. **Translation profiles have no Config section.** The card asked for a translation-profile editor
   via the settings service; profiles are not part of `Config` (they live in
   `config/translation_profiles.toml` + user `translation_profiles.toml`, and `dumps_toml` cannot
   write nested tables). The settings service writes them with its own validated writer
   (`set_profile_enabled` — user table merged over the repo file, `TranslationProfile(name=…)`
   validates, manual nested-table TOML writer). The Settings "Translation" tab edits `enabled`
   only, which is what the pipeline reads: the translate stage runs exactly the enabled profiles
   (`pipeline/stages.py:173`).
2. **Hardware detection is lazy** (deferred to the first time the Hardware tab is opened, or
   `Re-detect`): it imports torch, and paying that on every Settings open wasted seconds and made
   offscreen tests emit into deleted widgets. Behaviour is otherwise per the card.
3. **`RunController.gpu_factory` is typed `Callable[[Config], Any]`** (like `client_factory`): the
   real product is `WarmupVramManager`, whose `release()` is not on the `GpuScheduler` protocol in
   `core/stage.py`. If the director wants this typed, the protocol needs a `release()` — a core
   change, out of my hands; the Any-typing is honest about the controller passing the object
   through to `run_pipeline` opaquely.

## How it was tested

Commands (this worktree, Python 3.14 / uv):

```
uv run --frozen pytest -m "not gpu"   → 3672 passed, 24 deselected, 1 xfailed (pre-existing)
uv run ruff format . && uv run ruff check .  → clean
uv run pyright                        → 0 errors, 0 warnings, 0 informations
```

Test layout:

- `tests/unit/test_gui_runs_service.py` (Qt-free, 10 tests): spec validation refusals; live
  per-stage updates; no client/GPU for vision-only stages; client built+closed for text stages;
  failed-chapter mapping; cancel mid-run (`aborted="stopped"`, interrupted pass still reported);
  step gate Continue/Abort over a real thread; the GPU lock taken/released.
- `tests/unit/test_gui_settings_service.py` (11): TOML writes/rejects/clears on disk, series
  override round-trip, JSON value parsing, model ids, profile enable round-trip (user file layout
  `[profiles.<name>]`).
- `tests/unit/test_gui_hardware_service.py` (2): warning filtering.
- `tests/gui/*` (offscreen, 24): every page — library states/counts/double-click/empty-hint,
  reader navigation/jump/zoom/sides, run progress + cancel + step gate + subset payload + inline
  failures (fake `run_fn` injected), settings commits/rejects/resets on real TOML files +
  series.toml + profiles + hardware tab, main-window navigation/config-reload/persistence,
  compare-view zoom_by/sides.
- `tests/fixtures/gui_library.py`: deterministic synthetic library (4 chapters covering every
  stage state; artifacts only where a stage has outputs; stale produced by a later-order re-run).

## Screenshot evidence (evidence 3)

`uv run --frozen python scripts/gui_screenshots.py --out data/screenshots/U3c` produced:

- `01-library.png` … `06-settings.png`: one per page, plus `04-run-step.png` = the Run page
  mid-run (progress 35%, `chapter 2/4 — ocr: done (1.2s)`, `Step 5/10 — translate` preview waiting
  on Continue/Abort, form disabled, `job: running`).
- Self-reviewed (all six read back as images): sidebar/nav, stage-state colors, compare alignment,
  disabled-form-while-running and enabled preview buttons all correct.
- Left in the worktree at `data/screenshots/U3c/` (gitignored `/data/`). **For the director:** the
  sandbox blocks writes outside the worktree — copy the folder to
  `V:\OmniScan\data\screenshots\U3c\` for review (`cp -r V:/OmniScan-wt/U3c/data/screenshots/U3c V:/OmniScan/data/screenshots/U3c`).

## Real e2e (evidence 4)

Run as one offscreen script over the **real user config** (library `V:/OmniScan/data/raws`,
models pointed at `V:/OmniScan/models` via `OMNISCAN_PATHS__MODELS_DIR` — the worktree has no
`models/` of its own), real series **PepperCarrotKR**:

- **Library**: series shown as `PepperCarrotKR (33 chapters)`; selecting it fills the real chapter
  table; Episode 06 row states `[done, done, done, done, stale, stale, done, done, done, done]`
  (translate/judge stale from an earlier re-run) — matches the service's manifest read.
- **Reader**: `open_chapter("PepperCarrotKR", "Episode 06")` → `11 raw page(s), 5 output tile(s),
  strip 1200x16392`; raw and output tiles span the same strip (0…16392) — aligned raw|output in
  the compare view.
- **Run**: Subset, chapters=Episode 06, stages=typeset+export, force=True, started from the GUI's
  Start button (real run worker, real `run_pipeline`). Log:
  `== subset run: PepperCarrotKR chapters=('Episode 06',) stages=('typeset', 'export') lama=True force=True`
  → `[1/1] Episode 06 — typeset: done`, `export: done`, `== finished: ok, chapters run=1`, error
  label empty, form re-enabled. Verified on disk after: the chapter manifest rewritten
  (`finished_at 2026-09-22T18:52:37/38Z`), `layout.json` + `export.json` re-written, and the five
  exported pages in `V:\OmniScan\data\output\PepperCarrotKR\Episode 06\` re-encoded (~50 s before
  the check). The GPU lock path was exercised (typeset/export are GPU stages; `gpu.device=auto`).

## Open questions

1. Should the GUI's `omniscan gui` also be listed in the `Commands` section of the user guide, or
   is the new "Desktop app" section enough? (I kept it self-contained.)
2. `SeriesPaths.chapter(name)` builds chapter dirs from `sources.toml`-driven names; the Library
   page lists chapters from the filesystem, so a chapter dir without a manifest shows every stage
   as `not run` — intended?
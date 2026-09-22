# U3e — Import/upload page: archives, JPEG normalisation, chapter preview and editing

**Owner:** GLM builder · **Branch:** `U3e` · **Worktree:** `V:\OmniScan-wt\U3e` · run with `--max-turns 300`
Read `CLAUDE.md` first. Then `src/omniscan/importer/plan.py` and `execute.py` (the existing backend: `plan_import`, `execute_import`, the three source shapes — single chapter, folder-of-folders, flat dump grouped by an explicit chapter marker in the filename), `src/omniscan/acquire/drm.py` (`NOTICE` — reuse it, do not write a new string), `src/omniscan/hw/detect.py` (`detect_hardware` — reuse for the notice's context line, see below), the existing standalone GUI pages and their patterns: `src/omniscan/gui/models_view.py` + `src/omniscan/gui/services/models.py` (the Qt-free service / Qt view split, and how a plan is shown before it is executed), `src/omniscan/gui/workers.py` (`run_task`, `WorkerSignals` — the QThreadPool worker pattern; reuse it, do not invent a second one), `scripts/gui_models_demo.py` (the offscreen screenshot pattern), `tests/gui/test_models_view.py` (test conventions: `QT_QPA_PLATFORM=offscreen`, `QT_QPA_FONTDIR` on Windows).
This card is a goal, not a spec: you design the classes; the acceptance evidence at the bottom is what counts.

## Why
The owner wants to be able to hand the program a manhwa/manga they already have (a folder of scans, or a `.zip`/`.cbz` archive) and have it become a properly-shaped library series, without touching a terminal: a visual page with a preview of what will be imported, the ability to see and correct how pages were grouped into chapters before committing, and every image normalised to JPEG (the format the rest of the pipeline is built around) with a clear notice when that conversion happens.

## Goal
A standalone `ImportView` widget (built and demoed exactly like `ModelsView` was in card U3a — not wired into any main window; there isn't one yet, card U3c is building it separately) that:
1. **Source picker**: a folder or a `.zip`/`.cbz` file, via a "Browse…" dialog and drag-and-drop onto the widget.
2. **Plan preview**: runs the existing `plan_import` (extended for archives, see below) and shows the result before anything is written — detected series name (editable), one row per detected chapter (editable name), its page count, and its files in order. The user must be able to **fix a wrong grouping** before committing: move a page to a different chapter, reorder pages within a chapter, rename or merge/split a chapter. Decide the exact editing UI (a tree/table with drag-and-drop, or simpler move-up/move-down + a chapter picker per file — your call, but it must be genuinely usable, not a JSON text box).
3. **Format normalisation**: any input image that is not already a JPEG gets converted (decode with PIL, re-encode as JPEG, quality 95 — match the project's existing export convention, `docs/OPEN_QUESTIONS.md` E1) instead of being copied byte-for-byte. The preview must show **which files will be converted** before the user commits (a count, and the per-file list on request — do not bury it). Say why in one honest sentence, e.g. "converted to JPEG for consistent, fast processing" — **do not claim a hardware-accelerated decoder is active**: today only the CPU `turbo` (libjpeg-turbo) codec is implemented (`src/omniscan/gpu/codec/select.py`); rocJPEG/nvJPEG/other vendor-specific hardware decoders are card C2, separately tracked and not built yet. You may show the machine's detected GPU vendor/backend as *context* (via `hw.detect.detect_hardware`) next to the notice, but the wording must stay accurate about what runs today.
4. **Commit**: a Copy/Move toggle, an "Import" button that runs `execute_import` (extended to actually perform the JPEG conversion, not just copy) on a worker thread with a progress bar (files processed / total), and a result summary (chapters written, files copied, files converted, duplicates skipped). The `acquire.drm.NOTICE` string ("You are responsible for having the right to download and process this content.") is shown somewhere visible on the page, reusing the constant.

## Backend changes (files you may modify)
- `src/omniscan/importer/plan.py` / `execute.py`: extend for
  - **Archives**: a `.zip` or `.cbz` file as the source (today `plan_import` requires an existing directory — decide whether to extract to a temp directory first and reuse the existing folder logic unchanged, or read the archive listing directly; the former is simpler and lower-risk, prefer it unless you have a measured reason not to). Nested folders inside the archive should map to the same three shapes the folder path already supports.
  - **JPEG conversion**: `ImportPlanItem`/`ImportResult` (or new fields on them) need to carry which files will be/were converted, not just copied — extend the dataclasses (they are plain frozen dataclasses, not `core/**`, so this is in scope). `execute_import` performs the actual conversion (PIL open → convert("RGB") if needed → save as JPEG quality 95) instead of `shutil.copy2` for non-JPEG sources; the existing idempotent/fail-fast/duplicate-detection behaviour must still hold (a re-run must not re-convert/re-copy a file already correctly in place).
  - Keep the existing three-shape planning logic (folder-of-folders / single chapter / flat dump) exactly as-is in behaviour for already-JPEG folder sources — this card must not change existing CLI import behaviour for users who don't need conversion or archives (`tests/unit/test_importer_*.py`, wherever they are, must stay green with no changes needed to their assertions, only extensions).
- `src/omniscan/cli.py`: `cmd_import`'s `source` argument currently requires an existing directory (`typer.Argument(exists=True, file_okay=False)`) — relax it so a `.zip`/`.cbz` file is also accepted, wired to the same extended `plan_import`. Add a note to the CLI's dry-run/normal output about conversions (file count), matching the existing output style.

## GUI files you may create
- `src/omniscan/gui/import_view.py` (`ImportView(QWidget)`), `src/omniscan/gui/services/importer.py` (Qt-free: wraps `plan_import`/archive handling/editing helpers/`execute_import` with a progress callback), `scripts/gui_import_demo.py` (offscreen screenshot script, same shape as `gui_models_demo.py`), `tests/gui/test_import_view.py`, `tests/unit/test_gui_importer_service.py`, extend `tests/unit/test_importer_plan.py`/`test_importer_execute.py` (or wherever the existing importer tests live — check the actual filenames, don't guess).
Do not create or wire a main window / navigation shell — that is card U3c, running separately right now; a file-level clash there is exactly what both cards must avoid. Do not edit `src/omniscan/core/**` (director-owned).

## Constraints
- No new dependency for archive reading (`zipfile` is stdlib) or JPEG conversion (Pillow is already a dependency).
- CPU-only; this card has nothing to do with GPU decode paths — do not touch `src/omniscan/gpu/**`.
- Tests use synthetic fixtures generated in code (small PNG/BMP/JPEG images built with PIL, a zip built with `zipfile` in a fixture) — never real manga/manhwa content.
- Offscreen Qt has no fonts: tests/scripts set `QT_QPA_FONTDIR` as the existing GUI tests do.

## Acceptance evidence (put it in `docs/reports/U3e.md`)
1. Backend tests: archive sources (zip and cbz extension, all three shapes inside), JPEG conversion (PNG/BMP/GIF-first-frame input → JPEG output, byte-identical pixels within JPEG's own lossy tolerance is not the bar — correct format/mode/dimensions is), idempotent re-run (already-converted file is not re-converted, hash-checked like the existing duplicate logic), mixed archive with both convertible and already-JPEG files, a corrupt/unreadable archive gives a clear `ImportPlanError`, existing plain-folder-of-JPEGs behaviour unchanged (existing tests still pass unmodified in their assertions).
2. GUI tests (offscreen) for `ImportView`: loads a synthetic plan, shows the chapter/file breakdown, editing a grouping (move a file to a different chapter / reorder / rename) updates the plan actually used at commit time, the conversion notice appears exactly when there is a non-JPEG file and not otherwise, the `NOTICE` string is present verbatim, Import runs the worker and shows progress, a failure (e.g. `ImportPlanError`) surfaces in the UI instead of crashing it.
3. Plain unit tests for the service layer (no Qt import).
4. A screenshot script run (`scripts/gui_import_demo.py --screenshot OUT.png`) against a synthetic source with a mix of JPEG/PNG files across 2–3 chapters, showing the plan preview with the conversion notice visible — leave the PNG in `V:\OmniScan\data\screenshots\U3e\` (gitignored; create it) for the director to look at.
5. A real check on this machine: build a small real folder (synthetic images, via a one-off script — not committed) with a `.zip` archive containing 2 chapters (one all-JPEG, one with a couple of PNGs), run it through the CLI `omniscan import` end to end, and through the GUI service layer, and confirm both land the same files in `data/library` (or wherever the test config points) with the PNGs converted. State exactly what you ran and its output in the report.
6. `uv run --frozen pytest -q -m "not gpu"`, `ruff format . && ruff check .`, `pyright` clean. `docs/USER_GUIDE.md`: extend the existing import section (or add one) covering archives, conversion, and the GUI page.

## Out of scope
Wiring `ImportView` into a main window (U3c's job once both are merged — the director will do the small connecting piece, or write a tiny follow-up card). Any hardware-accelerated JPEG decoder (rocJPEG/nvJPEG/etc. — card C2, separately tracked, deliberately deferred). Downloading or fetching content from any URL or site — this card only ever operates on a local folder or archive file the user already has on disk. Chapter *matching across two different series* (that's card CM1, a different problem — this card's "chapter grouping" is about correctly grouping one source's own pages into that source's own chapters, not aligning two languages' chapter sets against each other).

## Commands to run before finishing
```bash
uv run --frozen pytest tests/gui/test_import_view.py tests/unit/test_gui_importer_service.py tests/unit/test_importer_plan.py tests/unit/test_importer_execute.py tests/unit/test_docs.py -q
uv run --frozen pytest -q -m "not gpu"
uv run --frozen ruff format . && uv run --frozen ruff check .
uv run --frozen pyright
```

## Report
`docs/reports/U3e.md`: Changes, Tests (commands + results), Deviations, Questions. Commit early (`U3e: WIP archive + conversion backend`), final commit `U3e: import/upload page`. You have 300 tool calls. If anything is unclear, stop, write the question in the report, commit and end.

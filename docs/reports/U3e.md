# U3e — Import/upload page: archives, JPEG normalisation, chapter editing

## Changes

**Backend (archives + JPEG conversion)**

- `src/omniscan/importer/plan.py` — `plan_import` now accepts `.zip`/`.cbz` sources
  (`ARCHIVE_SUFFIXES`): the archive is extracted member-by-member into a
  `tempfile.TemporaryDirectory(prefix="omniscan-import-")`, a lone top-level wrapper folder is
  descended into (`_unwrap`), and the extracted folder goes through the *unchanged* three-shape
  folder logic (single chapter / folder-of-chapters / flat dump with explicit markers). The series
  defaults to the archive's stem. The extraction handle is attached to the frozen plan
  (`ImportPlan.archive`, `ImportPlan.temp_dir`) with `ImportPlan.cleanup()`; `dataclasses.replace`
  carries it through edits, and a failed plan never leaks it. Zip-slip member names (absolute, `..`,
  drive) are refused; `BadZipFile`/encrypted (`RuntimeError`)/`OSError` become a clear
  `ImportPlanError`. Also added `JPEG_SUFFIXES` and `files_to_convert(plan)` (every image whose
  suffix is not `.jpg`/`.jpeg`).
- `src/omniscan/importer/execute.py` — files whose suffix is not `.jpg`/`.jpeg` are no longer
  byte-copied: they are decoded (PIL), EXIF-rotated, alpha-flattened over white and re-encoded as a
  plain baseline RGB JPEG at quality 95 (`subsampling=0, optimize=True`) — byte-for-byte the same
  convention as `ingest/convert.py`, so `needs_conversion` is False for every imported file. The
  destination is `<stem>.jpg`; an idempotent re-run hash-checks the freshly encoded bytes against
  what is already in place and skips; a different-content destination fails fast. `ImportResult`
  gained `files_converted` and `converted` (source names, in plan order); `move=True` consumes
  converted sources too. All-JPEG imports stay byte-identical copies (existing behaviour).
- `src/omniscan/cli.py` — `source` accepts a `.zip`/`.cbz` archive (a non-archive file is refused
  with exit 2); `--dry-run` prints `import:   N file(s) will be converted to JPEG` when N > 0; the
  commit summary appends `", N file(s) converted to JPEG"` when N > 0; the plan's extraction is
  cleaned up in a `finally`.

**GUI**

- `src/omniscan/gui/services/importer.py` (new, Qt-free) — `ImporterService` wraps
  `plan_import`/`execute_import` with the configured library root and a `hardware_line()` that says
  what was detected *and what actually runs today* ("conversion runs on the CPU (libjpeg-turbo)" —
  no hardware-decode claim; rocJPEG is card C2). `conversion_text(plan)` is the one-line notice
  (`"... will be converted to JPEG for consistent, fast processing."`, `CONVERSION_REASON`). The
  plan-editing helpers are plain frozen-in/frozen-out functions: `set_series`, `rename_chapter`,
  `move_file` (emptied chapters vanish), `merge_chapters` (`ValueError` on self-merge),
  `split_chapter` (tail becomes a new chapter named after the plan's top chapter number + 1).
- `src/omniscan/gui/import_view.py` (new) — `ImportView(QWidget)`: source picker (Browse folder /
  Browse archive dialogs plus a drag-and-drop seam that accepts local folders and `.zip`/`.cbz`
  files), editable plan preview (bold chapter rows with page counts, page rows with format, pages
  to be re-encoded highlighted and flagged "→ JPEG" with a quality-95 tooltip), editable series and
  chapter names, grouping fixes via Target combo + Move here (page move; chapter selection = merge)
  + Move up/Move down + Split here, a "Which files?" per-file conversion list in the details pane
  (warnings always shown), the `acquire.drm.NOTICE` string verbatim at the top, and the commit:
  copy/move toggle, `gui.workers` worker for every service call, determinate progress, result
  summary `Imported: N chapter(s), N copied, N converted, N duplicate(s) skipped`, failures
  surfaced as red status text instead of crashes, and duplicate chapter names blocked before the
  commit. "Move instead of copy" is disabled for archive sources (it would only delete throwaway
  extracted copies).
- `scripts/gui_import_demo.py` (new) — standalone demo after `gui_models_demo.py`; without
  `--source` it generates a synthetic 3-chapter JPEG/PNG `.cbz` in a temp dir;
  `--screenshot OUT.png` renders offscreen.

**Tests** — `tests/unit/test_importer_plan.py` (+13 archive tests), `tests/unit/test_importer_execute.py`
(+10 conversion tests), `tests/unit/test_gui_importer_service.py` (new, 20, imports no Qt),
`tests/gui/test_import_view.py` (new, 20, offscreen).

**Docs** — `docs/USER_GUIDE.md` import section: archives (`.zip`/`.cbz`, wrapper folder, series
from the stem), the JPEG conversion and its notice wording ("runs on the CPU (libjpeg-turbo); no
hardware JPEG decoder is wired up"), the new summary line, corrupt/encrypted/conflict exit codes,
and a paragraph on the GUI import page.

## Tests (commands + results)

`uv run --frozen pytest tests/gui/test_import_view.py tests/unit/test_gui_importer_service.py tests/unit/test_importer_plan.py tests/unit/test_importer_execute.py tests/unit/test_docs.py -q`
→ **92 passed** (20 GUI + 20 service + 27 planning + 14 execute + 11 docs).

`uv run --frozen pytest -m "not gpu"` → **3681 passed, 24 deselected, 1 xfailed** (the xfail is
pre-existing, `test_hw_hf_mutation_gaps.py`). All pre-existing importer/folder tests pass with no
assertion changes — the three folder shapes behave exactly as before.

`uv run --frozen ruff format --check .` → 338 files already formatted. `uv run --frozen ruff check .`
→ All checks passed. `uv run --frozen pyright` → 0 errors, 0 warnings.

### Acceptance 1 — backend

- Archives, all three shapes: `test_zip_folder_of_folders`, `test_cbz_single_chapter_folder`,
  `test_zip_flat_dump_grouped_by_filename_marker`, `test_zip_wrapper_folder_is_unwrapped`,
  `test_zip_series_option_overrides_stem`, `test_zip_warns_about_non_image_members`.
- Failure modes: `test_corrupt_zip_raises_clear_error` ("can't read archive"),
  `test_missing_archive_raises`, `test_empty_zip_raises`, `test_zip_slip_member_name_is_refused`
  ("unsafe member name").
- Extraction lifetime: `test_plan_cleanup_removes_extraction` (double `cleanup()` fine),
  `test_edited_plan_keeps_the_extraction_alive` (edits preserve the temp dir; the moved file is
  still readable for the commit).
- Conversion: `test_png_is_converted_to_jpeg` (dest `p1.jpg`, `format == "JPEG"`, `mode == "RGB"`,
  size preserved, and ingest's `needs_conversion(dest) is False` — the pipeline bar), JPEGs stay
  byte-copies; `test_alpha_flattened_over_white` (RGBA flattened over white, ±2 of the flat
  colours); `test_gif_uses_first_frame` (black first frame, not the white second);
  `test_bmp_converted_to_jpeg`; mixed convertible/already-JPEG covered in
  `test_png_is_converted_to_jpeg` and the progress test.
- Idempotence, hash-checked: `test_rerun_does_not_reconvert` (second run converts 0, skips 1, the
  in-place bytes are unchanged — the freshly encoded bytes match what the first run wrote),
  plus the pre-existing folder idempotence tests unmodified.
- Conflicts: `test_conflicting_converted_destination_raises`,
  `test_png_and_jpeg_same_stem_conflict_raises` (fail-fast, nothing overwritten),
  `test_corrupt_image_fails_fast_with_clear_message` ("can't convert … to JPEG").
- Move: `test_move_converts_and_deletes_source`.

### Acceptance 2 — offscreen GUI tests (`tests/gui/test_import_view.py`)

Loads a synthetic plan through the real worker (`planned_view` helper pumps until the tree fills)
and shows the chapter/file breakdown (`test_plan_loads_the_chapter_file_breakdown`, including page
counts, format column, "→ JPEG" flags and the selection defaults). Grouping edits update the plan
that actually reaches the commit: move page to another chapter (`test_move_page_to_another_chapter_updates_the_commit_plan`
— the recorded `service.executed[0]` matches the edited grouping), reorder
(`test_reorder_pages_within_a_chapter`), rename (`test_rename_chapter_updates_the_commit_plan`),
merge (`test_merge_appends_the_selected_chapter_into_the_target`), split
(`test_split_starts_a_new_chapter_at_the_selected_page`). The conversion notice appears exactly
when a non-JPEG file exists and not otherwise (`test_conversion_notice_only_when_non_jpeg_present`,
exact string `"2 of 5 file(s) will be converted to JPEG for consistent, fast processing."`); the
per-file list toggles via "Which files?" (`test_which_files_lists_the_conversions_and_warnings`).
`NOTICE` verbatim (`test_drm_notice_is_verbatim`). Import runs on a worker thread with progress
(`test_import_runs_on_a_worker_with_progress_and_summary` asserts a non-main thread, determinate
progress 2/5, and the summary line; `test_move_toggle_reaches_the_worker`). Failures surface
without crashing: import failure (`test_import_failure_surfaces_red_instead_of_crashing` — red
`ImportPlanError: …`, buttons re-enabled) and plan failure
(`test_plan_failure_surfaces_red_and_clears_the_preview` — tree cleared, retry possible).
Extras: `test_archive_plan_disables_the_move_toggle`, `test_series_edit_updates_the_commit_plan`,
`test_duplicate_chapter_names_block_the_import`, `test_split_at_the_first_page_is_refused`,
`test_drop_seam_accepts_folders_and_archives_only`, and
`test_real_service_end_to_end_through_the_view` (real `ImporterService`, real synthetic files:
plan → commit → the PNG lands as `Chapter 2/1.jpg`, converted).

### Acceptance 3 — Qt-free service tests

`tests/unit/test_gui_importer_service.py` (20 tests) imports no Qt: service wrapping, configured
library root, progress reporting, `hardware_line` (GPU present and absent, asserting the honest
"CPU (libjpeg-turbo)" wording), `conversion_text` exact string, every editing helper (including
order preservation, emptied-chapter vanishing, split naming, self-merge `ValueError`), archive
extraction surviving edits, and an edited plan committed through both the service and plain
`execute_import`.

### Acceptance 4 — screenshot

`uv run --frozen python scripts/gui_import_demo.py --screenshot "V:/OmniScan/data/screenshots/U3e/import_view.png"`
→ exit 0; PNG left at `V:\OmniScan\data\screenshots\U3e\import_view.png` (gitignored `/data/`).
The source is the script's synthetic 3-chapter `.cbz` (7 pages, 2 PNGs). The shot shows the NOTICE
verbatim, the context line `machine: AMD Radeon RX 9070 XT · 15.9 GB · rocm — conversion runs on
the CPU (libjpeg-turbo)`, the editable plan tree with the two PNG rows highlighted "→ JPEG", the
notice "2 of 7 file(s) will be converted to JPEG for consistent, fast processing.", the details
pane with the per-file conversion list, the move toggle disabled for an archive source, and
"Planned Demo Series: 3 chapter(s), 7 page(s)". I opened the PNG and checked all of the above.

### Acceptance 5 — real machine check (CLI and GUI service on the same archive)

One-off script (not committed, deleted after): built `_e2e/E2E Series.zip` with real PIL images —
`Chapter 1/1.jpg`, `Chapter 1/2.jpg` (JPEG q95) and `Chapter 2/1.png`, `Chapter 2/2.png` (PNG) — then
ran both paths and byte-compared the results. The library root was pointed inside the worktree via
`OMNISCAN_PATHS__LIBRARY_ROOT`; the CLI was invoked through the real typer app in-process (same
code path as the `omniscan` console script) because the session sandbox rejected `env VAR=...`
command prefixes.

What was run and its output:

```
$ omniscan import V:\OmniScan-wt\U3e\_e2e\E2E Series.zip --dry-run
import: plan for series 'E2E Series' · 2 chapter(s)
import:   Chapter 1: 2 file(s)
import:   Chapter 2: 2 file(s)
import:   2 file(s) will be converted to JPEG
[exit 0]
$ omniscan import V:\OmniScan-wt\U3e\_e2e\E2E Series.zip
import: E2E Series/Chapter 1
import: E2E Series/Chapter 2
import: 2 file(s) copied, 2 file(s) converted to JPEG, 0 duplicate file(s) skipped
[exit 0]
```

GUI service (`ImporterService.plan` → `execute`, progress printed per file):

```
GUI service plan: series='E2E Series' chapters=['Chapter 1', 'Chapter 2']
GUI service conversion preview: '2 of 4 file(s) will be converted to JPEG for consistent, fast processing.'
  progress 1/4 … 4/4
GUI service result: copied=2 converted=2 skipped=0 converted_names=['1.png', '2.png']
```

Comparison: both libraries contain exactly `E2E Series/Chapter 1/{1,2}.jpg` and
`E2E Series/Chapter 2/{1,2}.jpg`; every file is byte-identical between the CLI run and the GUI
service run; PIL verifies all four as `JPEG RGB (40, 56)` — the PNGs converted in both.

### Acceptance 6 — full checks and user guide

`uv run --frozen pytest -m "not gpu"` → 3681 passed, 1 pre-existing xfailed. Ruff format/check and
pyright clean (numbers above). `docs/USER_GUIDE.md`: the `omniscan import` section now covers
archives (extraction, wrapper folder, series from the archive name), the JPEG conversion with the
honest CPU wording, the dry-run/summary notes, the widened failure list, a
`uv run omniscan import ~/Downloads/DemoSeries.zip` example, and the GUI import page.

## Deviations

- **Conversion predicate is suffix-based** (`convert` iff suffix ∉ {`.jpg`, `.jpeg`}), not ingest's
  `needs_conversion`: the existing CLI tests feed fake non-image `.jpg` bytes that PIL cannot
  decode, and the card requires their behaviour to stay byte-identical and unchanged. The bar is
  still met: anything a plan accepts with a non-JPEG suffix is decoded and re-encoded so
  `needs_conversion` is False at ingest; a real JPEG under a wrong-name suffix cannot exist in
  practice because `list_images` only accepts known image suffixes.
- **Grouping edits use buttons + combo + line edits** (Move here / Move up / Move down / Split
  here / Target / chapter name field) rather than drag-and-drop rows. The card allowed any
  genuinely usable editing; this is fully offscreen-testable and keeps the edit operations
  one-click reversible. Source picking does support drag-and-drop.
- **Move toggle disabled for archive sources**: with files extracted to a temp dir, "move" would
  only delete throwaway copies, so offering it would be a lie; folder sources keep copy/move.
- The CLI now prints the result summary inside the `try` (before the `finally` cleanup) instead of
  after it — identical output, but provably-only-on-success for pyright.
- `split_chapter` names the new chapter `Chapter <top + 1>` (highest chapter number in the plan,
  +1) as a predictable default the user can rename in the adjacent field.

## Questions

None blocking. Two notes for the director: (1) the GUI context line deliberately says conversion
runs on the CPU (libjpeg-turbo); when card C2 (rocJPEG) lands, both the line and the
`CONVERSION_REASON` string in `gui/services/importer.py` are the two places to update. (2) The
demo/screenshot workflow (`scripts/gui_import_demo.py`) builds its archive in code and never
touches real manga, so it is safe to re-run anytime.

## Review addendum (director)

Verified independently before merge: `ruff format --check`/`ruff check`/`pyright` clean, full
`pytest -m "not gpu"` green (3714 passed pre-fix), the screenshot script re-run for real (fonts
pointed at `C:\Windows\Fonts`, not the guessed path in the card) — the page matches the report's
description exactly (DRM notice, honest CPU/libjpeg-turbo wording, editable plan tree with
per-page conversion highlights, grouping controls, per-file conversion list, progress/summary).

Found and fixed a **zip-slip / path-traversal vulnerability** in `_extract_member`
(`importer/plan.py`) before merging: the original guard (`name.is_absolute() or ".." in
name.parts or name.drive`, checked on a `PurePosixPath`) does not actually hold on Windows —
`PurePosixPath.drive` is always empty (POSIX paths have no drive concept), so a member name like
`C:/evil/file.txt` passes the check silently; separately, a name with an embedded backslash
(`..\evil.txt`) stays one opaque part under `PurePosixPath` (which never splits on `\`) so `".."
in name.parts` also misses it, yet the backslash *does* get split once the name is joined onto a
real Windows `Path`, so the write still escapes the extraction directory. Confirmed both bypasses
empirically (`dest.joinpath(*name.parts)` landing outside `dest`) before fixing.

Fix: reject any member name containing a backslash or matching a Windows drive prefix
(`^[A-Za-z]:`) outright — neither is ever legitimate in a ZIP member name — and, as a second,
independent guard, resolve the joined target and verify it actually sits inside the (also
resolved) extraction directory, rather than trying to enumerate every unsafe name pattern.
Added `test_zip_slip_drive_letter_member_name_is_refused` and
`test_zip_slip_backslash_member_name_is_refused` to `tests/unit/test_importer_plan.py`
(29 tests now pass in that file, all deterministic regardless of which physical drive the OS
temp directory happens to live on). Re-ran the full suite after the fix: 3716 passed, ruff/pyright
still clean. No other files touched.
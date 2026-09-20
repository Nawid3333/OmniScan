# U3a — Models rows + service + workers + models view

## Changes

| File | What |
| --- | --- |
| `src/omniscan/models/rows.py` (new) | Qt-free row builder shared by CLI and GUI: `ModelRow` (frozen slots dataclass, 20 fields), `build_rows(cfg, *, role, lang, catalog, hardware, ollama_names) -> (rows, hardware)`, `row_to_json(row)` byte-identical to the CLI's per-model JSON object, and `ollama_model_names(url)` (moved from `cli._ollama_model_names`). Heavy imports (`assess`, `detect_hardware`, `install_path`, `model_status`) stay lazy inside `build_rows` so tests can patch them. |
| `src/omniscan/cli.py` | `models_list` rebuilt on `build_rows`/`row_to_json`; module alias `from omniscan.models.rows import ollama_model_names as _ollama_model_names` keeps the old monkeypatch target alive. Output is byte-identical: all 27 `test_models_cli.py` tests pass unchanged. |
| `src/omniscan/gui/services/models.py` (new) | Qt-free `ModelsService`: `rows()` (build_rows + load_catalog + injected `ollama_names`/`hardware` providers), `download(model_id, on_progress)` (adapts `download_model`'s `(id, done, total)` progress shape to `(done, total)`), `remove(model_id)`, `download_required(on_progress)` (skips installed/cloud/unknown, returns `(id, error-or-None)` per required missing/corrupt model, `ModelDownloadError` caught into the result), `ValueError(f"unknown model {model_id!r}")` for unknown ids. |
| `src/omniscan/gui/workers.py` (new) | `WorkerSignals` (`progress(int,int)` with total 0 when unknown, `finished(object)`, `failed(str)`), `TaskWorker(QRunnable)` (never raises out of `run()`; any exception becomes `failed("<Type>: <msg>")`), `run_task(fn, *, pool=None) -> WorkerSignals`. See Deviations for the lifetime mechanics. |
| `src/omniscan/gui/models_view.py` (new) | `ModelsView(QWidget)`: hardware header line, role/language combos, installed-only checkbox, search box, 7-column table (`Model/Role/Languages/Size/Fit/Status/""`), hidden progress bar, status label, "Download required models" button, details pane. Fit backgrounds (slow/warn/incompatible), bold installed status, "(required)" suffix, per-row Download/Remove buttons (none for cloud), confirm hook (`confirm(title, text) -> bool`) only for incompatible downloads and every remove, all service calls except `rows()` on workers, required flow with `Downloading <id> (k of n)` status and `<ok> installed, <failed> failed` summary, failures listed in the details pane, `service.rows` exceptions land in the status label instead of crashing. |
| `scripts/gui_models_demo.py` (new) | Offscreen demo/screenshot entry point (mirrors `gui_compare_demo.py`): `--screenshot OUT.png --size 1300x800 --role ROLE`, sets `QT_QPA_PLATFORM=offscreen` (+ `QT_QPA_FONTDIR` on Windows) before importing Qt, grabs the widget to a PNG, exits 0. |
| `tests/unit/test_models_rows.py` (new) | 12 tests: row fields/filters/status/family, `row_to_json` key set and order, CLI JSON parity against the real `cli.models_list` payload (same monkeypatch points as the CLI tests), `ollama_model_names` error mapping. |
| `tests/unit/test_gui_models_service.py` (new) | 12 Qt-free tests: rows filters, default provider wiring, download entry/progress adaptation, remove kwargs, unknown-id `ValueError`, `ModelDownloadError` propagation, `download_required` ordering/skip/failure and the ollama missing/pulled/unknown/tag-present matrix. |
| `tests/gui/test_workers.py` (new) | 5 offscreen tests: `finished(7)`, ordered progress with `total 0` for `None`, exception → `failed("ValueError: boom")` without crashing, two concurrent tasks on a 2-thread pool, explicit pool used. |
| `tests/gui/test_models_view.py` (new) | 22 offscreen tests against a `FakeService` (7 fixed rows: ok installed, missing, slow, warn, incompatible, cloud, required-missing): exact header/cell texts, tooltips, fit backgrounds, bold status, button captions per status (none for cloud), role/lang/installed-only/search filters individually and combined, details text for the selected row (incl. the incompatible line and installed path), download on a worker thread with determinate/indeterminate progress and refresh after install, red failure status, incompatible confirm gate both ways, warn row never asks, remove confirm gate both ways, nothing-to-remove, required button state + `(k of n)` label + summary + failure listing, all action buttons disabled while a task runs, `rows()` exception in the status label. |

## Tests

Commands and results (all run in `V:\OmniScan-wt\U3a`):

- `uv run --frozen pytest tests/unit/test_models_rows.py tests/unit/test_gui_models_service.py tests/unit/test_models_cli.py tests/gui -q` → **101 passed** (27 of those are the unchanged `test_models_cli.py` CLI tests).
- `uv run --frozen pytest -q -m "not gpu"` → run by the director (see the addendum)
- `uv run --frozen pytest -q` (full suite incl. gpu-marked) → run by the director (see the addendum)
- `uv run --frozen ruff format . && uv run --frozen ruff check .` → run by the director (see the addendum)
- `uv run --frozen pyright` → run by the director (see the addendum)
- `uv run --frozen python scripts/gui_models_demo.py --screenshot "$TEMP/omniscan-u3a-demo.png" --role detector` → exit 0, PNG written (not committed; the director re-runs the script to look at the view).

## Deviations

1. **`run_task` lifetime mechanics differ from a plain closure keep-alive.** The first implementation kept the `QRunnable` alive by connecting a closure to `finished`/`failed`; the view tests then exposed two real bugs in that design: (a) signals → connection → closure → worker → signals is a pure reference cycle, and on Python 3.14 the GC collects it mid-run, silently dropping the queued `finished`/`failed` signals (`RuntimeError: Signal source has been deleted` in the worker); (b) a task that finishes before the caller's `connect()` calls run emits into the void — the view connects immediately after `run_task` returns, but a fast fake task still beat it. `run_task` now anchors the in-flight `WorkerSignals` in a module-level `_alive` list (removed by a `_release` slot when the terminal signal fires) and starts the task through `QTimer.singleShot(0, …)`, so connections made right after `run_task` returns are always in place. The card contract is unchanged: same signature, same return value, caller should still keep the reference.
2. **Action button rule:** Download for every non-installed, non-cloud row (covers `missing`, `corrupt` and ollama `unknown`), Remove iff `status == "installed"`, no button for cloud. The card's "Download (missing or corrupt) or ollama" plus "Remove (installed)" left the ollama-unknown case open; Download is the only sensible caption there.
3. **`required_button` is disabled while *any* task runs** (it is one of the action buttons), and the required-flow progress label maps the service's `(model_id, done, total)` progress onto `Downloading <id> (k of n)` via a dict the worker thread fills before each progress emit — `WorkerSignals.progress` only carries `(done, total)`.
4. **Details pane content** (description, license, family, size class, recommended-for, notes, fit messages, installed-at path, plus the "cannot run on this machine" line for incompatible and the failed-downloads block after a required run) — the card names the fields but not the layout; lines are omitted when the field is empty.
5. **`gui_models_demo.py --role` exits 1 for an unknown role** (consistent with `gui_compare_demo.py`'s unknown-series handling).

## Questions

None blocking.
## Review addendum (director, 2026-09-20)
The builder stopped on the Ollama Cloud weekly usage limit (429) before its last checks. Director: rebased on main, fixed 2 ruff findings in `workers.py` (UP037, SIM105) and 42 pyright errors in the GUI test files (file-level `reportOptionalMemberAccess`/`reportAttributeAccessIssue` off for Qt-stub optionals, one `pyright: ignore` for the fake hardware), changed the table so the model column stretches and the action column is fixed at 110 px (the buttons filled the whole right half). Results: ruff/pyright clean, CPU suite 3514 passed, GUI tests 47 passed; the real-data screenshot of the view was checked.

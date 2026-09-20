# U3a — Models view: the settings list of every model with hardware fit and download buttons

**Owner:** GLM builder · **Branch:** `U3a` · **Worktree:** `V:\OmniScan-wt\U3a` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read `docs/DECISIONS.md` section "Desktop app" (PySide6 **native widgets**, Qt-free service layer, workers in threads), `docs/PRODUCT_SPEC.md` sections 0–2, `src/omniscan/cli.py` `models_list` / `models_download` / `models_remove` (the JSON shape is the stable interface the GUI mirrors), `src/omniscan/models/{catalog,store,download}.py`, `src/omniscan/hw/{detect,assess}.py`, the existing GUI code `src/omniscan/gui/strip_view.py` + `tests/gui/conftest.py` (test setup: `QT_QPA_PLATFORM=offscreen`, `qapp` fixture, `pytest.importorskip("PySide6")`).
PySide6 is an optional extra already installed in the shared `.venv`; **never run `uv sync`** (use `uv run --frozen …`).

## Why
Owner decision: nothing heavy ships with the program; the settings screen lists **every model** (purpose, size, languages, hardware needs, status) with a **download button** and the user decides; the program detects the hardware and warns per model ("needs 8 GB of GPU memory, your card has 4 GB"). The catalog (~34 models: PP-OCR v5/v6 in every size, PaddleOCR-VL, manga-ocr, detector, LaMa, LLMs), the hardware detection and the fit rules exist and are reachable through `omniscan models list --json`. This card builds the **view** for them, plus the small refactor that lets the CLI and the GUI share one row builder.

## Files you may create / modify
- `src/omniscan/models/rows.py` (create — Qt-free), `src/omniscan/cli.py` (modify **only** `models_list` to use `build_rows`; its output text and JSON must stay byte-identical)
- `src/omniscan/gui/services/models.py` (create — Qt-free), `src/omniscan/gui/workers.py` (create), `src/omniscan/gui/models_view.py` (create)
- `tests/unit/test_models_rows.py`, `tests/unit/test_gui_models_service.py` (create), `tests/gui/test_workers.py`, `tests/gui/test_models_view.py` (create); existing `tests/unit/test_models_cli.py` must pass **unchanged**
- `scripts/gui_models_demo.py` (create: offscreen screenshot of the view, like `scripts/gui_compare_demo.py`)
- `docs/reports/U3a.md` (create)
Anything else is off-limits (especially `pyproject.toml`, `uv.lock`, `src/omniscan/core/**`, `config/models.toml`).

## Part 1 — `models/rows.py` (Qt-free, shared by CLI and GUI)
```python
@dataclass(frozen=True, slots=True)
class ModelRow:
    id: str; name: str; kind: str; role: str | None; family: str | None; size_class: str | None
    size_mb: int; required: bool; format: str; license: str; description: str
    langs: tuple[str, ...]; recommended_for: tuple[str, ...]; notes: str; used_by: tuple[str, ...]
    status: str                              # missing | installed | corrupt | cloud | unknown
    installed_path: str | None
    fit_level: str                           # ok | slow | warn | incompatible
    fit_device: str | None
    fit_messages: tuple[str, ...]

def build_rows(cfg: Config, *, role: str | None = None, lang: str | None = None,
               catalog: Sequence[ModelEntry] | None = None, hardware: HardwareInfo | None = None,
               ollama_names: set[str] | None = None) -> tuple[list[ModelRow], HardwareInfo]: ...
def row_to_json(row: ModelRow) -> dict[str, Any]: ...      # exactly the per-model object `omniscan models list --json` prints today (same keys, same order, `compatibility` nested)
```
`build_rows` does what `models_list` does today (catalog → role/lang filter → `model_status` with `ollama_names` → `detect_hardware(cfg.paths.models_dir)` unless `hardware` is given → `assess`); `ollama_names` is the parameter `None` = "Ollama unreachable" (the CLI passes what `_ollama_model_names` returns; `build_rows` itself never talks to Ollama). Refactor `models_list` to call it; **all existing CLI tests pass unchanged** and the JSON of the CLI equals `{"models_dir": ..., "models": [row_to_json(r) ...], "hardware": asdict(hw)}`.

## Part 2 — `gui/services/models.py` (Qt-free)
```python
class ModelsService:
    def __init__(self, cfg: Config, *, ollama_names: Callable[[], set[str] | None] | None = None, hardware: Callable[[], HardwareInfo] | None = None) -> None: ...
    def rows(self, *, role: str | None = None, lang: str | None = None) -> tuple[list[ModelRow], HardwareInfo]: ...
    def download(self, model_id: str, on_progress: Callable[[int, int | None], None] | None = None) -> str: ...   # returns download_model's result ("mirror" | "upstream" | "ollama" | "already installed")
    def remove(self, model_id: str) -> bool: ...
    def download_required(self, on_progress: Callable[[str, int, int | None], None] | None = None) -> list[tuple[str, str | None]]: ...   # (model id, error text or None) for every required model that is missing/corrupt, in catalog order; one failure does not stop the others
```
`download`/`remove` look the entry up in the catalog (`ValueError(f"unknown model {model_id!r}")`), call `models.download.download_model/remove_model` with `cfg.paths.models_dir` and `ollama_url=cfg.ollama.local_url`; `ModelDownloadError` propagates. `ollama_names` default = the same helper the CLI uses for the daemon query (import it lazily; when it is private in `cli.py`, move the tiny query function to `models/rows.py` as `ollama_model_names(url)` and make `cli.py` import it — behaviour unchanged).

## Part 3 — `gui/workers.py`
```python
class WorkerSignals(QObject):
    progress = Signal(int, int)        # done, total (total 0 when unknown)
    finished = Signal(object)          # the return value
    failed = Signal(str)               # "<ExceptionType>: <message>"

class TaskWorker(QRunnable):
    def __init__(self, fn: Callable[[Callable[[int, int | None], None]], Any]) -> None: ...   # fn receives a progress callback
    signals: WorkerSignals
def run_task(fn, *, pool: QThreadPool | None = None) -> WorkerSignals: ...     # starts a TaskWorker on the (global) pool and returns its signals
```
The worker calls `fn(progress)`; the progress callback emits `progress(done, total or 0)`; a return value → `finished`; **any** exception → `failed` (never crashes the thread). Signals are delivered to the GUI thread (queued connections). Keep a reference to `signals` until finished (the returned object; document it).

## Part 4 — `gui/models_view.py`
```python
class ModelsView(QWidget):
    def __init__(self, service: ModelsService | Any, parent: QWidget | None = None, *, confirm: Callable[[str, str], bool] | None = None) -> None: ...
    def refresh(self) -> None: ...                       # re-read rows (also after a download/remove finished)
    table: QTableWidget; hardware_label: QLabel; role_combo: QComboBox; lang_combo: QComboBox
    installed_only: QCheckBox; search: QLineEdit; status_label: QLabel; progress: QProgressBar; required_button: QPushButton
    details: QTextBrowser
```
`confirm(title, text) -> bool` is the question hook (default: `QMessageBox.question`); tests pass a lambda.
Layout and behaviour (exact):
- **Header** `hardware_label`: one line built from `HardwareInfo`: `"<best GPU name> · <vram> GB · <backend> — RAM <ram> GB — disk <free> GB free — torch <build>"`; without a GPU `"No GPU found — CPU only — RAM …"`. (Field names: read `hw/detect.py`.)
- **Filters**: `role_combo` (`"All roles"` + the roles present in the rows, sorted), `lang_combo` (`"All languages"` + the languages present, sorted), `installed_only`, `search` (case-insensitive substring of id/name/description); filtering is client-side over the rows already loaded (no service call), every change re-populates the table. Rows are ordered as the service returns them (catalog order).
- **Table** columns (header text exact): `Model`, `Role`, `Languages`, `Size`, `Fit`, `Status`, `` (action column). `Model` = the row's `name` (tooltip: the `id`), `Role` = role or `-`, `Languages` = `", ".join(langs)` or `-`, `Size` = `"<n> MB"` or `"<x.y> GB"` from 1024 MB up, `"cloud"` for a cloud model (size 0), `Fit` = the fit level word, background colour by level: `ok` none, `slow` light yellow, `warn` light orange, `incompatible` light red; the `Fit` cell tooltip = the fit messages joined by newlines (empty for a clean `ok`), `Status` = status word (`installed` in bold). The action cell holds a `QPushButton`: `Download` (status missing/corrupt/unknown, or ollama), `Remove` (status installed), nothing for `cloud`. Required models show `(required)` after the name.
- **Details** (`details`, updated when the current row changes): description, `license`, `family`/`size_class` when set, `recommended for: ko` when set, `notes` when set, the fit messages, the installed path when installed, and for `incompatible` the words `This model cannot run on this machine.`
- **Download button**: for a `warn`/`slow` model no question; for **`incompatible`** call `confirm("Download anyway?", "<name> is not compatible with this machine:\n<messages>")` first — `False` → nothing happens. Then `run_task(lambda progress: service.download(id, progress))`: disable that row's button and all other action buttons, show `progress` (a determinate bar when `total > 0`, busy/indeterminate `0..0` otherwise), `status_label` = `"Downloading <name> …"`; on `finished` → `status_label` = `"<name>: installed (<source>)"`, `refresh()`; on `failed` → `status_label` = `"<name>: <error text>"` (red text via a style sheet), buttons re-enabled, `refresh()`.
- **Remove button**: `confirm("Remove model", "Delete <name> from disk?")`, then `run_task(lambda progress: service.remove(id))`; `status_label` = `"<name>: removed"` or `"<name>: nothing to remove"`; `refresh()`.
- **`required_button`** (`"Download required models"`; disabled when nothing required is missing): `run_task` over `service.download_required(...)`; the status label shows `"Downloading <id> (k of n) …"`; at the end a summary `"<ok> installed, <failed> failed"` (failures listed in `details`).
- The view never blocks the GUI thread: every service call except `rows` at construction/refresh runs in a worker; `rows`/`refresh` are fast in practice but must be wrapped so an exception shows in `status_label` instead of crashing.

## Part 5 — `scripts/gui_models_demo.py`
`uv run --frozen python scripts/gui_models_demo.py [--screenshot OUT.png] [--size 1300x800] [--role ROLE]`: builds the real `ModelsService(get_config())`, shows the `ModelsView`; screenshot mode sets `QT_QPA_PLATFORM=offscreen` and (on Windows) `QT_QPA_FONTDIR=C:\Windows\Fonts` **before** importing Qt (the offscreen platform has no fonts), grabs the widget to a PNG and exits 0. The director uses it to look at the real thing.

## Acceptance tests (offscreen, no network, no Ollama, no downloads)
**`test_models_rows.py`** (Qt-free): `build_rows` with a hand-made catalog (zip installed, hf missing, cloud, ollama with `ollama_names` given / `None`) and a fake `HardwareInfo`: fields, statuses, fit levels/messages/device, filters `role`/`lang`, catalog order kept; `row_to_json` equals the object the CLI prints (compare against `omniscan models list --json` output of the same catalog in a CliRunner test: identical dicts); the CLI text output unchanged (the existing `test_models_cli.py` passes unchanged).
**`test_gui_models_service.py`** (Qt-free): `download`/`remove` call the functions with the right entry and arguments (monkeypatch `download_model`/`remove_model` in `omniscan.gui.services.models`), unknown id → `ValueError`, progress callback forwarded, `ModelDownloadError` propagates; `download_required` downloads exactly the required missing/corrupt models in order, a failing one is reported and the rest still run, an installed one is skipped.
**`test_workers.py`**: `run_task` with a function returning 7 → `finished(7)` arrives (use `QSignalSpy`/`qtbot`-free `QEventLoop` with a timeout); progress values arrive in order with `total 0` for `None`; an exception → `failed("ValueError: boom")` and no crash; two tasks run concurrently and both finish.
**`test_models_view.py`** (with a `FakeService` returning fixed rows: one ok installed, one missing ok, one slow, one warn, one incompatible, one cloud, one required-missing): header text exact; table cell texts/tooltips/backgrounds per the definition; button captions per status (none for cloud); filters (role, language, installed-only, search) each change the row count as expected and combine; details text for a selected row (incompatible line present); Download on an ok model calls `service.download(id, …)` in a **worker thread** (record `threading.get_ident()`), shows the progress bar determinate/indeterminate as fed, ends with the installed message and a refresh (the fake flips the status); failure → red status text with the error; Download on the incompatible row with `confirm` → False: service not called; → True: called; Remove with confirm False/True; `required_button` state and the summary text; all action buttons are disabled while a task runs; an exception in `service.rows` shows in `status_label`.
**Regression**: `uv run --frozen pytest -q -m "not gpu"` green; `tests/unit/test_models_cli.py` unchanged and green; `tests/unit/test_docs.py` green.

## Out of scope
The main window, navigation, settings editing, reader/library pages (card U3c), LLM model pulls with progress details, download resume/cancel, theming.

## Commands to run before finishing
```bash
uv run --frozen pytest tests/unit/test_models_rows.py tests/unit/test_gui_models_service.py tests/unit/test_models_cli.py tests/gui -q
uv run --frozen pytest -q -m "not gpu"
uv run --frozen ruff format . && uv run --frozen ruff check .
uv run --frozen pyright
```

## Report
`docs/reports/U3a.md`: Changes, Tests (commands + results), Deviations, Questions. **Commit early** (`U3a: WIP rows + service`) once Parts 1–2 and their tests pass, again after the workers; the final commit is `U3a: models view`. You have 150 tool calls in total. If anything is unclear: stop, write the question under Questions, commit, and end.

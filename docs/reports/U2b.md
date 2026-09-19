# U2b — The pipeline loads models from the model manager's folders (and `doctor` reports missing ones)

## Changes

- **`src/omniscan/models/resolve.py`** (new): `local_model_source(repo, models_dir, catalog=None) ->
  str | None` — the installed folder `<models_dir>/<id>` as a string when the catalog (default
  `load_catalog()`) has a `zip` entry with `upstream_repo == repo` and
  `model_status(entry, models_dir, ollama_names=None) == "installed"`; otherwise `None`. Never
  raises: an unreadable/corrupt catalog (`OSError`/`ValueError`) → `None`, unknown repo → `None`,
  non-zip entries never match. `models_dir` is coerced with `Path(...)` (same as
  `download_model`/`remove_model`), so a `str` works too (the card's live one-liner passes one).
- **`src/omniscan/detect/model.py`**: `Detector.load(cfg, device, models_dir=None)` (new optional
  last parameter). With `models_dir` and an installed folder: `from_pretrained(source,
  local_files_only=True)` for model and processor, no `revision`, INFO `loading <repo> from <path>`
  (new module logger `log`). Otherwise exactly today's hub behaviour with the pinned `revision` —
  plus, only when `models_dir` was given, one WARNING `model <repo> is not installed in <models_dir>;
  using the Hugging Face hub/cache. Run "omniscan models download --required" to install it.`
- **`src/omniscan/ocr/model.py`**: the same change for `LineDetector.load` (`cfg.det_repo`,
  `cfg.det_revision`) and `LineRecognizer.load` (`cfg.rec_repo`, `cfg.rec_revision`).
- **`src/omniscan/gpu/groups.py`**: `load_vision` passes `models_dir=cfg.paths.models_dir` to all
  three `load` calls. `load_inpaint`/LaMa untouched (it already reads `<models_dir>/lama/`).
- **`src/omniscan/doctor.py`**: new `check_models(cfg)` registered in `_CHECKS` after `paths`:
  required catalog entries → all `installed` → OK `"3 required models installed"` (real count);
  otherwise WARN `f"{n} required model(s) not installed ({total_mb} MB): {ids}; run "omniscan models
  download --required" (until then they are loaded from the Hugging Face hub/cache)"` (`total_mb` =
  summed `size_mb`); a catalog error → WARN with the exception text. One `try` around catalog load +
  status computation, so the check itself never raises (and `run_all_checks` wraps it anyway).
  Module docstring updated.
- **Tests**: `tests/unit/test_models_resolve.py` (new, 10), `test_detect_model.py` (+3: local with
  INFO log, fallback with one WARNING, no `models_dir` → no warning), `test_ocr_model.py` (+3:
  LineDetector local, LineRecognizer local, LineRecognizer fallback), `test_gpu_groups.py`
  (recorder signatures now take `models_dir`; asserts `cfg.paths.models_dir` reaches all three),
  `test_doctor.py` (+4: all installed OK, missing WARN with ids/size/command, corrupt counts as not
  installed, broken catalog WARN; `REQUIRED_NAMES` gained `models` between `paths` and `codec`).
- **Docs**: `docs/USER_GUIDE.md` — `### omniscan doctor` mentions the new `models` row; `### omniscan
  models` gained the "How the pipeline finds models" paragraph. `docs/MODELS.md` — one sentence on
  the pipeline loading the installed folder and doctor reporting the missing ones.

## Tests

```
uv run pytest tests/unit/test_models_resolve.py tests/unit/test_detect_model.py \
  tests/unit/test_ocr_model.py tests/unit/test_gpu_groups.py tests/unit/test_doctor.py \
  tests/unit/test_docs.py
→ 66 passed

uv run pytest          → 2871 passed (2:20, GPU tests included, 0 failed)
uv run ruff format . && uv run ruff check .   → clean
uv run pyright                → 0 errors
```

All new tests are CPU-only and network-free; `from_pretrained` is faked with recorders, the catalog
with hand-built `ModelEntry` lists monkeypatched into `omniscan.models.resolve.load_catalog`, and
install state is faked with folders + `.installed.json` as in the U2a store tests.

Note: with the repo's `addopts = "-ra -q"`, an extra `-q` on the command line makes pytest `-qq`,
which suppresses the summary line — the card's `uv run pytest -q` prints only the progress dots.
The counts above come from a run without the extra `-q`.

## Live check (test 6, real machine)

1. `uv run omniscan doctor` — the new `models` row (nothing installed in the worktree's models dir):

```
| models        | WARN   | 3 required model(s) not installed (262 MB):        |
|               |        | detector-comic-text-bubble,                        |
|               |        | ocr-det-ppocrv5-server,                            |
|               |        | ocr-rec-korean-ppocrv5-mobile; run "omniscan       |
|               |        | models download --required" (until then they are   |
|               |        | loaded from the Hugging Face hub/cache)            |
+-----------------------------------------------------------------------------+
7 ok, 4 warn, 0 fail
```

2. `uv run omniscan models download ocr-rec-korean-ppocrv5-mobile` with
   `OMNISCAN_PATHS__MODELS_DIR=<temp dir>` — the private mirror 404s (anonymous), the upstream
   fallback fetched the pinned revision from Hugging Face:

```
Fetching 6 files: 100%|##########| 6/6 [00:00<00:00, 17.09it/s]
ocr-rec-korean-ppocrv5-mobile: installed from upstream
```

   (First attempt: uv could not parse a backslash path in `--env-file`, the var was ignored and the
   model landed in the worktree's gitignored `models/`; moved to the intended temp dir, re-ran →
   `ocr-rec-korean-ppocrv5-mobile: already installed`. Nothing was committed.)

3. `LineRecognizer.load(OcrConfig(), torch.device("cpu"), <temp dir>)` (script with
   `logging.basicConfig(level=logging.INFO)`):

```
INFO omniscan.ocr.model: loading PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors from C:\Users\limex\AppData\Local\Temp\u2b_live_9kg5hbj5\ocr-rec-korean-ppocrv5-mobile
Loading weights: 100%|##########| 884/884 [00:00<00:00, 15216.34it/s]
loaded OK: 6903367 params, dtype torch.float32
```

   The installed folder is used (INFO line with the folder path) and the real recognizer loads from
   it with `local_files_only=True`. The temp dir was not committed and the scratch files were removed.

## Deviations

- **`local_model_source` accepts `models_dir: Path | str`** (card: `Path`): the card's own live
  one-liner passes a string, which crashed in `install_path` (`TypeError: 'str' and 'str'` for `/`)
  before the coercion. Same pattern as `download_model`/`remove_model` (declared `Path`, coerced
  with `Path(models_dir)`); `Path` callers are unchanged. One extra test covers it.
- **Loader tests monkeypatch `resolve.load_catalog`** with hand-built entries instead of writing
  temp TOML catalogs: hermetic and independent of `config/models.toml` hashes. The TOML path is
  covered by `test_models_catalog.py` (U2a), the unreadable-catalog test, and the live check above
  (real catalog + real folder).
- **`check_models` treats every non-`installed` status as not installed** (so `corrupt` counts, per
  the card). For a required ollama entry with an unreachable daemon this would read `unknown` and be
  counted as missing — no such entries exist today; flagged under Questions.
- `resolve.py` and `doctor.py` use Python 3.14's paren-less `except OSError, ValueError:` (PEP 758),
  matching `store.py`'s style.

## Questions

1. **Two resolve helpers now exist**: `store.resolve_model_path` (by catalog *id*, used by the U2a
   CLI) and `resolve.local_model_source` (by *upstream repo*, used by the loaders). Different keys,
   so both are legitimate — but if the director prefers one shared helper, `local_model_source`
   could take an entry and both could be built on it.
2. **`unknown` vs missing in `check_models`**: should an `unknown` status (required ollama model,
   daemon unreachable) be excluded from the "not installed" count? Today only zip entries are
   required, so this cannot trigger.
3. **HF cache adoption** (U2a report Q2): with U2b installed, manually copied Hugging Face cache
   folders still count as `corrupt` (no `.installed.json`) — doctor warns and the loaders ignore
   them. Out of scope here per the card; the fix would be a `models verify --adopt` or marker
   writer for cache copies.
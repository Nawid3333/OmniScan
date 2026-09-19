# U2b — The pipeline loads models from the model manager's folders (and `doctor` reports missing ones)

**Owner:** GLM builder · **Branch:** `U2b` · **Worktree:** `V:\OmniScan-wt\U2b` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read the merged U2a code and report: `src/omniscan/models/{catalog,store,download}.py`, `config/models.toml`, `docs/reports/U2a.md`; the loaders `src/omniscan/detect/model.py` (`Detector.load`), `src/omniscan/ocr/model.py` (`LineDetector.load`, `LineRecognizer.load`), `src/omniscan/gpu/groups.py` (`build_vram_manager`), `src/omniscan/doctor.py`, and their tests `tests/unit/test_detect_model.py`, `tests/unit/test_ocr_model.py`, `tests/unit/test_gpu_groups.py`, `tests/unit/test_doctor.py` (how `from_pretrained` is faked).

## Why
`omniscan models download` now puts the vision/OCR models into `<paths.models_dir>/<id>/` (mirror first, Hugging Face as fallback), but the loaders still call `from_pretrained("<repo id>")`, i.e. they ignore that folder and use the Hugging Face hub/cache. This card makes the loaders **prefer the installed folder**, so an installed app works offline and the settings screen's "installed" means "used".
LaMa needs nothing: `inpaint/lama_weights.py` already looks at `<models_dir>/lama/<file>`, the same path the manager installs to.

## Files you may create / modify
- `src/omniscan/models/resolve.py` (create), `src/omniscan/detect/model.py`, `src/omniscan/ocr/model.py`, `src/omniscan/gpu/groups.py`, `src/omniscan/doctor.py` (modify — only what is described below)
- `tests/unit/test_models_resolve.py` (create); `tests/unit/test_detect_model.py`, `test_ocr_model.py`, `test_gpu_groups.py`, `test_doctor.py` (extend)
- `docs/USER_GUIDE.md` (`### omniscan models`: a paragraph "how the pipeline finds models"; `### omniscan doctor`: the new check), `docs/MODELS.md` (one sentence), `docs/reports/U2b.md` (create)
Do not modify `src/omniscan/core/**`, `config/models.toml` or the U2a download code.

## Part 1 — `models/resolve.py`
```python
def local_model_source(repo: str, models_dir: Path, catalog: Sequence[ModelEntry] | None = None) -> str | None
```
Returns the folder `<models_dir>/<id>` **as a string** when the catalog (default: `load_catalog()`) has a `zip` entry whose `upstream_repo == repo` **and** `model_status(entry, models_dir, ollama_names=None) == "installed"`; otherwise `None` (never raises for a missing/corrupt/unknown entry; a catalog that cannot be loaded → `None`).

## Part 2 — loaders
`Detector.load(cfg, device, models_dir: Path | None = None)`, `LineDetector.load(cfg, device, models_dir=None)`, `LineRecognizer.load(cfg, device, models_dir=None)` (new optional last parameter; existing calls stay valid):
- With `models_dir` given: `source = local_model_source(<repo>, models_dir)`. When it is a path: call `from_pretrained(source, local_files_only=True)` for the model **and** the processor, **without** `revision`, and log at INFO `loading <repo> from <path>` (use the module's logger; add `logging.getLogger(__name__)` if missing).
- Otherwise (no `models_dir`, or `local_model_source` returned `None`): exactly today's behaviour (`from_pretrained(<repo>, **extra)` with the pinned `revision`), plus — only when a `models_dir` was given — one WARNING `model <repo> is not installed in <models_dir>; using the Hugging Face hub/cache. Run "omniscan models download --required" to install it.`
`gpu/groups.py`: pass `cfg.paths.models_dir` to the three `load` calls.

## Part 3 — `doctor`
New check function `check_models(cfg: Config) -> CheckResult` registered in `run_all_checks` after `check_paths`: reads the catalog, computes `model_status(entry, cfg.paths.models_dir, ollama_names=None)` for the entries with `required == True`; all installed → OK `"3 required models installed"` (use the real count); otherwise WARN `f"{n} required model(s) not installed ({total_mb} MB): {ids}; run \"omniscan models download --required\" (until then they are loaded from the Hugging Face hub/cache)"` (`total_mb` = sum of `size_mb` of the missing ones). A catalog error → WARN with the exception text. Never raises. Follow the `CheckResult` style of the other checks and keep `run_all_checks` never raising.

## Acceptance tests (CPU, no downloads; fake `from_pretrained`)
1. **`local_model_source`** with a temp `models_dir` and a small hand-made catalog (`zip` entries; make the folder + a matching `.installed.json` as in the U2a store tests): installed → the folder path string; missing folder → `None`; folder with a wrong-hash marker (corrupt) → `None`; repo not in the catalog → `None`; a `file`/`ollama` entry never matches; two entries with different repos resolve independently; an unreadable catalog file → `None`.
2. **Loaders — local:** with the three catalog repos installed, `Detector.load`, `LineDetector.load`, `LineRecognizer.load` (fake `from_pretrained` recording arguments, as in the existing tests) receive the folder path, `local_files_only=True` and **no** `revision`, for both model and processor; an INFO log record `loading <repo> from` is emitted (caplog).
3. **Loaders — fallback:** `models_dir` given but nothing installed → the repo id and the pinned `revision` (when the config pins one) exactly as before, and one WARNING containing `omniscan models download --required`; no `models_dir` argument → today's behaviour and **no** warning.
4. **`build_vram_manager`:** the vision loader passes `cfg.paths.models_dir` to all three `load` calls (patch the three `load` classmethods with recorders); the LaMa loader is unchanged.
5. **doctor:** all required installed → OK with the count; two missing → WARN naming them with the summed size and the download command; corrupt counts as not installed; broken catalog → WARN; `run_all_checks` still never raises (existing test) and now contains a `models` check.
6. **Live check for the report (do not commit anything):** `uv run omniscan doctor` on the real machine (the models are not installed there, so expect the WARN with 262 MB); then `OMNISCAN_PATHS__MODELS_DIR=<temp dir>` with `uv run omniscan models download ocr-rec-korean-ppocrv5-mobile` and a Python one-liner `LineRecognizer.load(OcrConfig(), torch.device("cpu"), <temp dir>)` proving that the local folder is used (log line). Paste the outputs.
7. `tests/unit/test_docs.py` and the whole suite stay green.

## Out of scope
Adopting existing Hugging Face cache copies, removing the hub fallback, per-language model selection, changing LaMa's download, any GUI.

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_models_resolve.py tests/unit/test_detect_model.py tests/unit/test_ocr_model.py tests/unit/test_gpu_groups.py tests/unit/test_doctor.py tests/unit/test_docs.py -q
uv run pytest -q
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
`docs/reports/U2b.md`: Changes, Tests (commands + results), the live outputs, Deviations, Questions. **Commit early** (`U2b: WIP resolve+loaders`) once Parts 1–2 pass; the final commit is `U2b: pipeline loads installed models, doctor check`. You have 150 tool calls in total. If anything is unclear: stop, write the question under Questions, commit, and end.

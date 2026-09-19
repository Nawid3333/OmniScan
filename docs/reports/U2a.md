# U2a — Model catalog and manager

## Changes

- **`config/models.toml`** (new): the shipped catalog with the 8 entries from the card (3 required
  zip vision/ocr models, the LaMa `file`, 3 ollama LLMs, 1 cloud LLM), with per-format fields,
  mirror URLs under `https://github.com/Nawid3333/OmniScan/releases/download/models-v1/`, exact
  `bytes` and upstream repo/revision pins. Comments document the field meanings and the machine
  override rule.
- **`src/omniscan/models/__init__.py`** (new, empty), **`catalog.py`** (new): `ModelEntry` pydantic
  model (`extra="forbid"`), `validate_for_format()` (names the first missing format field),
  `default_catalog_path()` (repo) + `machine_catalog_path()` (`~/.config/omniscan/models.toml`),
  `load_catalog()` — repo order, then machine-only additions, machine entries replace same ids;
  duplicate ids / unknown keys / missing format fields raise `ValueError`.
- **`models/store.py`** (new): `install_path`, `model_status` (all five statuses per the card),
  `file_sha256`, `MARKER_NAME` (`.installed.json`), `resolve_model_path` (U2b's hook — returns the
  path only when the status is `installed`).
- **`models/download.py`** (new): `download_model` (mirror → upstream fallback for zip/file; ollama
  NDJSON `/api/pull`; cloud → error; "already installed" short-circuits with zero requests),
  `remove_model` (folder / file + empty parent / `DELETE /api/delete` / cloud → False),
  `verify_models`. Downloads stream into `<target>.part` hashed on the fly, deleted on every
  failure path; zips are member-validated against zip-slip (relative `../` and absolute/drive
  paths) before anything is extracted, then extracted into `models_dir` and marked with
  `.installed.json` (`id`, `sha256`, `source`, `revision`, `installed_at`); files are moved into
  place with `Path.replace`. `huggingface_hub.snapshot_download` is imported lazily on the zip
  upstream fallback only (injectable fake for tests).
- **`src/omniscan/cli.py`**: only the new `models` command group added (`list [--json]`,
  `download ID... [--required]`, `remove ID...`, `verify [ID...]`), plus the helpers
  `_ollama_model_names` (GET `/api/tags` with a 2 s timeout → `None` when unreachable) and
  `_models_progress` (5 %-step-throttled `id: 42% (66/158 MB)` lines). Everything else untouched.
- **`docs/MODELS.md`** (new): third-party notice table (model, purpose, upstream, revision,
  licence, size, mirror asset).
- **`README.md`**: status row `models — working`. **`docs/USER_GUIDE.md`**: `### omniscan models`
  subsection (commands, statuses, examples) and the `paths.models_dir` key row updated.
- **`.gitignore`**: `models/` → `/models/` (see Deviations).
- **Tests** (new): `test_models_catalog.py` (13), `test_models_store.py` (16), 
  `test_models_download.py` (19), `test_models_cli.py` (16).

## How tested

```
uv run pytest tests/unit/test_models_catalog.py tests/unit/test_models_store.py \
  tests/unit/test_models_download.py tests/unit/test_models_cli.py tests/unit/test_docs.py -q
→ 64 passed
uv run pytest -q            → all green (see below for the exact count)
uv run ruff format . && uv run ruff check .   → clean
uv run pyright              → 0 errors
```

All tests are CPU-only and network-free (`httpx.MockTransport`, a fake `snapshot_download`, temp
dirs, small synthetic zips with patched hashes). The catalog tests read the real
`config/models.toml` and check it against `InpaintConfig`/`DetectConfig`/`OcrConfig` and the model
names in `translation_profiles.toml`/`judge.toml`, so the catalog cannot drift from the pipeline
contracts.

Optional real check on `V:\OmniScan\models` (via `OMNISCAN_PATHS__MODELS_DIR`): `models list` shows
`inpaint-big-lama … installed` (size + sha256 of the real 205 MB file verified), the three HF zips
`missing` (they live in the Hugging Face cache — expected until U2b), the three local LLMs
`installed` (real Ollama daemon) and the cloud entry `cloud`; footer
`3 required model(s) missing, 262 MB to download`. `models verify` prints the four non-llm
statuses and exits 0.

## Deviations

- **`.gitignore`**: the unanchored `models/` pattern ignored the new `src/omniscan/models` package,
  so `git add` refused it. Changed it to `/models/` (anchored to the repo root — the weights dir is
  only there) and added a comment. Not in the card's file list, but required to commit the card's
  files; `library/ work/ output/ samples/` were left alone.
- **`default_catalog_path()`/`machine_catalog_path()`**: the card's `default_catalog_path()` returns
  the repo path; the machine override is a second function so `load_catalog(None)` can merge the two
  (repo order, machine-only additions) — same merge rule as the translation profiles. An explicit
  `path` argument reads only that file.
- **`installed_path` in `list --json`** is only non-null for models whose status is `installed`
  (matching `resolve_model_path` semantics); for missing/corrupt/cloud/ollama entries it is `null`.
- **`download --required`** prints `id: already installed` for required models that are already
  installed (instead of silently skipping them), and `download` with neither ids nor a
  missing-required model exits 2 with a message; the card's `ID...` is optional so `--required`
  alone works.
- **Upstream zip installs** write `.installed.json` with the CATALOG `sha256` (documented in a
  comment): upstream HF files cannot be compared against the mirror zip's hash (different
  packaging), as the card anticipated.
- `except` clauses use Python 3.14's paren-less multi-exception syntax where `ruff format` emitted
  it (PEP 758); compiles and lints fine.

## Questions

1. **`required` for `ocr-rec-korean-ppocrv5-mobile`**: the card says "required for Korean sources"
   but also fixes the test "exactly ids 1–3 are `required`" — I set `required = true` with a TOML
   comment noting it is Korean-specific. When more source languages arrive, this flag should become
   per-language (U2b/GUI concern).
2. **Zip status without a marker**: a `<models_dir>/<id>/` folder without `.installed.json` counts
   as `corrupt` (per the card). Real HF cache copies migrated manually would show as corrupt until
   U2b decides how to adopt them — worth a decision there.
3. **Ollama "already installed" in `download`**: `download_model` cannot know the daemon's tags
   (no `ollama_names` in its signature), so an already-pulled ollama model is re-pulled (a fast
   no-op daemon-side). The CLI avoids the re-pull by checking `/api/tags` first. If U2b or the GUI
   wants `download_model` itself to skip, it needs an `ollama_names` parameter — not added now to
   keep the card's signature.
## Review addendum (director)
- Rebased onto `main` (no conflicts); ruff, pyright, the four models test files and `test_docs.py` are green; the `.gitignore` change (`/models/` anchored) is accepted (it was hiding the new package).
- **Catalog check:** all four `zip`/`file` entries match the published `models-v1/manifest.json` (sha256, bytes, file name).
- **Live check on the real machine** (`OMNISCAN_PATHS__MODELS_DIR` pointing at an empty temp folder): `omniscan models list` shows the three required models as missing ("262 MB to download") and the three local Ollama models as installed via the real daemon; `models download ocr-rec-korean-ppocrv5-mobile` tried the private mirror (anonymous 404), fell back to Hugging Face and printed `installed from upstream`; `.installed.json` records the pinned revision; `models verify` says `installed`. So the fallback works end to end.
- **Answers to the questions:** (1) the `required` flag is fine for now. (2) A folder without `.installed.json` counts as corrupt — U2b decides how to adopt existing Hugging Face cache copies. (3) Accepted; the GUI/U2b will use the CLI path (which checks `/api/tags` first).
- Follow-ups: **U2b** (the pipeline loads models from `models_dir`, so the download is actually used) and a mutation review of `models/*` (Q4).

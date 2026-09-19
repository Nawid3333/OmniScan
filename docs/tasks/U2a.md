# U2a — Model catalog and manager: `config/models.toml`, `omniscan models list|download|remove|verify`

**Owner:** GLM builder · **Branch:** `U2a` · **Worktree:** `V:\OmniScan-wt\U2a` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read `src/omniscan/core/config.py` (`Config`, `PathsConfig.models_dir`, `InpaintConfig.lama_*`, `DetectConfig.repo`, `OcrConfig.det_repo/rec_repo`), `src/omniscan/inpaint/lama_weights.py` (how the LaMa file is downloaded and hash-checked today — reuse the idea, do not change that module),
`src/omniscan/llm/ollama.py` (`OllamaClient`, base URL handling), `config/translation_profiles.toml`, `config/judge.toml`, `src/omniscan/doctor.py` (how checks are written), `src/omniscan/cli.py` (a command group such as `queue`/`watermark` for the typer sub-app style), and `tests/unit/test_doctor.py` / `tests/unit/test_gpu_vram.py` (`httpx.MockTransport` usage).

## Why
The shipped app will be small and download the models the user chooses (owner decision 2026-09-19): a settings screen lists **every** model with its size and purpose, and the user decides what to download. The three Hugging Face vision models and the LaMa file are mirrored unchanged as assets of the GitHub release `models-v1` of `Nawid3333/OmniScan` (all Apache-2.0); large language models come from Ollama.
This card builds the catalog file, the manager and the CLI (`--json` output is the stable interface the future PySide6 settings screen will use). It does **not** change how the pipeline loads models (card U2b does that).

## Files you may create / modify
- `config/models.toml` (create — the catalog, contents below), `docs/MODELS.md` (create — third-party notice table: model, purpose, upstream, revision, licence, size, mirror)
- `src/omniscan/models/__init__.py` (empty), `models/catalog.py`, `models/store.py`, `models/download.py` (create)
- `src/omniscan/cli.py` (modify — ONLY add the `models` command group; keep everything else)
- `tests/unit/test_models_catalog.py`, `tests/unit/test_models_store.py`, `tests/unit/test_models_download.py`, `tests/unit/test_models_cli.py` (create)
- `README.md` (status row `models` `working — omniscan models list/download/remove/verify`) and `docs/USER_GUIDE.md` (a `### omniscan models` subsection; the catalog file and `paths.models_dir`) — `tests/unit/test_docs.py` must stay green
- `docs/reports/U2a.md` (create)
Do not modify `src/omniscan/core/**`, the loaders (`detect/model.py`, `ocr/model.py`, `inpaint/*`) or `config/default.toml`. **Do not download anything real in tests or during development**, except the single optional check at the end (below).

## Part 1 — `config/models.toml` (create exactly these entries, in this order)
Fields per `[[model]]`: `id`, `name`, `kind` (`vision` | `ocr` | `inpaint` | `llm`), `required` (bool: the pipeline cannot run without it), `format` (`zip` | `file` | `ollama` | `cloud`), `size_mb` (int, approximate download size), `license`, `description` (one sentence), `used_by` (list of config keys / profile names that use it), and per format:
- `zip`: `mirror_url`, `sha256`, `bytes`, `upstream_repo`, `upstream_revision`. The zip contains **only** files under a top-level folder `<id>/`; it is extracted into `<models_dir>/` so the model lives in `<models_dir>/<id>/`.
- `file`: `mirror_url`, `sha256`, `bytes`, `upstream_url`, `install_path` (relative to `models_dir`).
- `ollama`: `ollama_name` (the tag to pull). `cloud`: `ollama_name`, no download (runs on Ollama Cloud through the local daemon; needs an Ollama account).
Entries:
1. `detector-comic-text-bubble` — "Text and bubble detector", vision, required, zip, 159 MB, Apache-2.0, "RT-DETR-v2 that finds speech bubbles and free text.", used_by `["detect.repo"]`, mirror `https://github.com/Nawid3333/OmniScan/releases/download/models-v1/detector-comic-text-bubble.zip`, sha256 `cccc8a235d90839428327394cc466851961dfc13bf3a6a0658739a8d08224268`, bytes `158938404`, upstream_repo `ogkalu/comic-text-and-bubble-detector`, upstream_revision `16e8a622f91fabc6b5b65c96d32d1183f8843546`.
2. `ocr-det-ppocrv5-server` — "Text-line detector (PP-OCRv5 server)", ocr, required, zip, 80 MB, Apache-2.0, "Finds the lines of text inside a region.", used_by `["ocr.det_repo"]`, mirror `.../ocr-det-ppocrv5-server.zip`, sha256 `a08b334d8b8d6e835c696cfa2530ac1be1548a7e76920489ca18bf27ff4d23a9`, bytes `79265655`, upstream_repo `PaddlePaddle/PP-OCRv5_server_det_safetensors`, upstream_revision `cbea9f3c3254c6ff7b0016cfbf90549e1ad4c5bb`.
3. `ocr-rec-korean-ppocrv5-mobile` — "Korean text recognition (PP-OCRv5 mobile)", ocr, required for Korean sources, zip, 23 MB, Apache-2.0, "Reads Korean text lines.", used_by `["ocr.rec_repo"]`, mirror `.../ocr-rec-korean-ppocrv5-mobile.zip`, sha256 `addc407093d7537ae0fcc6b60274d014ec6a8ed70956a701bd2dcb5994aa948b`, bytes `22069043`, upstream_repo `PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors`, upstream_revision `6ef525a357645ce46495c7ed1fef622c4e009e7a`.
4. `inpaint-big-lama` — "LaMa inpainting (big-lama)", inpaint, **not** required (flat fill works without it), file, 206 MB, Apache-2.0, "Removes text from artwork so the English can be lettered.", used_by `["inpaint.lama_file"]`, mirror `.../big-lama.pt`, sha256 `344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9`, bytes `205669692`, upstream_url `https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt`, install_path `lama/big-lama.pt`.
5. `llm-translategemma-12b` — "TranslateGemma 12B", llm, not required, ollama, 8110 MB, license `"Gemma terms"`, "Local translation model (one candidate of the ensemble).", used_by `["translategemma-12b-local"]`, ollama_name `translategemma:12b`.
6. `llm-gemma4-12b` — "Gemma 4 12B", llm, not required, ollama, 7560 MB, `"Gemma terms"`, "Local general model used as a second translation candidate.", used_by `["gemma4-12b-local"]`, ollama_name `gemma4:12b`.
7. `llm-gemma4-31b` — "Gemma 4 31B", llm, not required, ollama, 19870 MB, `"Gemma terms"`, "Larger local model for machines with 24 GB or more of VRAM.", used_by `[]`, ollama_name `gemma4:31b`.
8. `llm-gemma4-31b-cloud` — "Gemma 4 31B (Ollama Cloud)", llm, not required, cloud, 0 MB, `"Gemma terms"`, "Runs on Ollama Cloud (no download; needs an Ollama account); default judge and third candidate.", used_by `["gemma4-31b-cloud", "judge"]`, ollama_name `gemma4:31b-cloud`.
The mirror base is `https://github.com/Nawid3333/OmniScan/releases/download/models-v1/` (write the full URLs). The mirror lives in a private repo today, so a download from it may fail with 404: the manager must then fall back to the upstream source automatically (Part 3).

## Part 2 — `models/catalog.py` and `models/store.py`
```python
ModelKind = Literal["vision", "ocr", "inpaint", "llm"]
ModelFormat = Literal["zip", "file", "ollama", "cloud"]

class ModelEntry(BaseModel):            # pydantic, extra="forbid"; fields exactly as in Part 1 (format-specific ones optional)
    ...
    def validate_for_format(self) -> None    # raises ValueError naming the missing field (e.g. "zip model 'x' needs sha256")

def default_catalog_path() -> Path            # <repo root>/config/models.toml, overridable per machine by ~/.config/omniscan/models.toml which REPLACES entries with the same id and may add new ones (mirror how translation profiles are merged)
def load_catalog(path: Path | None = None) -> list[ModelEntry]   # order: repo file order, then machine-only additions; duplicate ids in ONE file -> ValueError; unknown keys -> ValueError

Status = Literal["installed", "missing", "corrupt", "cloud", "unknown"]
def install_path(entry: ModelEntry, models_dir: Path) -> Path | None   # zip: models_dir/<id>; file: models_dir/<install_path>; ollama/cloud: None
def model_status(entry: ModelEntry, models_dir: Path, *, ollama_names: set[str] | None) -> Status
def resolve_model_path(entry_id: str, cfg: Config, catalog: Sequence[ModelEntry] | None = None) -> Path | None   # the install path when status == "installed" else None (U2b will use this)
```
`model_status`: `cloud` → `"cloud"`; `ollama` → `"installed"` when `ollama_names` is not None and contains the tag, `"missing"` when not contained, `"unknown"` when `ollama_names is None` (Ollama unreachable); `zip` → `"missing"` when the folder does not exist, `"installed"` when `<folder>/.installed.json` exists, parses, and its `sha256` equals the catalog's, else `"corrupt"`;
`file` → `"missing"` when absent; `"installed"` when the file's size equals `bytes` **and** its SHA-256 equals `sha256`; else `"corrupt"`.

## Part 3 — `models/download.py`
```python
class ModelDownloadError(RuntimeError): ...
def download_model(entry: ModelEntry, models_dir: Path, *, client: httpx.Client | None = None,
                   hf_download: Callable[..., str] | None = None, ollama_url: str | None = None,
                   on_progress: Callable[[str, int, int | None], None] | None = None) -> str   # returns "mirror" | "upstream" | "ollama" | "already installed"
def remove_model(entry: ModelEntry, models_dir: Path, *, client: httpx.Client | None = None, ollama_url: str | None = None) -> bool   # True when something was removed
def verify_models(entries, models_dir, ...) -> dict[str, Status]
```
- **zip / file from the mirror:** stream GET `mirror_url` (follow redirects) into `<target>.part`, hashing while writing, calling `on_progress(entry.id, done_bytes, total_or_None)`; when `sha256` (and `bytes`) do not match: delete the `.part`, treat as a failed source. A 404/403/network error/hash mismatch on the mirror → **fall back to the upstream source** (below); if that also fails raise `ModelDownloadError` with both messages. A hash mismatch on a `file` entry from the upstream URL is a hard error (the file is deleted).
  After a verified zip: extract into `models_dir` with **zip-slip protection** (every member path must resolve inside `models_dir/<id>/`; a member outside → `ModelDownloadError`, nothing extracted), then write `<models_dir>/<id>/.installed.json` = `{"id":..., "sha256":..., "source": "mirror"|"upstream", "revision":..., "installed_at": <ISO UTC>}`. The zip file is deleted after extraction. A `file` entry is moved into place with `os.replace` from its `.part`.
- **upstream for zip entries:** `hf_download(repo_id, revision=..., local_dir=...)` (default: `huggingface_hub.snapshot_download`, imported lazily; tests inject a fake) into `<models_dir>/<id>`; then write `.installed.json` with `sha256` of the **catalog** value and `source: "upstream"` (upstream files cannot be compared with the mirror hash; say so in a comment). **upstream for file entries:** GET `upstream_url`, same hash check.
- **ollama:** `POST {ollama_url}/api/pull` with `{"model": tag, "stream": true}`; read the NDJSON lines; progress from `completed`/`total` fields; an `error` field or a non-200 → `ModelDownloadError`; `ollama_url` defaults to `Config().ollama.base_url` (read how `OllamaConfig` names it). `cloud` entries: `ModelDownloadError("… runs on Ollama Cloud, nothing to download")`.
- Already installed (`model_status == "installed"`) → return `"already installed"` without any request. Never leave `.part` files behind (also on failure). `remove_model`: zip → delete the folder; file → delete the file (and the empty parent folder); ollama → `DELETE {ollama_url}/api/delete` `{"model": tag}`; cloud → False.

## Part 4 — CLI `omniscan models`
Sub-commands (typer sub-app registered like the others):
- `list [--json]`: reads the catalog and the statuses (`ollama_names` from `GET {ollama_url}/api/tags`, `None` when unreachable within 2 s). Text: one row per model `id  kind  size  required  status  description`, plus a footer line with the total size of the missing **required** models. `--json`: `{"models_dir": "...", "models": [{"id","name","kind","required","format","size_mb","license","description","used_by","status","installed_path"}...]}` (stable key order, UTF-8).
- `download ID... [--required]`: downloads the named models (unknown id → exit 2 with the list of known ids); `--required` adds every required model that is not installed; prints progress lines `id: 42% (66/158 MB)` at most once per 5 % step and a final `id: installed from mirror|upstream|ollama` / `id: already installed`; any `ModelDownloadError` → message on stderr and exit 1 after the other models were tried.
- `remove ID...`: removes (unknown id → exit 2); prints `id: removed` / `id: nothing to remove`.
- `verify [ID...]`: recomputes the statuses (default all non-llm models), prints them, exit 1 when any is `corrupt`.
The models directory is `get_config().paths.models_dir`.

## Acceptance tests (CPU, no network: `httpx.MockTransport`, fake `hf_download`, temp dirs; make small fake zips/files in the tests and patch the catalog entries' hashes to match them)
1. **Catalog file**: `load_catalog()` on the real `config/models.toml` returns 8 entries in the order above; every entry passes `validate_for_format`; all `zip`/`file` entries have 64-hex `sha256`, an `https://github.com/Nawid3333/OmniScan/releases/download/models-v1/` mirror URL and `bytes > 0`; ids are unique; exactly ids 1–3 are `required`; the `inpaint-big-lama` sha256 equals `InpaintConfig().lama_sha256`; `detector-comic-text-bubble.upstream_repo == DetectConfig().repo` and the two OCR upstream repos equal `OcrConfig().det_repo` / `rec_repo`; the `ollama`/`cloud` names match the model names in `config/translation_profiles.toml` and `config/judge.toml` (translategemma:12b, gemma4:12b, gemma4:31b-cloud).
2. **Catalog merging/validation**: a machine file replaces an entry with the same id and adds a new one; duplicate ids in one file, an unknown key, a `zip` entry without `sha256` → `ValueError` with the field name.
3. **Status**: every branch of `model_status` (missing folder; folder without/with a wrong-hash `.installed.json` → corrupt; file with wrong size → corrupt; file with wrong hash → corrupt; installed; ollama installed/missing/unknown; cloud); `resolve_model_path` returns the path only when installed.
4. **Download from the mirror**: a fake zip served by the mock transport → folder `<id>/` with the files, `.installed.json` (source `mirror`), no `.part`/zip left, progress callback monotonic and ending at the total; return value `"mirror"`; a second call → `"already installed"` and zero requests.
5. **Integrity**: wrong bytes served → the `.part` is deleted and the upstream fallback runs; both failing → `ModelDownloadError` containing both messages; zip-slip member (`../evil.txt`, an absolute path) → error and nothing extracted; a mirror 404 → upstream used (`"upstream"`, fake `hf_download` called with the catalog repo/revision and `local_dir=<models_dir>/<id>`); a `file` entry from upstream with a wrong hash → error and the file removed.
6. **Ollama**: a mocked NDJSON pull with progress lines then `{"status":"success"}` → returns `"ollama"` and reports progress; an `{"error": "..."}` line → `ModelDownloadError`; `cloud` entry → error message mentions Ollama Cloud; `remove_model` for ollama sends `DELETE /api/delete` with the tag.
7. **Remove**: folder/file removal, `False` when nothing to remove.
8. **CLI** (`CliRunner`, temp config with `models_dir` in a temp dir, catalog patched to small fake entries via the machine-file override or a monkeypatched `default_catalog_path`): `list` text and `--json` (parse it, check keys and statuses, Ollama unreachable → `unknown`); `download` a fake zip model → installed and re-listed as installed; `download nope` → exit 2; `--required`; `remove`; `verify` exit 1 with a corrupt file; `tests/unit/test_docs.py` and the whole suite stay green.

## Out of scope
Changing how the pipeline finds its models (U2b), a GUI, auto-update of the application, GitHub authentication for the private mirror (the upstream fallback covers it), resuming partial downloads, bandwidth limits, GGUF/llama.cpp models.

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_models_catalog.py tests/unit/test_models_store.py tests/unit/test_models_download.py tests/unit/test_models_cli.py tests/unit/test_docs.py -q
uv run pytest -q
uv run ruff format . && uv run ruff check .
uv run pyright
```
Optional single real check for the report (skip if it takes more than a few minutes): `uv run omniscan models list` and `uv run omniscan models verify` on the real `V:\OmniScan\models` folder (LaMa is there, the HF models are in the Hugging Face cache and will show as `missing` — that is expected until U2b).

## Report
`docs/reports/U2a.md`: Changes, Tests (commands + results), Deviations, Questions. **Commit early** (`U2a: WIP catalog+store`) once Part 2 and its tests pass, again after Part 3; the final commit is `U2a: model catalog and manager`. You have 150 tool calls in total. If anything is unclear: stop, write the question under Questions, commit, and end.

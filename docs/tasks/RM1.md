# RM1 — remove the acquisition subsystem (extract.pics, relay, `omniscan acquire`)

**Owner:** GLM builder · **Branch:** `RM1` · **Worktree:** `V:\OmniScan-wt\RM1` · run with `--max-turns 300`
Read `CLAUDE.md` first. This card is mostly deletion, not new design — the acceptance evidence is what
counts, but the file list below is close to a full spec since the boundary matters more than usual here.

## Why
The owner now supplies raw/reference chapters as local files/folders/archives (`omniscan import`, and the
GUI import page built in card U3e) instead of through automated site acquisition. The `acquire/` subsystem
(built around the extract.pics API, a Cloudflare Worker webhook relay, and `omniscan acquire`) has no
remaining purpose and should come out. Owner's own words: *"you can now delete the extract pics api and
stuff because now the manhwas and manga files will be provided by the user."*

## What to delete
- **`src/omniscan/acquire/`** — the whole package: `cli.py`, `completeness.py`, `download.py`,
  `extractor.py`, `extractpics.py`, `filters.py`, `pagerun.py`, `plan.py`, `run.py`, `selection.py`,
  `sources.py`, `__init__.py`. **Exception: `drm.py`'s `NOTICE` constant** — see "What to keep" below;
  everything else in `drm.py` (`DRM_PLATFORMS`, `drm_platform`, `check_allowed`, `DrmPlatformError`) is
  acquire-specific URL-checking logic with no remaining caller once `acquire/` is gone (`omniscan import`
  and the GUI import page only ever take a local folder or archive, never a URL — confirm this with a repo
  search before deleting, don't take it on faith).
- **`relay/`** (repo root) — the Cloudflare Worker TypeScript project (`package.json`, `src/`, `test/`,
  `wrangler.jsonc`, etc.) built solely to receive extract.pics webhooks. Delete the whole directory.
- **`.github/workflows/relay-deploy.yml`** — the CI workflow that deployed it.
- **`src/omniscan/cli.py`** — remove `from omniscan.acquire.cli import acquire_app` and
  `app.add_typer(acquire_app, name="acquire")`.
- **`tests/unit/test_acquire_*.py`** (all of them — `test_acquire_cli.py`, `test_acquire_run.py`,
  `test_acquire_plan.py`, `test_acquire_selection.py`, `test_acquire_sources.py`,
  `test_acquire_completeness.py`, `test_acquire_extractpics.py`, `test_acquire_pagerun.py`,
  `test_acquire_extractor.py`, `test_acquire_download.py`, `test_acquire_drm.py`,
  `test_acquire_filters.py`, `test_acquire_mutation_gaps.py` — check for others with the same prefix).
- **`tests/mutants/acquire/`** (`download.py`, `rules.py`) — mutation-testing artifacts for the deleted code.
- **`docs/EXTRACTPICS_API.md`** — delete entirely (it documents an API client that no longer exists).

## What to keep (relocate, don't delete)
- The DRM notice string is still load-bearing: `src/omniscan/gui/import_view.py` (U3e, just merged) does
  `from omniscan.acquire.drm import NOTICE` and shows it on the import page (the legal "you are responsible
  for this content" disclaimer, which applies to user-supplied files just as much as it applied to acquired
  ones). Create a small new module — `src/omniscan/legal.py` is a reasonable name, your call — holding just
  `NOTICE = "You are responsible for having the right to download and process this content."` with a
  one-line module docstring, and update the import in `src/omniscan/gui/import_view.py` and
  `tests/gui/test_import_view.py` (search for other `NOTICE`/`acquire.drm` references first — don't assume
  these are the only two).

## `core/config.py` (director-owned — do not edit; described here so you know what's already gone)
By the time you rebase onto `main`, `src/omniscan/core/config.py` will no longer have `RelayConfig`, the
`relay: RelayConfig` field on `Config`, `extractpics_api_key`, or `relay_client_token` (the director removes
these directly, since they're in `core/**`). Your job: make sure nothing you keep still references them.
One known consumer outside `acquire/`: **`src/omniscan/doctor.py`'s `check_secrets`** currently checks
`secrets.extractpics_api_key` and `secrets.relay_client_token` — simplify it to only check
`secrets.ollama_api_key` (still present), matching this shape:
```python
def check_secrets(secrets: Secrets) -> CheckResult:
    """Check optional secrets; never prints values."""
    if secrets.ollama_api_key is None:
        return CheckResult("secrets", "WARN", "not set: OLLAMA_API_KEY")
    return CheckResult("secrets", "OK", "all optional secrets set")
```
Search the repo yourself for any other `RelayConfig`/`extractpics_api_key`/`relay_client_token`/`cfg.relay`
reference beyond `doctor.py` and `acquire/` — fix or delete whatever you find. If your local `core/config.py`
(before the director's trim lands) still has these fields when you start, that's fine — write your code
assuming they will be gone, and don't rely on them existing.

## Docs to update
- **`docs/USER_GUIDE.md`**: remove the entire `### omniscan acquire` section and its example commands;
  remove the `acquire.json`/`acquire_selection.json`/`accepted.json`/`sources.toml` rows from the data-layout
  table (or reword rows that also apply to `omniscan import` to drop the acquire half); remove
  `EXTRACTPICS_API_KEY=...` from the secrets example.
- **`docs/ARCHITECTURE.md`**: the raws-directory comment says "written only by `acquire`" — update to say
  `omniscan import` / the GUI import page; the secrets line lists `EXTRACTPICS_API_KEY` — remove it.
- **`docs/PLAN.md`, `docs/PRODUCT_SPEC.md`**: these are large historical planning documents with many
  acquire/extract.pics/relay mentions woven through them (architecture diagrams, milestone tables, the
  acquisition-layer section, task descriptions). You do not need to scrub every historical mention — these
  read as a project journal and past-tense mentions of what was built are fine to leave — but anything
  phrased as current/future/actionable (e.g. describing `acquire` as part of the present architecture, or
  listing acquire-related work as still to do) should be corrected or clearly marked superseded. Use your
  judgement; say in the report what you changed vs left as historical record.
- **`docs/OPEN_QUESTIONS.md`**: add a row to the "Decisions (answered)" table recording this — date, "raws
  are now exclusively user-supplied (`omniscan import` / GUI import page); the acquisition subsystem
  (extract.pics client, webhook relay, `omniscan acquire`) was removed" — and drop/update A3/A4 (the
  Cloudflare/extract.pics credential questions) since they're moot now.
- **`docs/CHECKPOINT.md`, `docs/HANDOFF.md`, `docs/DECISIONS.md`, `docs/GPU_NOTES.md`, `docs/NEXT.md`**:
  grep each for acquire/extract.pics/relay mentions; fix anything that would mislead a reader about the
  *current* state of the repo (these are less "historical journal" than PLAN.md/PRODUCT_SPEC.md, so lean
  toward correcting rather than leaving stale).

## Files you may create / modify
Everything named above. Do not edit `src/omniscan/core/**` (see the config.py section above — the director
handles that part). Do not touch `src/omniscan/importer/**`, `src/omniscan/gui/import_view.py` beyond the
one import-line change, or anything from cards CM1/GL1/U3c/U3e that isn't directly about the `NOTICE` import.

## Acceptance evidence (put it in `docs/reports/RM1.md`)
1. `git grep -i "extractpics\|extract\.pics" -- src tests docs` (excluding `docs/PLAN.md`/`docs/PRODUCT_SPEC.md`
   historical mentions you deliberately kept, which you list explicitly) returns nothing unexpected — paste
   the command and its output (or the filtered output) in the report.
2. `git grep -rn "omniscan\.acquire\|acquire_app"` (outside `docs/PLAN.md`/`docs/PRODUCT_SPEC.md`) returns
   nothing.
3. `uv run --frozen pytest -q -m "not gpu"` green with no `test_acquire_*` collected (confirm the count of
   deselected/collected tests dropped by roughly the number of deleted test files, and say why any remaining
   drop/gain is expected).
4. `ruff format . && ruff check .` clean, `pyright` clean (this will only fully succeed once the director's
   `core/config.py` trim is applied — note in the report what you verified on your own branch and that the
   director will do a final combined check before merging).
5. `docs/USER_GUIDE.md`/`docs/ARCHITECTURE.md` no longer describe `omniscan acquire` as available; a fresh
   `uv run --frozen omniscan --help` shows no `acquire` subcommand.
6. State plainly in the report which doc mentions you left as historical record (PLAN.md/PRODUCT_SPEC.md)
   and which you corrected, so the director can spot-check the judgement call.

## Commands to run before finishing
```bash
uv run --frozen pytest -q -m "not gpu"
uv run --frozen ruff format . && uv run --frozen ruff check .
uv run --frozen pyright
uv run --frozen omniscan --help
```

## Report
`docs/reports/RM1.md`: Changes, Tests (commands + results), Deviations, Questions. Commit early
(`RM1: WIP removal`), final commit `RM1: remove acquisition subsystem`. You have 300 tool calls. If
anything is unclear (e.g. a doc section that's ambiguous to classify as historical vs. current), say so in
the report rather than guessing on something irreversible like a doc rewrite that loses information.

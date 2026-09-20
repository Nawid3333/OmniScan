# F2a — Promo filter wired into the pipeline (tier 2: file-level in `ingest`, slice-level in `slice`)

**Owner:** GLM builder · **Branch:** `F2a` · **Worktree:** `V:\OmniScan-wt\F2a` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read `docs/PRODUCT_SPEC.md` section 4 (finding + design), all of `src/omniscan/filter/` (`decide.py`, `hashing.py`), `src/omniscan/ingest/` (`__init__.py` `ingest_chapter`, `stage.py`, `strip.py`, `layout.py`), `src/omniscan/slicer/stage.py` + `strategies.py` (`slice_with_strategy`), `src/omniscan/core/schemas.py` (`IngestArtifact.filtered_files`, `SourceFile`, `Slice.filtered`, `FilterArtifact`, `FilterDecision`), `src/omniscan/core/config.py` (`FilterConfig`: `enabled`, `threshold` — **director-owned, already present; do not edit `core/**`**), the `filter` command group in `src/omniscan/cli.py`, and the tests `tests/unit/test_filter_decide.py`, `test_filter_hashing.py`, `test_ingest*.py`, `test_slicer_slice.py`, `test_cli.py`.

## Why
The owner's promo/credit filtering must work when a series is run: scanlation groups' banners, credit pages and ads must not reach OCR, translation or the exported chapter. **Finding (2026-09-20): today nothing applies the filter.** `omniscan filter run` computes decisions into `filter.json` and copies matches to `_filtered/`, but `SourceFile.filtered` / `Slice.filtered` are never set by anything, so `omniscan run` translates and exports the promo pages. This card wires **tier 2** (perceptual hash against the user's example images) into the pipeline:
- **file level, in `ingest`**: a raw file whose dHash matches an example is left out of the strip (and copied to `_filtered/`, never deleted), its name goes to `IngestArtifact.filtered_files`;
- **slice level, in `slice`**: after slicing, each slice's dHash is compared with the examples; a match sets `Slice.filtered = True` (detect, OCR-relevant stages and export already skip filtered slices);
- **`filter.json` becomes the user's override file**: only manual entries (`method="manual"`), read by both stages as an *input* — a restore re-runs what depends on it. Tiers 1 (post-OCR text patterns) and 3 (position heuristics) are cards F2b/F2c.

## Files you may create / modify
- `src/omniscan/filter/apply.py` (create), `src/omniscan/filter/hashing.py` (add `dhash_tensor`), `src/omniscan/filter/__init__.py` (exports), `src/omniscan/filter/decide.py` (only if you need small helpers; existing functions keep their behaviour and tests)
- `src/omniscan/ingest/__init__.py`, `src/omniscan/ingest/stage.py` (file-level check), `src/omniscan/slicer/stage.py` (slice-level check)
- `src/omniscan/cli.py` (modify **only** the `filter` group: `run` reworked, `restore` updated, new `add`)
- any code that assumes `SourceFile.index == position in files` (grep `\.index` on `ingest.files` users in `src/omniscan/**`, incl. `web/app.py` and `gui/services/library.py` — fix what breaks; see "Indices")
- tests: `tests/unit/test_filter_apply.py`, `test_filter_hashing.py` (extend), `test_ingest_filter.py`, `test_slicer_filter.py`, `test_filter_cli.py` (create); existing tests keep passing (adjust only where the old `filter run` behaviour is intentionally replaced)
- `docs/USER_GUIDE.md` (rewrite the promo-filter section with the real commands), `docs/reports/F2a.md`
Anything else is off-limits (especially `src/omniscan/core/**`).

## Interfaces (exact)
```python
# filter/hashing.py (addition)
def dhash_tensor(image: torch.Tensor, hash_size: int = 8) -> int: ...
    """uint8 [3, h, w] (any device) -> the same kind of hash as `dhash`: grayscale (ITU-R 601 luma 0.299/0.587/0.114),
    bilinear antialiased resize to (hash_size, hash_size + 1) rows x cols, then the same MSB-first left>right comparison.
    One host transfer of hash_size*(hash_size+1) values."""

# filter/apply.py
@dataclass(frozen=True, slots=True)
class Overrides:
    files_restored: frozenset[int]; files_forced: frozenset[int]        # raw-file indices (position in `list_images(raw_dir)`)
    slices_restored: frozenset[int]; slices_forced: frozenset[int]      # slice indices

def load_overrides(filter_json: Path) -> Overrides: ...              # manual entries only; last entry per (target, index) wins; a missing or invalid file -> empty Overrides
def file_verdict(index: int, similarity_score: float, matched: str | None, threshold: float, overrides: Overrides) -> Literal["filtered", "keep"]: ...
def slice_verdict(index: int, similarity_score: float, matched: str | None, threshold: float, overrides: Overrides) -> Literal["filtered", "keep"]: ...
def examples_fingerprint(examples: Sequence[ExampleHash]) -> str: ...     # sha256 hex of "name:hash" lines in order

# ingest_chapter gains keyword-only parameters
def ingest_chapter(raw_dir, series, chapter, cache_dir, quality=95, *, examples: Sequence[ExampleHash] = (), threshold: float = 0.90,
                   overrides: Overrides | None = None, filtered_dir: Path | None = None) -> IngestResult
```

## Definitions (exact)
- **Overrides**: `filter.json` may hold `FilterDecision`s; only those with `method == "manual"` count. `decision == "restored"` → restored; `decision == "filtered"` → forced. The **last** manual entry for a (target, index) decides (a later restore beats an earlier force and vice versa). `target == "file"` indices are positions in `list_images(raw_dir)` (natural order, **independent of filtering**); `target == "slice"` indices are `Slice.index`.
- **Verdict**: forced → `"filtered"`; restored → `"keep"`; otherwise `"filtered"` when `similarity_score >= threshold` and `matched is not None`, else `"keep"`. (`similarity_score` = the best example's `similarity(hash, example.hash)`, `0.0` without examples.) With `cfg.filter.enabled == False` **no automatic match counts** (forced entries still do): pass an empty `examples` list.
- **`dhash_tensor`**: for the same picture it must land within Hamming distance ≤ 6 of `dhash(PIL image)` (the two resize filters differ) — property test on ≥ 20 seeded synthetic images (noise, gradients, text-like stripes, a solid banner); identical tensors give identical hashes; a CUDA-or-CPU tensor gives the same value; 1-row/1-column inputs do not crash.
- **`ingest_chapter` with a filter**: enumerate `list_images(raw_dir)` as before (`index` = position in that list); for each file compute `dhash` of the **raw original** (`Image.open(path)`), take `best_match` against `examples`, apply `file_verdict`. Filtered files are **not converted, not laid out and not listed in `files`**; their names (file names, in raw order) go to `filtered_files`; each is copied byte-for-byte to `filtered_dir / name` when `filtered_dir` is given (created on demand, existing identical file left alone; never delete/modify the source). Kept files keep their **original raw index** in `SourceFile.index` (gaps are allowed) and their cache file name uses that index. If every file would be filtered → `ValueError("every image of <raw_dir> matched a promo example")`. Without examples and without overrides the output is **identical to today's** (same artifact bytes except the new empty `filtered_files` list) — regression test.
- **`IngestStage`**: `inputs` = the raw images **plus** `filter.json` (when it exists) plus the example image files of `promo_examples/global` and `promo_examples/<series>` (so adding or changing an example invalidates the stage); `config_subset` = `{"quality": 95, "filter": cfg.filter.model_dump(), "examples": examples_fingerprint(...)}`; `version` 1 → 2 (comment `# 2: file-level promo filter`); `run` loads the examples once (`load_examples(cfg.paths.promo_examples, series)`, skipped when the filter is disabled), reads the overrides from `ctx.paths.artifact("filter.json")`, passes `filtered_dir=ctx.paths.filtered_dir`, and adds `"filtered_files": float(len(...))` to the metrics.
- **`SliceStage`**: after `slice_with_strategy`, when the filter is enabled and there are examples (or forced overrides): for every slice that is not `blank` compute `dhash_tensor(strip[:, s.y0:s.y1, :])`, `best_match`, `slice_verdict` → set `filtered=True` on that slice (build new `Slice` objects with `model_copy(update=…)`; `SlicesArtifact` is otherwise unchanged); write the images of filtered slices to `paths.filtered_dir / f"{chapter}_slice_{index:04d}.jpg"` **only as the old `decide_slices` did** (one JPEG per filtered slice, quality 95) — needs one host transfer per *filtered* slice only. Forced overrides apply even without examples. `inputs` additionally include `filter.json` and the example files; `config_subset` = `{**cfg.slicer.model_dump(), "filter": cfg.filter.model_dump(), "examples": <fingerprint>}`; `version` 3 → 4 (`# 4: slice-level promo filter`); metrics gain `"filtered": float(count)`.
- **Indices**: after this card `ingest.files` may have gaps in `index`. Search for code that relies on `files[i].index == i` or on `len(files)` being the raw file count (web API `source_names`, GUI library service, eval, preview, `_assemble_strip` in `cli.py`, `Slice.source_files` users) and make it use the real `index`/`name`; add a regression test for each fix you make.
- **CLI `filter` group** (the old `run` computed decisions into `filter.json`; that file is now overrides only):
  - `filter run SERIES [-c/--chapter NAME]... [--json]`: run `ingest` and `slice` (with `--force` semantics **off**: stages that are up to date stay skipped) for the chapters (all chapters by default) exactly as `omniscan slice` does, then print per chapter `filtered files: N, filtered slices: M` and a total line `filter: <files> file(s), <slices> slice(s) filtered in <chapters> chapter(s)` (`--json`: `{"series", "chapters": [{"chapter", "files": [names], "slices": [indices]}], "totals": {...}}` read from `ingest.json`/`slices.json`).
  - `filter restore SERIES CHAPTER TARGET INDEX`: unchanged CLI, but now **creates `filter.json` when it does not exist** (no more `FileNotFoundError`) and the printed text says `will apply on the next run`. `TARGET` is `file` or `slice`; a new sibling `filter force SERIES CHAPTER TARGET INDEX` appends a manual `filtered` decision the same way.
  - `filter add SERIES PATH [--global] [--name NAME]`: copy an image (JPEG/PNG, decodable by Pillow) into `<promo_examples>/<series>/` (or `global/`), refusing to overwrite (exit 2), printing the destination and `run 'omniscan filter run <series>' to apply it`.
  - `_assemble_strip` and the old `decide_*` functions may stay if other code still uses them; delete them only if nothing does (say so in the report).

## Acceptance tests (CPU, no models, synthetic images from Pillow/torch)
1. **`dhash_tensor`**: the properties above; a golden pair: for a horizontal gradient image `dhash` and `dhash_tensor` are both `0` bits pattern consistent (compute the actual values from the real functions first and pin them; distance ≤ 6).
2. **`load_overrides` / verdicts / fingerprint**: last-write-wins for both targets, `automatic` entries ignored, corrupt/missing file → empty, thresholds at the boundary (`score == threshold` filters, just below keeps), forced beats a non-match, restored beats a match, `examples_fingerprint` changes when a hash or a name changes and is stable otherwise.
3. **`ingest_chapter`**: 5 synthetic pages, pages 1 and 3 are near-copies of a promo example (add mild noise/resize): they end up in `filtered_files`, not in `files`, copied to `_filtered/` byte-identically, the strip height equals the sum of the 3 kept pages, kept files have original indices `[0, 2, 4]`, cache file names use those indices; all filtered → `ValueError`; override `files_restored={1}` keeps page 1; `files_forced={0}` drops page 0 without examples; disabled filter (empty examples) → identical to the pre-change artifact; the strip built by `build_strip` on that artifact has the right pixels (compare rows with the kept pages).
4. **`IngestStage`**: runs with a tmp `Config` (`paths.promo_examples`), writes `filtered_files`, metrics; adding an example file or editing `filter.json` makes the stage re-run (manifest invalidation: assert `run_stage` status `done` after the change and `skipped` without it); `filter.enabled = False` ignores examples.
5. **`SliceStage`**: a strip whose middle slice is a promo banner (build the ingest artifact by hand + a `load_strip` monkeypatch, as `tests/unit/test_slicer_slice.py`/stage tests do): that slice gets `filtered=True`, the others do not, its JPEG lands in `_filtered/`, `blank` slices are skipped, forced/restored overrides work by slice index, no examples and no overrides → no change and no `.cpu()`-heavy work (spy: `dhash_tensor` never called), invalidation on example/`filter.json` change.
6. **Downstream**: with a filtered slice, `detect` produces no regions inside it and `export` skips it (use the existing fake-model stage tests as a template — a pure unit test with fakes is enough); the web API `source_names` and the GUI `load_chapter_view` still produce the right names/tiles when `files` has index gaps.
7. **CLI**: `filter add` (copy, refusal to overwrite, non-image → exit 2), `filter run` on a synthetic library with a promo example (prints counts, `--json` parses, second run is a no-op), `filter restore …` then `filter run` → the restored file/slice is kept, `filter force`, docs test green. **Regression**: the whole CPU suite (`-m "not gpu"`) stays green; the golden E2E test (`tests/unit/test_e2e_synthetic.py`, GPU) is not affected because it has no examples — run it at the end if the machine allows and report the result.

## Out of scope
Tier 1 (post-OCR text patterns) and tier 3 (position heuristics) — cards F2b/F2c; embedding-based matching; the web "Filtered" view changes; batch scheduling across series; a GUI for examples.

## Commands to run before finishing
```bash
uv run --frozen pytest tests/unit/test_filter_apply.py tests/unit/test_filter_hashing.py tests/unit/test_ingest_filter.py tests/unit/test_slicer_filter.py tests/unit/test_filter_cli.py tests/unit/test_filter_decide.py tests/unit/test_docs.py -q
uv run --frozen pytest -q -m "not gpu"
uv run --frozen ruff format . && uv run --frozen ruff check .
uv run --frozen pyright
```

## Report
`docs/reports/F2a.md`: Changes, Tests (commands + results), Deviations, Questions. **Commit early** (`F2a: WIP overrides + ingest filter`) once the filter module, `dhash_tensor` and the ingest change with their tests pass; then the slice part; the final commit is `F2a: promo filter wired into ingest and slice`. You have 150 tool calls in total. If anything is unclear: stop, write the question under Questions, commit, and end.

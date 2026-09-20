# O1c — OCR qualification suite: which model is the default for which language?

**Owner:** GLM builder · **Branch:** `O1c` · **Worktree:** `V:\OmniScan-wt\O1c` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read `docs/PRODUCT_SPEC.md` sections 1–2, `src/omniscan/cli.py` `cmd_eval` (how a chapter is scored: `load_truth`, `load_english_pages`, `score_chapter`), `src/omniscan/eval/{truth,score,metrics}.py`, `src/omniscan/pipeline/runner.py` (`run_pipeline`), `src/omniscan/gpu/groups.py` (`build_vram_manager`), `src/omniscan/core/config.py` (`OcrConfig`: `engine`, `det_model`, `rec_model`, `lang`), `config/models.toml` (OCR entries: ids, `langs`, `recommended_for`), `tests/unit/test_eval_cli.py`, `test_eval_score.py`, `test_pipeline_runner.py` (fakes).

## Why
Owner rule: **defaults come from measurements, not from names** — "v6 biggest" is only a default where it wins. The catalog holds every OCR model; the engines (PP-OCR with any model ids, manga-ocr, PaddleOCR-VL) exist. This card builds the **qualification suite**: run every candidate configuration over the Pepper&Carrot ground-truth chapters per language, score them, print a table and a recommendation. It is also what the model-watch job (card W1) runs when a new upstream model appears. Known facts that shape the metrics (2026-09-20):
- Ground truth: `data/translated-check/<Series>/<Chapter>/truth/<lang>/*.svg` (Korean `kr`, Japanese `ja`, Chinese `cn`), scored by `omniscan.eval`.
- **Box-level CER is not comparable across engines**: the truth has finer boxes than our regions on dense pages (Japanese episode 6: 159 truth boxes vs 91 regions → micro CER 0.84 although the text is right). The **primary metrics are `ocr_chrf_mean` (page-level chrF) and `recall_chars`**; `cer_micro` is reported but never used for a decision.
- Measured so far: Korean episode 6, page chrF: v5 server det + v5 Korean rec 0.475, v6 medium det+rec 0.105; Japanese episode 6: manga-ocr 2025 page chrF 0.686; PaddleOCR-VL reads stylised lettering best but costs ~3.6 s per region.

## Files you may create / modify
- `src/omniscan/eval/qualify.py` (create — library, no CLI), `scripts/qualify_ocr.py` (create — thin runner), `config/qualification.toml` (create — candidates and datasets)
- `tests/unit/test_eval_qualify.py` (create — all logic with fakes, no GPU, no data)
- `docs/benchmarks/ocr-qualification.md` (create: the method, the decision rule, and an empty results section that the director fills after a real run), `docs/reports/O1c.md`
Anything else is off-limits (especially `src/omniscan/core/**`, `config/models.toml`, `eval/score.py`/`truth.py` — read-only).

## Interfaces (exact)
```python
@dataclass(frozen=True, slots=True)
class Candidate:
    id: str; engine: Literal["ppocr", "manga_ocr", "paddleocr_vl"]
    det_model: str | None; rec_model: str | None; langs: tuple[str, ...]
@dataclass(frozen=True, slots=True)
class Dataset:
    lang: str; series: str; chapters: tuple[str, ...]; truth: str        # truth = the folder under truth/ (kr, ja, cn)
@dataclass(frozen=True, slots=True)
class Measurement:
    candidate: str; lang: str; series: str; chapter: str
    recall_chars: float; chrf: float | None; cer_micro: float | None
    seconds: float; peak_vram_gib: float | None; regions: int; error: str | None

def load_plan(path: Path) -> tuple[list[Candidate], list[Dataset]]: ...        # config/qualification.toml; ValueError with the entry number/field on bad input
def select(candidates: Sequence[Candidate], datasets: Sequence[Dataset], *, lang: str | None, only: Sequence[str] | None) -> list[tuple[Candidate, Dataset]]: ...   # candidate applies to a dataset when dataset.lang in candidate.langs
def candidate_config(cfg: Config, cand: Candidate, dataset: Dataset, work_root: Path) -> Config: ...      # cfg copy: paths.work_root, ocr.engine/det_model/rec_model/lang
def summarize(measurements: Sequence[Measurement]) -> dict[tuple[str, str], Summary]: ...                # (lang, candidate) -> means over chapters (errors excluded but counted)
def recommend(summaries, *, lang: str, current_default: str, min_gain: float = 0.02, max_recall_loss: float = 0.02) -> Recommendation: ...
def render_markdown(summaries, recommendations, *, installed: Mapping[str, bool] | None = None) -> str: ...
def run_qualification(pairs, cfg, *, run_candidate: RunFn, models_installed: Callable[[Candidate], list[str]] | None = None, log: Callable[[str], None] = ...) -> list[Measurement]: ...
```
`RunFn = Callable[[Config, Candidate, Dataset, str], Measurement]` is what does the real work (`default_run_candidate` in the same module; tests inject fakes).

## Definitions (exact)
- **`config/qualification.toml`**: `[[candidate]]` tables (`id`, `engine`, optional `det_model`, `rec_model`, `langs = [..]`) and `[[dataset]]` tables (`lang`, `series`, `chapters = [..]`, `truth`); ids unique; unknown engine or a `langs` entry that no dataset uses is fine. Ship these entries: candidates `ppocr-v5-ko` (`ocr-det-ppocrv5-server` + `ocr-rec-korean-ppocrv5-mobile`, langs `ko`), `ppocr-v5-server-multi` (`ocr-det-ppocrv5-server` + `ocr-rec-ppocrv5-server`, langs `zh`, `en`, `ja`), `ppocr-v6-tiny`, `ppocr-v6-small`, `ppocr-v6-medium` (v6 det+rec of that size; langs `zh`, `en`, `ja`, `ko`), `manga-ocr-2025` (engine `manga_ocr`, `ocr-rec-manga-ocr-2025`, langs `ja`), `manga-ocr-base` (`ocr-rec-manga-ocr-base`, `ja`), `paddleocr-vl-1.6` (engine `paddleocr_vl`, `ocr-vl-1.6`, langs `ko`, `ja`, `zh`, `en`); datasets `ko` → `PepperCarrotKR` chapters `Episode 06`, `Episode 09`, truth `kr`; `ja` → `PepperCarrotJA` `Episode 06`, `09`, `12`, `22`, truth `ja`; `zh` → `PepperCarrotCN` the same four chapters, truth `cn`. (Check every model id exists in `config/models.toml`; a test does.)
- **`select`**: all (candidate, dataset) pairs with `dataset.lang in candidate.langs`, filtered by `lang` and by candidate ids in `only`; order = candidates in file order, then datasets in file order.
- **`candidate_config`**: `cfg.model_copy(deep=True)` then `paths.work_root = work_root`, `ocr.engine`, `ocr.det_model`, `ocr.rec_model`, `ocr.lang = dataset.lang if dataset.lang in ("ko","zh","ja","en") else cfg.ocr.lang` (`cn` truth folder maps to language `zh`: the dataset's `lang` field is already `zh`), nothing else changes. The original `cfg` is not mutated.
- **`default_run_candidate(cfg, cand, dataset, chapter)`**: (1) in `cfg.paths.work_root` (a dedicated qualification work root, default `<work_root>_qual`, decided by the script) make sure `ingest`, `slice`, `detect` are done for the chapter — `run_pipeline(cfg, series, [chapter], stages=["ingest","slice","detect","ocr"], gpu=..., force=False)`; the vision stages are skipped when up to date, the `ocr` stage re-runs because its config subset changed; (2) reset `torch.cuda.reset_peak_memory_stats()` before and read `torch.cuda.max_memory_allocated()/2**30` after when CUDA is available; time the whole call with `time.perf_counter()`; (3) load `ocr.json`, `ingest.json`, the truth (`load_truth(check_dir, dataset.truth, ingest)` with `check_dir = cfg.paths.library_root.parent / "translated-check" / series / chapter / "truth"`, `load_english_pages`) and `score_chapter(...)` → `Measurement(recall_chars=report.recall_chars, chrf=report.ocr_chrf_mean, cer_micro=report.cer_micro, regions=len(regions.regions), ...)`; (4) **every** exception → a `Measurement` with `error="<ExceptionType>: <message>"` and zeros/None (a missing model must not stop the run: the message tells which id is missing — `omniscan models download <id>`); the GPU manager is released after each candidate (`gpu.release()` in `finally`).
- **`summarize`**: per (lang, candidate) the mean of `recall_chars`, `chrf` (ignoring None), `cer_micro`, `seconds` per chapter, max `peak_vram_gib`, number of chapters and number of errors; a candidate with errors on **all** chapters has `chrf=None` and is never recommended.
- **`recommend(summaries, lang, current_default)`**: `current_default` = the candidate id that is the default today for that language (script option `--default ko=ppocr-v5-ko --default ja=… --default zh=…`; when omitted the language's default is the candidate whose model ids equal today's `OcrConfig` defaults for `ko` and, for other languages, `ppocr-v5-server-multi`). Best = highest mean `chrf` among candidates of that language with no errors; **`PROMOTE <best>`** only when `best != current_default` **and** `best.chrf - default.chrf >= min_gain` **and** `default.recall_chars - best.recall_chars <= max_recall_loss`; otherwise **`KEEP <current_default>`** with the reason (`"best candidate <id> gains only 0.013 chrF"`, `"no data for the default"`, …). Ties on chrf → the higher `recall_chars`, then the smaller `seconds`.
- **`render_markdown`**: for each language a heading `## <lang> (<series>, <n> chapter(s))`, a table `candidate | chrF | char recall | CER (box, not comparable) | s/chapter | VRAM GiB | errors` sorted by chrF descending (None last), the default marked with `★`, a line `Recommendation: KEEP …|PROMOTE …`, and — when `installed` is given — a note listing candidates skipped because a model is not installed (`omniscan models download …`). Numbers rounded to 3 decimals; deterministic output (golden-tested).
- **`scripts/qualify_ocr.py`**: options `--plan config/qualification.toml`, `--lang`, `--only ID[,ID…]`, `--chapters N` (first N chapters per dataset), `--default LANG=ID` (repeatable), `--json OUT.json`, `--markdown OUT.md`, `--dry-run` (print the pairs and the missing models, run nothing), `--download` (download missing models through `models.download.download_model` after one confirmation prompt listing ids and sizes; `--yes` skips the prompt). Exit codes: 0 done, 1 when **every** measurement failed, 2 for bad arguments. It prints the markdown to stdout and a progress line per (candidate, chapter) to stderr. It never writes into the normal `work_root`.

## Acceptance tests (CPU, fakes, tmp dirs; no GPU, no models, no `data/`)
1. `load_plan` on the shipped `config/qualification.toml`: 8 candidates, 3 datasets, every candidate id unique, **every `det_model`/`rec_model` exists in `config/models.toml`** with a matching `role`, every dataset's series/chapters non-empty; bad files (duplicate id, missing `engine`, unknown engine, empty `chapters`, wrong types) → `ValueError` naming the entry.
2. `select`: language filter, `only`, order, a candidate without matching dataset yields nothing.
3. `candidate_config`: the fields set, the input config untouched, `cn` → `zh` mapping via the dataset's `lang`.
4. `run_qualification` with a fake `RunFn` (returns scripted `Measurement`s, records calls): call order, the exception in one candidate becomes an `error` measurement while later pairs still run, `models_installed` returning missing ids → the pair is skipped with a log line and an `error="missing model: <id>"` measurement.
5. `summarize`/`recommend`: hand-made measurements: means over chapters, errors excluded and counted, all-errors candidate never best, PROMOTE exactly at the `min_gain` boundary (`>=`), KEEP when the gain is `0.019`, KEEP when the recall loss exceeds `0.02`, the tie-break order, the default itself best → KEEP, no data for the default → KEEP with that reason.
6. `render_markdown` golden text for a two-language example (one candidate with an error, one uninstalled, the default marked).
7. `default_run_candidate` with monkeypatched `run_pipeline`, `load_truth`, `load_english_pages`, `score_chapter`, `RegionsArtifact.load`/`IngestArtifact.load` (or tiny real artifacts in a tmp work dir): produces the `Measurement` fields from the fake report (`recall_chars`, `ocr_chrf_mean`, `cer_micro`), seconds > 0, GPU manager released even when scoring raises, the exception text lands in `error`.
8. Script (`subprocess` or calling `main(argv)` with monkeypatched `run_candidate`): `--dry-run` prints the pairs and exits 0 without running anything; bad `--default` → exit 2; all failing → exit 1; `--json`/`--markdown` files written; `--only` restricts. `tests/unit/test_docs.py` and the CPU suite stay green.

## Out of scope
Running the suite for real (the director does it and pastes the results into `docs/benchmarks/ocr-qualification.md`), changing `recommended_for`/defaults in `config/models.toml`, translation quality, detection quality, Chinese/Japanese truth-assignment fixes in `eval/`, the GitHub Action (card W1).

## Commands to run before finishing
```bash
uv run --frozen pytest tests/unit/test_eval_qualify.py tests/unit/test_docs.py -q
uv run --frozen pytest -q -m "not gpu"
uv run --frozen ruff format . && uv run --frozen ruff check .
uv run --frozen pyright
```

## Report
`docs/reports/O1c.md`: Changes, Tests (commands + results), Deviations, Questions. **Commit early** (`O1c: WIP plan + recommend`) once `load_plan`, `select`, `summarize`, `recommend` and their tests pass; the final commit is `O1c: OCR qualification suite`. You have 150 tool calls in total. If anything is unclear: stop, write the question under Questions, commit, and end.

"""OCR qualification (card O1c): score candidate OCR configurations against the Pepper&Carrot truth.

A candidate is one (engine, det_model, rec_model) combination from `config/qualification.toml`;
`select` pairs the candidates with the per-language datasets, `run_qualification` measures every
pair chapter by chapter, and `summarize`/`recommend`/`render_markdown` turn the measurements into
the table and the KEEP/PROMOTE verdict the director uses to pick a per-language default.

Metrics: page-level chrF (`ocr_chrf_mean`) and `recall_chars` decide; box-level `cer_micro` is
reported but never decides (the truth has finer boxes than our regions on dense pages, so it is
not comparable across engines). This suite is also what the model-watch job (card W1) runs when a
new upstream model appears.
"""

from __future__ import annotations

import sys
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import torch

from omniscan.core.config import Config, series_config
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import IngestArtifact, RegionsArtifact
from omniscan.eval.score import score_chapter
from omniscan.eval.truth import load_english_pages, load_truth
from omniscan.gpu.device import resolve_device
from omniscan.gpu.groups import build_vram_manager
from omniscan.models.catalog import load_catalog
from omniscan.models.store import model_status
from omniscan.pipeline.runner import run_pipeline

Engine = Literal["ppocr", "manga_ocr", "paddleocr_vl"]

_ENGINES: tuple[str, ...] = ("ppocr", "manga_ocr", "paddleocr_vl")
_LANGS = ("ko", "zh", "ja", "en")  # the languages OcrConfig.lang accepts (`cn` truth maps to zh)
_CANDIDATE_FIELDS = frozenset({"id", "engine", "det_model", "rec_model", "langs"})
_DATASET_FIELDS = frozenset({"lang", "series", "chapters", "truth"})


@dataclass(frozen=True, slots=True)
class Candidate:
    """One OCR configuration to measure: an engine plus its catalog model ids."""

    id: str
    engine: Engine
    det_model: str | None = None
    rec_model: str | None = None
    langs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Dataset:
    """One ground-truth set: a series' chapters scored against the truth folder `truth`."""

    lang: str
    series: str
    chapters: tuple[str, ...]
    truth: str


@dataclass(frozen=True, slots=True)
class Measurement:
    """One (candidate, chapter) run: the scores, the cost, and the error when it failed."""

    candidate: str
    lang: str
    series: str
    chapter: str
    recall_chars: float | None = None
    chrf: float | None = None
    cer_micro: float | None = None
    seconds: float = 0.0
    peak_vram_gib: float | None = None
    regions: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Summary:
    """Per (lang, candidate) means over the chapters (errors excluded but counted)."""

    lang: str
    candidate: str
    series: str
    chapters: int
    errors: int
    recall_chars: float | None
    chrf: float | None  # None when every chapter errored: such a candidate is never recommended
    cer_micro: float | None
    seconds: float | None
    peak_vram_gib: float | None


@dataclass(frozen=True, slots=True)
class Recommendation:
    """The verdict for one language: KEEP names the default, PROMOTE names the winner."""

    lang: str
    verdict: Literal["KEEP", "PROMOTE"]
    default: str  # the current default the verdict was measured against
    candidate: str  # the candidate the verdict names: the default (KEEP) or the winner (PROMOTE)
    reason: str


class RunFn(Protocol):
    """One (candidate, chapter) measurement; `gpu` is a manager `run_qualification` may share
    across a candidate's chapters/languages (see `default_run_candidate`)."""

    def __call__(
        self, cfg: Config, cand: Candidate, dataset: Dataset, chapter: str, *, gpu: Any | None = None
    ) -> Measurement: ...


ModelsInstalledFn = Callable[[Candidate], list[str]]


def _log_stderr(message: str) -> None:
    """Progress default: one line per event on stderr (stdout carries the markdown table)."""
    print(message, file=sys.stderr)


# ---------------------------------------------------------------- the plan


def load_plan(path: Path) -> tuple[list[Candidate], list[Dataset]]:
    """Candidates and datasets of a qualification plan; ValueError names the entry on bad input."""
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: invalid TOML: {exc}") from exc
    candidates = [
        _candidate(table, index, path) for index, table in enumerate(_tables(data, "candidate", path), 1)
    ]
    datasets = [_dataset(table, index, path) for index, table in enumerate(_tables(data, "dataset", path), 1)]
    seen: set[str] = set()
    for cand in candidates:
        if cand.id in seen:
            raise ValueError(f"{path}: duplicate candidate id {cand.id!r}")
        seen.add(cand.id)
    return candidates, datasets


def _tables(data: dict[str, object], key: str, path: Path) -> list[dict[str, object]]:
    tables = data.get(key, [])
    if not isinstance(tables, list) or not all(isinstance(table, dict) for table in tables):
        raise ValueError(f"{path}: [{key}] must be an array of tables")
    return tables


def _candidate(table: dict[str, object], index: int, path: Path) -> Candidate:
    unknown = sorted(set(table) - _CANDIDATE_FIELDS)
    if unknown:
        raise ValueError(f"{path}: candidate #{index}: unknown field(s): {', '.join(unknown)}")
    cid = table.get("id")
    if not isinstance(cid, str) or not cid:
        raise ValueError(f"{path}: candidate #{index}: id must be a non-empty string")
    where = f"{path}: candidate #{index} ({cid!r})"
    engine = table.get("engine")
    if not isinstance(engine, str):
        raise ValueError(f"{where}: engine must be one of: {', '.join(_ENGINES)}")
    if engine not in _ENGINES:
        raise ValueError(f"{where}: unknown engine {engine!r} (known: {', '.join(_ENGINES)})")
    langs = table.get("langs")
    if not isinstance(langs, list) or not all(isinstance(lang, str) and lang for lang in langs):
        raise ValueError(f"{where}: langs must be a list of language codes")
    return Candidate(
        id=cid,
        engine=cast("Engine", engine),
        det_model=_optional_id(table, "det_model", where),
        rec_model=_optional_id(table, "rec_model", where),
        langs=tuple(langs),
    )


def _optional_id(table: dict[str, object], field: str, where: str) -> str | None:
    value = table.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: {field} must be a catalog model id")
    return value


def _dataset(table: dict[str, object], index: int, path: Path) -> Dataset:
    unknown = sorted(set(table) - _DATASET_FIELDS)
    if unknown:
        raise ValueError(f"{path}: dataset #{index}: unknown field(s): {', '.join(unknown)}")
    where = f"{path}: dataset #{index}"
    values = {name: table.get(name) for name in ("lang", "series", "truth")}
    for name, value in values.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{where}: {name} must be a non-empty string")
    chapters = table.get("chapters")
    if not isinstance(chapters, list) or not chapters or not all(isinstance(c, str) and c for c in chapters):
        raise ValueError(f"{where}: chapters must be a non-empty list of chapter folder names")
    return Dataset(
        lang=str(values["lang"]),
        series=str(values["series"]),
        chapters=tuple(chapters),
        truth=str(values["truth"]),
    )


def select(
    candidates: Sequence[Candidate],
    datasets: Sequence[Dataset],
    *,
    lang: str | None,
    only: Sequence[str] | None,
) -> list[tuple[Candidate, Dataset]]:
    """All (candidate, dataset) pairs the candidate applies to, in file order of both.

    A candidate applies to a dataset when `dataset.lang in candidate.langs`; `lang` filters the
    datasets' language, `only` (when given, non-empty) the candidate ids.
    """
    keep = set(only) if only else None
    return [
        (cand, dataset)
        for cand in candidates
        for dataset in datasets
        if dataset.lang in cand.langs
        and (lang is None or dataset.lang == lang)
        and (keep is None or cand.id in keep)
    ]


def candidate_config(cfg: Config, cand: Candidate, dataset: Dataset, work_root: Path) -> Config:
    """A deep copy of `cfg` at the qualification work root, the series' settings then the candidate's OCR applied.

    The series' own `series.toml` is merged first (its non-OCR sections — slicer, detect, … — must
    apply exactly as a normal `omniscan run` on that series would see them), then the candidate's
    OCR overrides are the last word: a series' `[ocr]` section must never beat the candidate under
    test (bug O1e). A broken `series.toml` raises `SeriesConfigError`. Run the pipeline on the
    result with `merge_series_config=False` (see `default_run_candidate`) — note `core.stage.
    make_context` still re-merges per chapter until the core-side half of the fix recorded in
    docs/reports/O1e.md lands.
    """
    out = series_config(cfg, SeriesPaths.from_config(cfg, dataset.series).library_dir).model_copy(deep=True)
    out.paths.work_root = work_root
    out.ocr.engine = cand.engine
    out.ocr.det_model = cand.det_model
    out.ocr.rec_model = cand.rec_model
    if dataset.lang in ("ko", "zh", "ja", "en"):
        out.ocr.lang = dataset.lang
    return out


def missing_models(cand: Candidate, cfg: Config) -> list[str]:
    """Catalog ids of the candidate's models that are not installed (empty = everything present)."""
    entries = {entry.id: entry for entry in load_catalog()}
    missing: list[str] = []
    for model_id in (cand.det_model, cand.rec_model):
        if model_id is None:
            continue
        entry = entries.get(model_id)
        if entry is None or model_status(entry, cfg.paths.models_dir, ollama_names=None) != "installed":
            missing.append(model_id)
    return missing


# ---------------------------------------------------------------- running the pairs


def run_qualification(
    pairs: Sequence[tuple[Candidate, Dataset]],
    cfg: Config,
    *,
    run_candidate: RunFn,
    models_installed: ModelsInstalledFn | None = None,
    log: Callable[[str], None] = _log_stderr,
) -> list[Measurement]:
    """Run every pair chapter by chapter, one Measurement per chapter; nothing stops the run.

    A pair whose models are missing (`models_installed` lists them) is skipped with one
    `error="missing model: <id>"` measurement; an exception from `run_candidate` becomes an error
    Measurement and the later pairs still run. `cfg.paths.work_root` must already be the
    qualification work root; each pair runs with `candidate_config` applied on top of it.

    A candidate's model group (detector + OCR engine) is loaded **once** and shared across every
    chapter of every dataset that candidate applies to (`select()` already orders `pairs` by
    candidate, so this is a simple boundary check) — `ocr.lang` is a per-call setting, not part of
    which weights are loaded, so nothing is re-loaded when only the language changes. The manager is
    released when the candidate changes and, via `finally`, after the last one.
    """
    measurements: list[Measurement] = []
    gpu: Any | None = None
    current_candidate: str | None = None
    try:
        for cand, dataset in pairs:
            missing = models_installed(cand) if models_installed is not None else []
            if missing:
                ids = ", ".join(missing)
                log(
                    f"qualify: skip {cand.id}/{dataset.lang}: missing model: {ids}"
                    f" (omniscan models download {' '.join(missing)})"
                )
                measurements.append(
                    Measurement(
                        candidate=cand.id,
                        lang=dataset.lang,
                        series=dataset.series,
                        chapter=dataset.chapters[0],
                        error=f"missing model: {ids}",
                    )
                )
                continue
            pair_cfg = candidate_config(cfg, cand, dataset, cfg.paths.work_root)
            if cand.id != current_candidate:
                if gpu is not None:
                    gpu.release()
                    log(f"qualify: released GPU manager for {current_candidate!r}")
                gpu = build_vram_manager(pair_cfg)
                current_candidate = cand.id
                log(f"qualify: loaded {cand.id} ({cand.engine})")
            for chapter in dataset.chapters:
                try:
                    measurement = run_candidate(pair_cfg, cand, dataset, chapter, gpu=gpu)
                except Exception as exc:  # one failed chapter must not stop the remaining pairs
                    measurement = Measurement(
                        candidate=cand.id,
                        lang=dataset.lang,
                        series=dataset.series,
                        chapter=chapter,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                measurements.append(measurement)
                state = (
                    f"error: {measurement.error}"
                    if measurement.error
                    else f"done in {measurement.seconds:.1f}s"
                )
                log(f"qualify: {cand.id} {dataset.lang} {chapter}: {state}")
    finally:
        if gpu is not None:
            gpu.release()
    return measurements


def default_run_candidate(
    cfg: Config, cand: Candidate, dataset: Dataset, chapter: str, *, gpu: Any | None = None
) -> Measurement:
    """Measure one (candidate, chapter) for real: pipeline to ocr.json, then score against the truth.

    `cfg` comes from `candidate_config`: the series' own settings plus the candidate's OCR overrides
    are already applied, so the pipeline runs with `merge_series_config=False` — a re-merge would
    clobber the candidate with the series' own `[ocr]` section (bug O1e). The vision stages are
    skipped when up to date; the `ocr` stage re-runs because the candidate changes the OCR config
    subset. Every failure — pipeline, scoring, a missing model — becomes one error
    Measurement naming what went wrong (`omniscan models download <id>` for a missing model).
    `gpu`: pass an already-built manager to reuse it across chapters/languages of the same candidate
    (`run_qualification` does this — a candidate's model group only depends on `cand.engine`/
    `det_model`/`rec_model`, never on `dataset.lang`, so nothing needs reloading between them); the
    caller then owns its lifecycle and this function never releases it. `gpu=None` (e.g. calling this
    function directly) builds and releases its own manager exactly as before. The caller holds the
    GPU lock on real hardware either way.
    """
    owns_gpu = gpu is None
    try:
        device = resolve_device(cfg.gpu.device)
        track_vram = device.type == "cuda"
        if track_vram:
            torch.cuda.reset_peak_memory_stats(device)
        if owns_gpu:
            gpu = build_vram_manager(cfg)
        started = time.perf_counter()
        result = run_pipeline(
            cfg,
            dataset.series,
            [chapter],
            stages=["ingest", "slice", "detect", "ocr"],
            gpu=gpu,
            force=False,
            merge_series_config=False,
        )
        seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) / 2**30 if track_vram else None
        if chapter in result.failed:
            raise RuntimeError(f"pipeline failed: {result.failed[chapter]}")
        work = SeriesPaths.from_config(cfg, dataset.series).chapter(chapter).work_dir
        ingest = IngestArtifact.load(work / "ingest.json")
        regions = RegionsArtifact.load(work / "ocr.json")
        check_dir = cfg.paths.library_root.parent / "translated-check" / dataset.series / chapter / "truth"
        truth, stats = load_truth(check_dir, dataset.truth, ingest)
        english = load_english_pages(check_dir, ingest)
        report = score_chapter(dataset.series, chapter, ingest, regions, None, truth, english, stats)
        return Measurement(
            candidate=cand.id,
            lang=dataset.lang,
            series=dataset.series,
            chapter=chapter,
            recall_chars=report.recall_chars,
            chrf=report.ocr_chrf_mean,
            cer_micro=report.cer_micro,
            seconds=seconds,
            peak_vram_gib=peak,
            regions=len(regions.regions),
        )
    except Exception as exc:  # every failure lands in the report, the run goes on
        return Measurement(
            candidate=cand.id,
            lang=dataset.lang,
            series=dataset.series,
            chapter=chapter,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if owns_gpu and gpu is not None:
            gpu.release()  # only when this call built it itself; a shared manager outlives the call


# ---------------------------------------------------------------- the verdict


def summarize(measurements: Sequence[Measurement]) -> dict[tuple[str, str], Summary]:
    """Per (lang, candidate) means over the chapters; errors are excluded from the means, counted."""
    groups: dict[tuple[str, str], list[Measurement]] = {}
    for measurement in measurements:
        groups.setdefault((measurement.lang, measurement.candidate), []).append(measurement)
    summaries: dict[tuple[str, str], Summary] = {}
    for (lang, candidate), group in groups.items():
        ok = [m for m in group if m.error is None]
        summaries[(lang, candidate)] = Summary(
            lang=lang,
            candidate=candidate,
            series=group[0].series,
            chapters=len(group),
            errors=len(group) - len(ok),
            recall_chars=_mean([m.recall_chars for m in ok]),
            chrf=_mean([m.chrf for m in ok]),
            cer_micro=_mean([m.cer_micro for m in ok]),
            seconds=_mean([m.seconds for m in ok]),
            peak_vram_gib=_max([m.peak_vram_gib for m in ok]),
        )
    return summaries


def _mean(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _max(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def recommend(
    summaries: Mapping[tuple[str, str], Summary],
    *,
    lang: str,
    current_default: str,
    min_gain: float = 0.02,
    max_recall_loss: float = 0.02,
) -> Recommendation:
    """KEEP or PROMOTE for one language: PROMOTE needs a chrF gain of `min_gain` at bounded recall loss.

    Best = highest mean chrF among the language's candidates with data (ties: higher char recall,
    then less time). The default itself is never replaced without data: no measurements or an
    all-errors run for it always answer KEEP.
    """
    group = [summary for (summary_lang, _cand), summary in summaries.items() if summary_lang == lang]
    default_sum = next((s for s in group if s.candidate == current_default), None)
    if default_sum is None or default_sum.chrf is None:
        return Recommendation(
            lang=lang,
            verdict="KEEP",
            default=current_default,
            candidate=current_default,
            reason="no data for the default",
        )
    best = min(
        (s for s in group if s.chrf is not None),
        key=lambda s: (-(s.chrf or 0.0), -(s.recall_chars or 0.0), s.seconds or 0.0),
    )
    if best.candidate == current_default:
        return Recommendation(
            lang=lang,
            verdict="KEEP",
            default=current_default,
            candidate=current_default,
            reason="the default is the best candidate",
        )
    gain = (best.chrf or 0.0) - default_sum.chrf
    if gain < min_gain:
        return Recommendation(
            lang=lang,
            verdict="KEEP",
            default=current_default,
            candidate=current_default,
            reason=f"best candidate {best.candidate} gains only {gain:.3f} chrF",
        )
    loss = (default_sum.recall_chars or 0.0) - (best.recall_chars or 0.0)
    if loss > max_recall_loss:
        return Recommendation(
            lang=lang,
            verdict="KEEP",
            default=current_default,
            candidate=current_default,
            reason=f"best candidate {best.candidate} loses {loss:.3f} char recall",
        )
    return Recommendation(
        lang=lang,
        verdict="PROMOTE",
        default=current_default,
        candidate=best.candidate,
        reason=f"gains {gain:.3f} chrF with {loss:.3f} char recall loss",
    )


def _num(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def render_markdown(
    summaries: Mapping[tuple[str, str], Summary],
    recommendations: Sequence[Recommendation],
    *,
    installed: Mapping[str, bool] | None = None,
) -> str:
    """The deterministic markdown report: one section per language, best candidate first."""
    verdicts = {rec.lang: rec for rec in recommendations}
    lines: list[str] = ["# OCR qualification", ""]
    for lang in sorted({summary_lang for summary_lang, _cand in summaries}):
        rows = [s for (summary_lang, _cand), s in summaries.items() if summary_lang == lang]
        rec = verdicts.get(lang)
        lines.append(f"## {lang} ({rows[0].series}, {max(s.chapters for s in rows)} chapter(s))")
        lines.append("")
        lines.append(
            "| candidate | chrF | char recall | CER (box, not comparable) | s/chapter | VRAM GiB | errors |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for s in sorted(rows, key=lambda s: (s.chrf is None, -(s.chrf or 0.0))):
            name = f"{s.candidate} ★" if rec is not None and s.candidate == rec.default else s.candidate
            lines.append(
                f"| {name} | {_num(s.chrf)} | {_num(s.recall_chars)} | {_num(s.cer_micro)} "
                f"| {_num(s.seconds)} | {_num(s.peak_vram_gib)} | {s.errors} |"
            )
        lines.append("")
        if rec is not None:
            lines.append(f"Recommendation: {rec.verdict} {rec.candidate} ({rec.reason})")
            lines.append("")
    if installed is not None:
        not_installed = sorted(cid for cid, ok in installed.items() if not ok)
        if not_installed:
            lines.append(
                f"Skipped, models not installed: {', '.join(not_installed)}"
                " — install with `omniscan models download <model id>`"
            )
            lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"

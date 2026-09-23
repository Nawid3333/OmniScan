"""Run the OCR qualification plan (card O1c) and print the table plus KEEP/PROMOTE verdicts.

Usage (repo root):
  uv run python scripts/qualify_ocr.py [--plan config/qualification.toml] [--lang ko]
      [--only ID,ID] [--chapters N] [--default LANG=ID ...] [--json OUT.json] [--markdown OUT.md]
      [--dry-run] [--download] [--yes]

Every (candidate, dataset) pair runs chapter by chapter through the real pipeline into a dedicated
qualification work root (`<work_root>_qual` — the normal work root is never written). Exit codes:
0 done, 1 when every measurement failed, 2 for bad arguments.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from omniscan.core.config import Config, load_config
from omniscan.eval.qualify import (
    default_run_candidate,
    load_plan,
    missing_models,
    recommend,
    render_markdown,
    run_qualification,
    select,
    summarize,
)
from omniscan.models.catalog import load_catalog

# Today's defaults (mirrors OcrConfig): v5 server det + v5 Korean mobile rec for Korean, the v5
# multilingual pair for the other languages. `--default LANG=ID` overrides per language.
_FALLBACK_DEFAULTS = {"ko": "ppocr-v5-ko"}
_OTHER_DEFAULT = "ppocr-v5-server-multi"


def _echo(text: str) -> None:
    """Print text that may hold non-ASCII (Windows pipes must not crash on it)."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None and (sys.stdout.encoding or "").lower().replace("-", "") != "utf8":
        reconfigure(encoding="utf-8", errors="replace")
    print(text)


def _parse_defaults(entries: Sequence[str]) -> dict[str, str]:
    defaults: dict[str, str] = {}
    for entry in entries:
        lang, sep, candidate = entry.partition("=")
        if not sep or not lang.strip() or not candidate.strip():
            raise ValueError(f"--default expects LANG=ID, got {entry!r}")
        defaults[lang.strip()] = candidate.strip()
    return defaults


def main(argv: Sequence[str] | None = None) -> int:
    """Run the plan and return the process exit code (0 done, 1 all failed, 2 bad arguments)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", type=Path, default=Path("config/qualification.toml"))
    parser.add_argument("--lang", help="only this dataset language (ko, ja, zh, ...)")
    parser.add_argument("--only", default="", help="comma-separated candidate ids to run")
    parser.add_argument("--chapters", type=int, metavar="N", help="only the first N chapters per dataset")
    parser.add_argument(
        "--default",
        action="append",
        default=[],
        metavar="LANG=ID",
        help="today's default per language (repeatable)",
    )
    parser.add_argument("--json", type=Path, help="write the measurements as JSON")
    parser.add_argument("--markdown", type=Path, help="write the markdown report")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the pairs and missing models, run nothing"
    )
    parser.add_argument("--download", action="store_true", help="download missing candidate models first")
    parser.add_argument("--yes", action="store_true", help="with --download: skip the confirmation prompt")
    args = parser.parse_args(argv)

    try:
        defaults = _parse_defaults(args.default)
        if args.chapters is not None and args.chapters < 1:
            raise ValueError(f"--chapters expects N >= 1, got {args.chapters}")
    except ValueError as exc:
        print(f"qualify: {exc}", file=sys.stderr)
        return 2

    try:
        candidates, datasets = load_plan(args.plan)
    except (OSError, ValueError) as exc:
        print(f"qualify: {exc}", file=sys.stderr)
        return 2

    known = {cand.id for cand in candidates}
    unknown = sorted({cid for cid in defaults.values() if cid not in known})
    if unknown:
        print(f"qualify: unknown --default candidate id(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    datasets = [
        dataclasses.replace(dataset, chapters=dataset.chapters[: args.chapters]) for dataset in datasets
    ]
    only = [item.strip() for item in args.only.split(",") if item.strip()]
    pairs = select(candidates, datasets, lang=args.lang, only=only)
    if not pairs:
        print("qualify: no (candidate, dataset) pairs selected", file=sys.stderr)
        return 2

    cfg = load_config().model_copy(deep=True)
    cfg.paths.work_root = Path(f"{cfg.paths.work_root}_qual")  # never write into the normal work root
    pair_candidates = list(dict.fromkeys(cand for cand, _dataset in pairs))
    missing = {cand.id: missing_models(cand, cfg) for cand in pair_candidates}

    if args.dry_run:
        _echo(f"dry run: {len(pairs)} pair(s) into work root {cfg.paths.work_root}")
        for cand, dataset in pairs:
            _echo(f"  {cand.id} {dataset.lang} {dataset.series}: {', '.join(dataset.chapters)}")
        for cand_id, model_ids in missing.items():
            if model_ids:
                _echo(f"  missing models for {cand_id}: {', '.join(model_ids)} (omniscan models download)")
        return 0

    if args.download and any(missing.values()):
        if not _confirm_download(missing, ask=not args.yes):
            print("qualify: download declined — affected candidates will be skipped", file=sys.stderr)
        else:
            _download_missing(missing, cfg)
            missing = {cand.id: missing_models(cand, cfg) for cand in pair_candidates}

    hw_lock = None
    if cfg.gpu.device != "cpu":
        from omniscan.gpu.lock import acquire_gpu_lock, release_gpu_lock

        hw_lock = acquire_gpu_lock(
            on_wait=lambda: print("qualify: waiting for exclusive GPU access...", file=sys.stderr)
        )
    try:
        measurements = run_qualification(
            pairs,
            cfg,
            run_candidate=default_run_candidate,
            models_installed=lambda cand: missing[cand.id],
        )
    finally:
        if hw_lock is not None:
            release_gpu_lock(hw_lock)

    summaries = summarize(measurements)
    langs = list(dict.fromkeys(dataset.lang for _cand, dataset in pairs))
    recommendations = [
        recommend(
            summaries,
            lang=lang,
            current_default=defaults.get(lang, _FALLBACK_DEFAULTS.get(lang, _OTHER_DEFAULT)),
        )
        for lang in langs
    ]
    installed = {cand.id: not missing[cand.id] for cand in pair_candidates}
    markdown = render_markdown(summaries, recommendations, installed=installed)
    _echo(markdown)

    if args.json is not None:
        args.json.write_text(
            json.dumps([dataclasses.asdict(m) for m in measurements], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"qualify: wrote {args.json}", file=sys.stderr)
    if args.markdown is not None:
        args.markdown.write_text(markdown, encoding="utf-8")
        print(f"qualify: wrote {args.markdown}", file=sys.stderr)

    if measurements and all(m.error is not None for m in measurements):
        print("qualify: every measurement failed", file=sys.stderr)
        return 1
    return 0


def _confirm_download(missing: dict[str, list[str]], *, ask: bool) -> bool:
    """One confirmation prompt naming the missing model ids and sizes; False when declined."""
    entries = {entry.id: entry for entry in load_catalog()}
    ids = list(dict.fromkeys(model_id for model_ids in missing.values() for model_id in model_ids))
    listed = ", ".join(
        f"{model_id} ({entries[model_id].size_mb} MB)" for model_id in ids if model_id in entries
    )
    print(f"qualify: {len(ids)} model(s) to download: {listed}")
    if not ask:
        return True
    try:
        answer = input("Download now? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def _download_missing(missing: dict[str, list[str]], cfg: Config) -> None:
    """Download every missing model id once, through models.download.download_model."""
    from omniscan.models.download import ModelDownloadError, download_model

    entries = {entry.id: entry for entry in load_catalog()}
    for model_id in dict.fromkeys(model_id for model_ids in missing.values() for model_id in model_ids):
        entry = entries.get(model_id)
        if entry is None:
            print(f"qualify: {model_id} is not in the catalog", file=sys.stderr)
            continue
        try:
            source = download_model(entry, cfg.paths.models_dir)
        except ModelDownloadError as exc:
            print(f"qualify: {exc}", file=sys.stderr)
            continue
        print(f"qualify: {model_id}: installed from {source}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

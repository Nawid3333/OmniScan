import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # see the README: stale bytecode can poison verdicts

P = "src/omniscan/qa/progress.py"
CLI = "src/omniscan/cli.py"
WEB = "src/omniscan/web/app.py"
MUTANTS = [
    (
        P,
        "if r.kind in LINE_KINDS and r.text.strip()]",
        "if r.text.strip()]",
        "sound effects and watermarks count as lines",
    ),
    (
        P,
        "if r.kind in LINE_KINDS and r.text.strip()]",
        "if r.kind in LINE_KINDS]",
        "empty regions count as lines",
    ),
    (
        P,
        'english.get(region_id, "").strip())',
        'english.get(region_id, ""))',
        "a blank English line counts as translated",
    ),
    (
        P,
        'if status.get(region_id) == "checked")',
        'if status.get(region_id) != "todo")',
        "edited lines count as checked",
    ),
    (P, 'if problem.status != "checked")', ")", "problems on checked lines count"),
    (
        P,
        "last_stage=done[-1] if done else None,",
        "last_stage=done[0] if done else None,",
        "the first stage is reported",
    ),
    (
        P,
        'stages[name].status == "failed"), None)',
        'stages[name].status == "done"), None)',
        "a done stage reads as failed",
    ),
    (
        P,
        '    exported = export is not None and export.status == "done"\n',
        "    exported = export is not None\n",
        "a failed export counts as exported",
    ),
    (
        P,
        "        rerun = any(stages[name].finished_at > export.finished_at for name in STAGE_ORDER if name in stages)",
        "        rerun = any(record.finished_at > export.finished_at for record in stages.values())",
        "the qa re-read makes the export outdated",
    ),
    (
        P,
        "        rerun = any(stages[name].finished_at > export.finished_at for name in STAGE_ORDER if name in stages)",
        "        rerun = False",
        "a re-run stage leaves the export current",
    ),
    (P, "(store.EDITS_FILE, CLEANUP_FILE)", "(store.EDITS_FILE,)", "a hand clean leaves the export current"),
    (
        P,
        "        outdated = rerun or any(when > export.finished_at for when in edited)",
        "        outdated = rerun",
        "a hand edit leaves the export current",
    ),
    (
        CLI,
        '        return f"failed: {progress.failed}"',
        '        return "failed"',
        "the failed stage is not named",
    ),
    (
        CLI,
        'return f"through {progress.last_stage}" if progress.last_stage else "not started"',
        'return f"through {progress.last_stage}"',
        "a chapter that never ran reads 'through None'",
    ),
    (
        CLI,
        '        export = ("outdated" if progress.outdated else "yes") if progress.exported else "no"',
        '        export = "yes" if progress.exported else "no"',
        "an outdated export reads as current",
    ),
    (
        CLI,
        '            f"{progress.checked}/{progress.lines}",',
        '            f"{progress.translated}/{progress.lines}",',
        "the checked column shows the translated count",
    ),
    (
        WEB,
        "        return [asdict(progress) for progress in series_progress(series_paths(series))]",
        "        return []",
        "the web answers nothing",
    ),
]

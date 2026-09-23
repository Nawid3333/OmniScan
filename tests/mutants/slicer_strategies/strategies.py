"""Hand-written mutants for the slicer strategies and compare mode (card Q6).

Tests used for the verdicts: `tests/unit/test_slicer_strategies.py`,
`tests/unit/test_slicer_compare.py`, `tests/unit/test_slicer_compare_cli.py`, plus the gap tests
they spawned in `tests/unit/test_slicer_strategies_mutation_gaps.py` (and the appended int-median
test in `test_slicer_compare.py`).
"""

MUTANTS = [
    # ---- dispatcher (slice_with_strategy) -----------------------------------
    (
        "src/omniscan/slicer/strategies.py",
        "name = cfg.strategy if strategy is None else strategy",
        "name = cfg.strategy if strategy is None else cfg.strategy",
        "explicit strategy ignored",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "if name not in STRATEGIES:",
        "if name in STRATEGIES:",
        "invert unknown-name check",
    ),
    # ---- slice_page ----------------------------------------------------------
    (
        "src/omniscan/slicer/strategies.py",
        "if not source_files and height > 0:",
        "if not source_files:",
        "drop empty-strip page guard",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "if not source_files and height > 0:",
        "if not source_files or height > 0:",
        "page fallback and -> or",
    ),
    # ---- slice_fixed ---------------------------------------------------------
    (
        "src/omniscan/slicer/strategies.py",
        "while height - pos > cfg.target_height:",
        "while height - pos >= cfg.target_height:",
        "fixed loop > -> >=",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        """if height > pos:
        boundaries.append((pos, height, False))
    if len(boundaries) > 1""",
        """if height >= pos:
        boundaries.append((pos, height, False))
    if len(boundaries) > 1""",
        "fixed tail guard > -> >=",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "height - pos < cfg.min_height",
        "height - pos <= cfg.min_height",
        "merge threshold < -> <=",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "y0, _, forced = boundaries[-2]",
        "y0, _, forced = boundaries[-1]",
        "merge reads tail slice",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        """boundaries.append((pos, height, False))
    if len(boundaries) > 1""",
        """boundaries.append((pos, height, True))
    if len(boundaries) > 1""",
        "fixed tail forced flag",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "(y0, height, forced)",
        "(y0, height, False)",
        "merged slice drops forced flag",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "boundaries.append((pos, pos + cfg.target_height, True))",
        "boundaries.append((pos, pos + cfg.target_height, False))",
        "fixed cut forced -> False",
    ),
    # ---- gutter_cut_rows -----------------------------------------------------
    (
        "src/omniscan/slicer/strategies.py",
        "luma.var(dim=1, unbiased=False) < cfg.gutter_variance",
        "luma.var(dim=1, unbiased=False) <= cfg.gutter_variance",
        "variance < -> <=",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "[(s + e) // 2 for s, e in zip",
        "[(s + e + 1) // 2 for s, e in zip",
        "run centre floor -> ceil",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "if e - s >= cfg.gutter_min_rows]",
        "if e - s > cfg.gutter_min_rows]",
        "run length >= -> >",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "e - s >= cfg.gutter_min_rows]",
        "e - s >= cfg.gutter_min_rows - 1]",
        "min_rows threshold - 1",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        """padded = torch.cat(
        (
            torch.zeros(1, dtype=torch.bool, device=strip.device),
            gutter,""",
        """padded = torch.cat(
        (
            gutter,""",
        "drop leading zero pad",
    ),
    # ---- slice_simple_gutter -------------------------------------------------
    (
        "src/omniscan/slicer/strategies.py",
        "while height - pos > cfg.max_height:",
        "while height - pos >= cfg.max_height:",
        "gutter loop > -> >=",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "candidates = [c for c in cut_rows if lo <= c <= hi]",
        "candidates = [c for c in cut_rows if lo <= c < hi]",
        "window top <= -> <",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "cut = min(candidates, key=lambda c: (abs(c - target), c))",
        "cut = max(candidates, key=lambda c: (abs(c - target), c))",
        "pick min -> max",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "key=lambda c: (abs(c - target), c)",
        "key=lambda c: (abs(c - target), -c)",
        "tie prefers larger row",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "abs(c - target)",
        "abs(c - pos)",
        "measure from window start",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "boundaries.append((pos, target, True))",
        "boundaries.append((pos, target, False))",
        "forced fallback drops flag",
    ),
    # ---- _finalize -----------------------------------------------------------
    (
        "src/omniscan/slicer/strategies.py",
        "blank=bool(stats.uniform[y0:y1].all().item())",
        "blank=bool(stats.uniform[y0:y1].any().item())",
        "blank all -> any",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "if f.y0 < y1 and f.y1 > y0]",
        "if f.y0 < y1 or f.y1 > y0]",
        "file overlap and -> or",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        "if f.y0 < y1 and f.y1 > y0]",
        "if f.y0 <= y1 and f.y1 > y0]",
        "file start < -> <=",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        """strip_width=strip.shape[2],
        strip_height=strip.shape[1],""",
        """strip_width=strip.shape[1],
        strip_height=strip.shape[2],""",
        "strip dims swapped",
    ),
    (
        "src/omniscan/slicer/strategies.py",
        '{**cfg.model_dump(), "strategy": name}',
        "{**cfg.model_dump()}",
        "params drop strategy",
    ),
    # ---- compare.py ----------------------------------------------------------
    (
        "src/omniscan/slicer/compare.py",
        "min_height=min(heights) if heights else 0",
        "min_height=max(heights) if heights else 0",
        "summary min -> max",
    ),
    (
        "src/omniscan/slicer/compare.py",
        "max_height=max(heights) if heights else 0",
        "max_height=min(heights) if heights else 0",
        "summary max -> min",
    ),
    (
        "src/omniscan/slicer/compare.py",
        "median_height=int(statistics.median(heights)) if heights else 0",
        "median_height=int(statistics.mean(heights)) if heights else 0",
        "summary median -> mean",
    ),
    (
        "src/omniscan/slicer/compare.py",
        "median_height=int(statistics.median(heights)) if heights else 0",
        "median_height=statistics.median(heights) if heights else 0",
        "median height loses int()",
    ),
    (
        "src/omniscan/slicer/compare.py",
        "forced=sum(1 for s in artifact.slices if s.forced_cut)",
        "forced=sum(1 for s in artifact.slices if s.blank)",
        "forced counts blank",
    ),
    (
        "src/omniscan/slicer/compare.py",
        "cuts=tuple(s.y1 for s in artifact.slices[:-1])",
        "cuts=tuple(s.y1 for s in artifact.slices)",
        "cuts include last slice",
    ),
]
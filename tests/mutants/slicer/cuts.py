"""Mutants for src/omniscan/slicer/cuts.py (tests: test_slicer_cuts, test_slicer_properties, gaps)."""

C = "src/omniscan/slicer/cuts.py"

MUTANTS = [
    # ---- _cost
    (C, "if length < min_height:", "if length <= min_height:", "min_height < -> <="),
    (C, "c += 4.0", "c += 40.0", "min-height penalty 4 -> 40"),
    (C, "if length > max_height:", "if length >= max_height:", "max_height > -> >="),
    (C, "c += 100.0", "c += 400.0", "max-height penalty 100 -> 400"),
    # ---- _least_detail
    (C, "lo = max(t - 256, a + 1)", "lo = max(t - 128, a + 1)", "detail window -256 -> -128"),
    (C, "hi = min(t + 256, b - 1)", "hi = min(t + 128, b - 1)", "detail window +256 -> +128"),
    (C, "lo = max(t - 256, a + 1)", "lo = max(t - 512, a + 1)", "detail window -256 -> -512"),
    (C, "m = int(window.min().item())", "m = int(window.max().item())", "least detail -> most detail"),
    (C, "dist = (idx + lo - t).abs()", "dist = (idx + lo).abs()", "tie distance drops -t"),
    (
        C,
        "return lo + int(idx[dist.argmin()].item())",
        "return lo + int(idx[dist.argmax()].item())",
        "tie argmin -> argmax",
    ),
    # ---- _dp_cuts
    (C, "if seg > hard_max:", "if seg < hard_max:", "hard_max > -> <"),
    (C, "for j in range(1, m + 2):", "for j in range(1, m + 1):", "dp loop skips final node"),
    (C, "seg = pos[j] - pos[i]", "seg = pos[j] - pos[i] + 1", "segment length +1"),
    (
        C,
        "_dp_cuts(a, b, cands, cfg.target_height, cfg.min_height, cfg.max_height, cfg.hard_max_height)",
        "_dp_cuts(a, b, cands, cfg.max_height, cfg.min_height, cfg.target_height, cfg.hard_max_height)",
        "swap target_height and max_height args",
    ),
    (
        C,
        "_dp_cuts(a, b, cands, cfg.target_height, cfg.min_height, cfg.max_height, cfg.hard_max_height)",
        "_dp_cuts(a, b, cands, cfg.target_height, cfg.max_height, cfg.min_height, cfg.hard_max_height)",
        "swap min_height and max_height args",
    ),
    (C, "best = dp[m + 1]", "best = dp[m]", "dp result at wrong node"),
    (C, "= [None] * (m + 2)", "= [None] * (m + 1)", "dp table one short"),
    # ---- plan_cuts: candidates
    (
        C,
        "if length >= cfg.min_height + min_band:",
        "if length > cfg.min_height + min_band:",
        "edge/centre threshold >= -> >",
    ),
    (C, "edges.add(band.y0 + min_band // 2)", "edges.add(band.y0 + min_band // 2 + 1)", "edge y0 +1"),
    (C, "edges.add(band.y1 - min_band // 2)", "edges.add(band.y1 - min_band // 2 - 1)", "edge y1 -1"),
    (C, "centres.add((band.y0 + band.y1) // 2)", "centres.add((band.y0 + band.y1 + 1) // 2)", "centre +1"),
    # ---- plan_cuts: forced cuts
    (C, "if gap > cfg.hard_max_height:", "if gap >= cfg.hard_max_height:", "hard_max gap > -> >="),
    (
        C,
        "k = math.ceil(gap / cfg.max_height) - 1",
        "k = math.floor(gap / cfg.max_height) - 1",
        "ceil -> floor",
    ),
    (C, "k = math.ceil(gap / cfg.max_height) - 1", "k = math.ceil(gap / cfg.max_height)", "drop -1"),
    (C, "t = a + round(i * gap / (k + 1))", "t = a + int(i * gap / (k + 1))", "round -> int (floor)"),
    (
        C,
        "mandatory = sorted({0, height} | edges | set(forced))",
        "mandatory = sorted({0, height} | edges)",
        "drop forced from mandatory",
    ),
    (
        C,
        "positions = [0, *sorted(centres | edges), height]",
        "positions = [*sorted(centres | edges), height]",
        "drop 0 from positions",
    ),
    # ---- plan_cuts: flags / output
    (
        C,
        "for y in edges:\n        cuts[y] = False",
        "for y in edges:\n        cuts[y] = True",
        "edge cuts marked forced",
    ),
    (
        C,
        "for y in forced:\n        cuts[y] = True",
        "for y in forced:\n        cuts[y] = False",
        "forced cuts marked normal",
    ),
    (C, "cuts.setdefault(y, False)", "cuts.setdefault(y, True)", "dp cuts marked forced"),
    (
        C,
        "return [Cut(y=y, forced=cuts[y]) for y in sorted(cuts)]",
        "return [Cut(y=y, forced=cuts[y]) for y in sorted(cuts, reverse=True)]",
        "cuts sorted descending",
    ),
]

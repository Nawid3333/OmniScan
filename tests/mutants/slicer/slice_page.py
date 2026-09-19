"""Mutants for src/omniscan/slicer/slice.py and page_mode.py (tests: slice, page_mode, properties, gaps)."""

SL = "src/omniscan/slicer/slice.py"
PM = "src/omniscan/slicer/page_mode.py"

MUTANTS = [
    # ---- slice.py
    (
        SL,
        "height, width = strip.shape[1], strip.shape[2]",
        "height, width = strip.shape[2], strip.shape[1]",
        "swap width and height",
    ),
    (
        SL,
        "stats = row_stats(strip, cfg.uniform_tol)",
        "stats = row_stats(strip, cfg.band_min_px)",
        "row_stats tol -> band_min_px",
    ),
    (
        SL,
        "find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)",
        "find_uniform_bands(stats, cfg.band_min_px + 1, cfg.uniform_tol, cfg.max_drift)",
        "band_min_px +1",
    ),
    (
        SL,
        "find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)",
        "find_uniform_bands(stats, cfg.max_drift, cfg.uniform_tol, cfg.band_min_px)",
        "swap min_band and max_drift args",
    ),
    (
        SL,
        "cuts = plan_cuts(height, bands, stats.detail, cfg)",
        "cuts = plan_cuts(width, bands, stats.detail, cfg)",
        "plan_cuts with width",
    ),
    (
        SL,
        "boundaries = [0, *(c.y for c in cuts), height]",
        "boundaries = [0, *(c.y for c in cuts), height - 1]",
        "last boundary -1",
    ),
    (
        SL,
        "forced = cuts[i].forced if i < len(cuts) else False",
        "forced = cuts[i].forced if i <= len(cuts) else False",
        "guard < -> <=",
    ),
    (
        SL,
        "blank = bool(stats.uniform[y0:y1].all().item())",
        "blank = bool(stats.uniform[y0:y1].any().item())",
        "blank all -> any",
    ),
    (SL, "if f.y0 < y1 and f.y1 > y0]", "if f.y0 <= y1 and f.y1 > y0]", "file overlap < -> <="),
    (SL, "if f.y0 < y1 and f.y1 > y0]", "if f.y0 < y1 or f.y1 > y0]", "file overlap and -> or"),
    (
        SL,
        "slices.append(Slice(index=i, y0=y0, y1=y1, blank=blank, forced_cut=forced, source_files=files))",
        "slices.append(Slice(index=i + 1, y0=y0, y1=y1, blank=blank, forced_cut=forced, source_files=files))",
        "slice index +1",
    ),
    (
        SL,
        "slices.append(Slice(index=i, y0=y0, y1=y1, blank=blank, forced_cut=forced, source_files=files))",
        "slices.append(Slice(index=i, y0=y1, y1=y0, blank=blank, forced_cut=forced, source_files=files))",
        "swap slice y0 y1",
    ),
    (
        SL,
        "slices.append(Slice(index=i, y0=y0, y1=y1, blank=blank, forced_cut=forced, source_files=files))",
        "slices.append(Slice(index=i, y0=y0, y1=y1, blank=not blank, forced_cut=forced, source_files=files))",
        "negate blank",
    ),
    (SL, "bands=bands,", "bands=[],", "artifact bands dropped"),
    (SL, "params=cfg.model_dump(),", "params={},", "artifact params dropped"),
    (SL, "for i in range(len(boundaries) - 1):", "for i in range(len(boundaries)):", "slice loop one extra"),
    # ---- page_mode.py
    (PM, "if not source_files:", "if source_files:", "empty check inverted"),
    (PM, "if strip_height != 0:", "if strip_height == 0:", "strip_height != -> =="),
    (PM, "if b.y0 < a.y0:", "if b.y0 > a.y0:", "order check inverted"),
    (PM, "if b.y0 != a.y1:", "if b.y0 > a.y1:", "tiling != -> >"),
    (
        PM,
        'kind = "overlap" if b.y0 < a.y1 else "gap"',
        'kind = "gap" if b.y0 < a.y1 else "overlap"',
        "kind labels swapped",
    ),
    (PM, "abs(b.y0 - a.y1)", "(b.y0 - a.y1)", "drop abs in message"),
    (PM, "if first.y0 != 0:", "if first.y0 > 0:", "first page check != -> >"),
    (PM, "if last.y1 != strip_height:", "if last.y1 < strip_height:", "last page check != -> <"),
    (
        PM,
        "for i, f in enumerate(source_files)]",
        "for i, f in enumerate(source_files, 1)]",
        "enumerate from 1",
    ),
    (PM, "source_files=[f.index])", "source_files=[i])", "source_files uses slice index"),
    (
        PM,
        'slices=slices,\n        params={"mode": "page"},',
        'slices=slices,\n        params={"mode": "strip"},',
        "page mode label",
    ),
    (
        PM,
        "strip_width=strip_width,\n            strip_height=0,",
        "strip_width=strip_width,\n            strip_height=strip_height,",
        "empty case height",
    ),
]

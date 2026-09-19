"""Mutants for src/omniscan/slicer/bands.py (tests: test_slicer_bands, test_slicer_properties)."""

MUTANTS = [
    # ---- row_stats
    (
        "src/omniscan/slicer/bands.py",
        "end = min(start + chunk_rows, height)",
        "end = max(start + chunk_rows, height)",
        "chunk bound min->max",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "for start in range(0, height, chunk_rows):",
        "for start in range(1, height, chunk_rows):",
        "chunk loop starts at 1",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "lo = (med_i16 - tol).clamp(0, 255).to(torch.uint8)",
        "lo = (med_i16 - tol - 1).clamp(0, 255).to(torch.uint8)",
        "tol lower bound off by 1",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "hi = (med_i16 + tol).clamp(0, 255).to(torch.uint8)",
        "hi = (med_i16 + tol + 1).clamp(0, 255).to(torch.uint8)",
        "tol upper bound off by 1",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "det = diff.abs().amax(dim=(0, 2))",
        "det = diff.abs().amin(dim=(0, 2))",
        "detail max -> min",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "detail[start:end] = det.to(torch.uint8)",
        "detail[start:end] = det.to(torch.uint8) + 1",
        "detail + 1",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "unif = ((chunk >= lo.unsqueeze(2)) & (chunk <= hi.unsqueeze(2))).all(dim=0).all(dim=1)",
        "unif = ((chunk > lo.unsqueeze(2)) & (chunk <= hi.unsqueeze(2))).all(dim=0).all(dim=1)",
        "lower >= -> >",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "unif = ((chunk >= lo.unsqueeze(2)) & (chunk <= hi.unsqueeze(2))).all(dim=0).all(dim=1)",
        "unif = ((chunk >= lo.unsqueeze(2)) & (chunk < hi.unsqueeze(2))).all(dim=0).all(dim=1)",
        "upper <= -> <",
    ),
    (
        "src/omniscan/slicer/bands.py",
        ".all(dim=0).all(dim=1)",
        ".all(dim=0).any(dim=1)",
        "row uniform any -> all",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "uniform[start:end] = unif",
        "uniform[start:end] = ~unif",
        "invert uniform mask",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "chunk = strip[:, start:end, :]",
        "chunk = strip[:, start:end, : strip.shape[2] // 2]",
        "median over half width",
    ),
    # ---- find_uniform_bands: linking / starts / ends
    (
        "src/omniscan/slicer/bands.py",
        "diff = (med[:, 1:] - med[:, :-1]).abs().mean(dim=0)",
        "diff = (med[:, 1:] - med[:, :-1]).abs().amax(dim=0)",
        "drift mean -> max",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "link = uniform[1:] & uniform[:-1] & (diff <= max_drift)",
        "link = uniform[1:] & uniform[:-1] & (diff < max_drift)",
        "drift <= -> <",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "link = uniform[1:] & uniform[:-1] & (diff <= max_drift)",
        "link = uniform[1:] & uniform[:-1]",
        "drop drift test from link",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "link = uniform[1:] & uniform[:-1] & (diff <= max_drift)",
        "link = uniform[:-1] & (diff <= max_drift)",
        "drop uniform[1:] from link",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "link = uniform[1:] & uniform[:-1] & (diff <= max_drift)",
        "link = uniform[1:] & (diff <= max_drift)",
        "drop uniform[:-1] from link",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "link = uniform[1:] & uniform[:-1] & (diff <= max_drift)",
        "link = uniform[1:] | (uniform[:-1] & (diff <= max_drift))",
        "link and -> or",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "start = uniform.clone()",
        "start = torch.zeros_like(uniform)",
        "start from zeros not clone",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "start[1:] = uniform[1:] & ~link",
        "start[1:] = uniform[1:] & link",
        "start uses link not ~link",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "end[1:] = uniform[:-1] & ~link",
        "end[1:] = uniform[1:] & ~link",
        "end indexed uniform[1:]",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "if bool(uniform[-1].item()):",
        "if bool(uniform[0].item()):",
        "tail check uses row 0",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "end_idx.append(height)",
        "end_idx.append(height - 1)",
        "last band y1 - 1",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "end_idx.append(height)",
        "end_idx.append(height + 1)",
        "last band y1 + 1",
    ),
    # ---- band extraction loop
    ("src/omniscan/slicer/bands.py", "if y1 - y0 < min_band:", "if y1 - y0 <= min_band:", "min_band < -> <="),
    (
        "src/omniscan/slicer/bands.py",
        "band_med = stats.median[:, y0:y1]",
        "band_med = stats.median[:, y0 : y1 - 1]",
        "band median skips last row",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "color = (int(r), int(g), int(b))",
        "color = (int(b), int(g), int(r))",
        "swap r and b channels",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "is_gradient = bool((med[:, y1 - 1] - med[:, y0]).abs().max().item() > tol)",
        "is_gradient = bool((med[:, y1 - 1] - med[:, y0]).abs().max().item() >= tol)",
        "gradient > -> >=",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "is_gradient = bool((med[:, y1 - 1] - med[:, y0]).abs().max().item() > tol)",
        "is_gradient = bool((med[:, y1 - 1] - med[:, y0]).abs().max().item() < tol)",
        "gradient > -> <",
    ),
    (
        "src/omniscan/slicer/bands.py",
        "bands.sort(key=lambda b: b.y0)",
        "bands.sort(key=lambda b: -b.y0)",
        "sort descending",
    ),
]

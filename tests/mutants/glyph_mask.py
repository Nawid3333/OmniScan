"""Mutants for the glyph-precise inpaint masks (inpaint/glyph_mask.py, inpaint/pipeline.py).

Run: python scripts/mutate.py run tests/mutants/glyph_mask.py -t tests/unit/test_inpaint_glyph_mask.py \
     -t tests/unit/test_ocr_sfx.py
"""

G = "src/omniscan/inpaint/glyph_mask.py"
P = "src/omniscan/inpaint/pipeline.py"

MUTANTS = [
    (
        G,
        "    body = fill_holes(ink)\n    return ink if",
        "    body = ink\n    return ink if",
        "no hole filling",
    ),
    (
        G,
        "            return low_share_inside > low_share_border\n",
        "            return low_share_inside < 0.5\n",
        "minority rule only",
    ),
    (
        G,
        "    grow_px = min(limit, max(min_grow_px, rim + _AA_PX))",
        "    grow_px = min(limit, max(min_grow_px, _AA_PX))",
        "no outline growth",
    ),
    (
        G,
        '    projected = torch.einsum("c,chw->hw", axis, pixels)',
        "    projected = luminance(crop)",
        "luminance split only",
    ),
    (
        G,
        "    if float(projected[inside & ~low].mean()) - float(projected[low_inside].mean()) < min_contrast:",
        "    if False:",
        "no contrast guard",
    ),
    (
        G,
        "    return ink if int((body & inside).sum()) / n_inside > max_ink else body",
        "    return body",
        "keep a frame's filled holes",
    ),
    (
        G,
        "    return ink & near if 2 * int((ink & near).sum()) >= int(ink.sum()) else ink",
        "    return ink",
        "art of another colour kept with the letters",
    ),
    (
        G,
        "    rim = outline_width(crop, _filled(letters, inside, n_inside, max_ink), limit)",
        "    rim = outline_width(crop, body, limit)",
        "outline measured around same-coloured art too",
    ),
    (
        P,
        "                    ring_mask=local_ring(mask, cfg.glyph_ring_px),\n",
        "",
        "glyph flat fill samples the whole crop",
    ),
    (
        P,
        '        if region.kind == "sfx" and sfx_mode != "replace":\n            continue\n',
        "",
        "sfx always erased",
    ),
]

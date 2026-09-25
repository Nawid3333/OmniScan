"""Mutants for the glyph-precise inpaint masks (inpaint/glyph_mask.py, inpaint/pipeline.py).

Run: python scripts/mutate.py run tests/mutants/glyph_mask.py -t tests/unit/test_inpaint_glyph_mask.py
"""

G = "src/omniscan/inpaint/glyph_mask.py"
P = "src/omniscan/inpaint/pipeline.py"

MUTANTS = [
    (G, "    body = fill_holes(ink)\n", "    body = ink\n", "no hole filling"),
    (
        G,
        "            return low_share_inside > low_share_border\n",
        "            return low_share_inside < 0.5\n",
        "minority rule only",
    ),
    (
        G,
        "    grow_px = min(limit, max(min_grow_px, outline_width(crop, body, limit) + _AA_PX))",
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
        "        body = ink  # the ink encloses most",
        "        pass  # the ink encloses most",
        "keep a frame's filled holes",
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

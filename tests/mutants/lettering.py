"""Mutants for the lettering engine (typeset/fit.py, plan.py, sfx.py) and the SFX style measurement (ocr/sfx.py).

Run: python scripts/mutate.py run tests/mutants/lettering.py -t tests/unit/test_typeset_fit.py \
     -t tests/unit/test_typeset_plan.py -t tests/unit/test_ocr_sfx.py
"""

F = "src/omniscan/typeset/fit.py"
P = "src/omniscan/typeset/plan.py"
T = "src/omniscan/typeset/sfx.py"
S = "src/omniscan/ocr/sfx.py"

MUTANTS = [
    (F, '        if self.kind == "rect":', "        if True:", "every shape treated as a rectangle"),
    (
        F,
        "                        value += unit * self.break_cost(end)",
        "                        value += 0.0",
        "no phrasing when balancing",
    ),
    (
        F,
        "                    cost += _LONELY_PENALTY",
        "                    cost += 0.0",
        "stranded little words allowed",
    ),
    (
        F,
        "            score = size - _PHRASING_PX * setter.phrasing(ends)",
        "            score = size",
        "line count by size alone",
    ),
    (F, "            tokens = _hyphenated(", "            tokens = list(", "no hyphenation"),
    (
        P,
        "        cap = _size_cap(job.role, typical, cfg)",
        "        cap = None",
        "no chapter-wide size limit",
    ),
    (
        P,
        "        stroke_color=outline_for(color, region.stroke_color),",
        "        stroke_color=region.stroke_color or (0, 0, 0),",
        "no outline contrast guard",
    ),
    (T, "        angle=angle,", "        angle=0.0,", "sfx lettered level"),
    (
        S,
        "    upright = levelled(letters, thinnest_tilt(letters))",
        "    upright = letters",
        "weight measured on tilted strokes",
    ),
    (
        S,
        "outline is None or _far(inner, outline, _FILL_CONTRAST)",
        "True",
        "an outlined counter read as a fill",
    ),
]

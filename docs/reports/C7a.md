# C7a — Typeset layout engine

## Changes

New files only (nothing outside the card's list was touched):

- `src/omniscan/typeset/__init__.py` — empty package marker.
- `src/omniscan/typeset/fonts.py` — `DEFAULT_FONT_FILES` (the six-role table from the card),
  `fonts_dir()` (`OMNISCAN_FONTS_DIR` env override, else `<repo root>/fonts` development default),
  `default_font_path(role)` (raises the documented `FileNotFoundError` when the file is missing),
  `load_font(path, size_px)` (`ImageFont.truetype` behind `functools.lru_cache(maxsize=64)`;
  `size_px < 1` → `ValueError`, missing file → the same `FileNotFoundError` text).
- `src/omniscan/typeset/fit.py` — `Measurable` protocol, `FontFactory` alias, `wrap_words`
  (greedy, single-too-wide-word → own line + `too_wide` flag), `line_height`
  (`round(size*spacing)` clamped to ≥ 1), frozen `Fit` dataclass, `fit_text` (size search from
  `max_px` down; first size with `not too_wide and height <= max_h` wins; min size with
  `overflow=True` otherwise), `inscribed_box` (margin-shrink when `polygon is None`; otherwise the
  1.00 → 0.20 scale search accepting the first rectangle whose 24 boundary samples — 4 corners +
  5 interior points per side — are inside the polygon under even-odd with on-outline = outside,
  then margin-shrink, `round`, and a clamp that degenerates a too-tiny bubble to a zero-size box
  at its centre), and `layout_region` (fit + centred `BBox` + `LayoutItem`, not clamped on overflow).
- `tests/unit/test_typeset_fonts.py`, `tests/unit/test_typeset_fit.py` — all 16 acceptance
  behaviours, including the real-font property tests (2 fonts × 6 boxes × 8 sentences) with the
  optimality re-wrap check, the monotonicity check, and a test-local even-odd point-in-polygon
  used to verify the inscribed box on rectangle and concave-L polygons.
- `docs/reports/C7a.md` — this report.

## Tests

```
uv run pytest tests/unit/test_typeset_fonts.py tests/unit/test_typeset_fit.py -q
  → 221 passed
uv run pytest
  → 2195 passed, 3 warnings (pre-existing warnings in test_codec_turbo)
uv run ruff format . && uv run ruff check .
  → all clean
uv run pyright
  → 0 errors, 0 warnings, 0 informations
```

Notes on a few deterministic points the card calls out, verified by the tests:
- `line_height(30, 1.15) == 34`: `30 * 1.15` is exactly the double nearest 34.5 and Python's
  banker's rounding sends it to 34.
- `wrap_words("aa bb cc dd", Fake10, 55)` → `["aa bb", "cc dd"]`; a line exactly `max_width` wide
  fits; `"abcdefghijkl mm"` at width 50 → `(["abcdefghijkl", "mm"], True)`.
- Ellipse inscribed box lands at scale ≈ 0.70 (width 280/400, height 140/200), inside the card's
  [0.66, 0.72] window; the rectangle polygon first accepts at scale 0.99 because the scale-1.00
  corners lie on the outline and count as outside (≥ 0.98 as the card requires).

## Deviations

- The card's fake `factory = lambda path, size: Fake(size)` is written as a `def fake_factory`
  (ruff E731 bans lambda assignment); the semantics are identical.
- `fonts_dir()` treats an empty `OMNISCAN_FONTS_DIR` string as unset (a `Path("")` would be the
  cwd); the card only says "if set", so this is the conservative reading.
- In the polygon branch of `inscribed_box` the four shrunk-and-rounded coordinates are passed
  through one `_degenerate_centre` clamp helper, which is also reused for the `polygon is None`
  branch; behaviour matches the card (zero-size box at the centre, never an invalid `BBox`).

## Questions

- None blocking. One observation for the later typesetting cards: with `polygon` given, the
  accepted rectangle is always centred on the bubble centre (per the card), which for very lopsided
  bubbles can leave usable area unused; the card explicitly lists per-line width adaptation as out
  of scope, so this is just a flag for the director.
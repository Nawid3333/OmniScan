# U3b — Reader core: strip view, side-by-side compare view, library service

## Changes

- **`src/omniscan/gui/__init__.py`** (new): `"""Desktop GUI (PySide6)."""` — the gui package.
- **`src/omniscan/gui/services/__init__.py`** (new): the Qt-free service layer's package docstring.
- **`src/omniscan/gui/services/library.py`** (new, Qt-free): `Tile` (y0/y1/path/label/kind),
  `ChapterView` (series, chapter, strip size, raw/output tile tuples, `has_output`), `list_series`
  (library ∪ output roots, skips `_`/`.` names, `natural_key` order), `list_chapter_names`
  (raw chapters via `SeriesPaths.chapters()`, output folders when the library has none),
  `load_chapter_view`:
  - raw tiles from `ingest.json` (`filtered` files skipped, `index` order) or, when missing/invalid,
    stacked `list_images(raw_dir)` scaled to the first image's width;
  - output tiles from `slices.json` + `export.json` (exported→image, filtered→gap, else→missing;
    an export entry whose file is gone counts as missing) or, when either is missing/invalid,
    stacked `list_images(output_dir)` scaled to the chapter strip width (first image's width when
    there is none);
  - chapter `strip_width`/`strip_height`: ingest wins, else slices, else the raw fallback, else the
    output fallback; `FileNotFoundError` only when neither raw nor output dir exists. Sizes are
    read with Pillow headers only (`Image.open(...).size`).
- **`src/omniscan/gui/strip_view.py`** (new): `StripView(QAbstractScrollArea)` exactly per the card —
  `strip_y_changed`/`zoom_changed`, `set_tiles` (reset to 0 + fit width), `tiles`, `zoom`, `set_zoom`
  (clamped 0.05–4.0, keeps the strip y of the viewport top), `fit_width` (refits on resize until a
  manual zoom), `strip_y`, `set_strip_y` (clamped, `emit=False` moves silently), `max_strip_y`,
  `tile_at`. The vertical scrollbar works in widget pixels (`round(strip_height*zoom) - vp_h`
  range, value `round(strip_y*zoom)`); the strip is centred when narrower than the viewport; a
  horizontal scrollbar appears only when `strip_width*zoom > viewport_width` (ceil, so the card's
  rule is exact). `paintEvent` paints only viewport-intersecting tiles at the size the tile's strip
  range dictates (SmoothPixmapTransform), gaps as BDiag-hatched rectangles with centred labels over
  the palette `Base`, images loaded lazily via `QImage(str(path))` into an LRU
  (`max_cached_images`), unreadable files drawn like missing tiles. Ctrl+wheel zooms ×1.15 per notch
  through `set_zoom`; plain wheel is forwarded to the vertical scrollbar (see Deviations #2).
- **`src/omniscan/gui/compare_view.py`** (new): `CompareView(QWidget)` per the card — top row
  (`sync_checkbox` "Linked scrolling", `fit_button` "Fit width", stretch, `chapter_label`
  `"<series> — <chapter>"`), `QSplitter` with both `StripView`s each under its caption
  (`left_caption` "Raw", `right_caption` "Output" / "Output (not translated yet)"). Linked mode
  tracks the view that scrolled/zoomed last as the master (also in independent mode, so switching
  back re-aligns the *other* view to it) and follows with `emit=False` — no feedback loops.
  `set_chapter` feeds both views and fits/zeroes them while cross-sync is blocked, then aligns the
  right view to the left (linked). Checkbox and `set_sync_mode` stay consistent without double
  `sync_mode_changed` emissions (the mode is set before the checkbox, so the toggled handler no-ops).
- **`scripts/gui_compare_demo.py`** (new): `SERIES CHAPTER [--screenshot OUT.png] [--size WxH]
  [--y STRIP_Y]` — loads `get_config()` (env overrides work), builds the `ChapterView`, shows the
  `CompareView`; `--screenshot` sets `QT_QPA_PLATFORM=offscreen` before importing Qt, scrolls to
  `--y`, saves `widget.grab()` and exits 0. Unknown series/chapter → stderr message, exit 1.
- **Tests** (new): `tests/gui/__init__.py`, `tests/gui/conftest.py` (`QT_QPA_PLATFORM=offscreen` at
  the top, session-scoped `qapp`), `tests/gui/test_strip_view.py` (13 tests: painting at strip y,
  clamping, zoom doubling/clamping/fit-refit, signal counts, `tile_at`, failed images, LRU, empty
  strip, wheel), `tests/gui/test_compare_view.py` (5 tests: set_chapter, no-output caption, linked
  scroll/zoom + same-strip-row pixels, independent + re-align on switch back, fit button + no
  signal loop), `tests/unit/test_gui_library.py` (10 tests: listing, full artifacts, missing
  slices, fallbacks, corrupt ingest, FileNotFoundError, and the demo script through `subprocess`).

## Tests

```
uv run --frozen pytest tests/unit/test_gui_library.py tests/gui -q   → 28 passed
uv run --frozen pytest -m "not gpu"                                  → 3069 passed, 20 deselected (1:04)
uv run --frozen ruff format . && uv run --frozen ruff check .         → clean
uv run --frozen pyright                                               → 0 errors, 0 warnings
```

All GUI tests run headless (`QT_QPA_PLATFORM=offscreen`, set in `tests/gui/conftest.py` before any
Qt import; every GUI test module starts with `pytest.importorskip("PySide6")`). The library tests
are Qt-free. The demo-script test (15) runs the script via `subprocess` with
`OMNISCAN_PATHS__{LIBRARY,WORK,OUTPUT}_ROOT` pointing at a tmp library/output built like test 2 and
verifies exit 0 + a 900×600 PNG (Pillow) and exit 1 + stderr message for an unknown series.
Pixel-level assertions grab the *viewport* (`view.viewport().grab()`) so frame/scrollbar chrome
does not shift coordinates; tests converge the widget size until the viewport is exactly the
intended size (scrollbar extents are style-dependent offscreen). The 3 warnings in the full run are
pre-existing (fastapi/starlette deprecations, a turbo.py numpy warning), none from this card.

Commits on `U3b`: `U3b: WIP library service`, `U3b: WIP strip view`, then the final
`U3b: strip view, compare view, library service`.

## Deviations

1. **Wheel events reach `wheelEvent` only via the viewport.** Qt delivers wheel events to the
   viewport, and `QAbstractScrollArea` forwards them to `wheelEvent`; a wheel event sent directly to
   the scroll area widget is swallowed by the default handler and never reaches the override
   (verified with a probe). The tests therefore send `QWheelEvent` through
   `QApplication.sendEvent(view.viewport(), ...)` — the card only says "through `sendEvent`".
2. **Plain wheel needs explicit forwarding.** The card says "plain wheel scrolls (default
   behaviour)", but `QAbstractScrollArea`'s default wheel handling ignores the wheel over the
   viewport (verified: value stays 0). `wheelEvent` therefore forwards plain wheel events to the
   vertical scrollbar, which implements Qt's usual wheel-scroll semantics (positive delta = up).
3. **Qt handler names get `# noqa: N802`.** The repo lint bans camelCase functions; Qt requires
   `resizeEvent`/`eventFilter`/`wheelEvent`/`paintEvent`. `pyproject.toml` is off-limits for this
   card, so the four overrides carry inline noqa comments (a later card may add a per-file ignore).
4. **`set_zoom` ends fit-width mode even when the zoom value is unchanged** (the card says "a manual
   `set_zoom` or Ctrl+wheel clears" the fit flag). Without this, pinning a zoom that happens to
   equal the fit zoom would silently re-fit on the next resize.
5. **Fit button in linked mode ends at the right side's fit zoom.** `fit_width()` is called on both
   (as the card says); in linked mode each emitted zoom is mirrored to the other side, so the final
   shared zoom is the last emitted one (the right view's). With equal splitter halves both are
   exactly fitted; with unequal halves the shared scale means one side is not exactly fitted — this
   is the "same strip rows at the same scale" rule from the card (see Questions).
6. **Test 15 lives in `tests/unit/test_gui_library.py`** (the file list has no separate demo-test
   file): it is Qt-free (subprocess-based) and reuses the module's full-artifact builder.
7. **`_update_bars(y=...)` keyword.** `set_zoom` captures the strip y *before* changing the zoom and
   hands it to the scrollbar update, so the old pixel position is rescaled, not misread.

## Questions

1. **Fit with unequal splitter halves (linked mode):** shared scale means one side cannot be
   exactly fitted. Should `fit_button` fit to the *minimum* of the two fit zooms (both sides fully
   visible) instead of the last side's fit? Left as-is for this card; a one-line choice for the
   main-window card.
2. **Synchronous image loading:** `paintEvent` loads tiles synchronously (`QImage`). For page-size
   tiles this is fast; if the reader ever loads huge originals, a later card may want async
   loading with a placeholder. No change needed now.
3. **Wheel zoom anchor:** the card says Ctrl+wheel zooms "around the viewport top"; that is what
   `set_zoom`'s keep-strip-y contract gives (the row at the viewport top stays fixed). Anchoring at
   the cursor instead would need per-event math — deferred unless wanted.
## Review addendum (director, 2026-09-20)
Rebased on `main`; `ruff format/check` clean, `pyright` 0 errors, GUI tests **28 passed**, full CPU suite green. Answers: (1) fit with unequal halves — keep as is; the main-window card decides; (2) synchronous loading is fine for now; (3) zoom around the viewport top is fine. **Live check on a real translated chapter** (`scripts/gui_compare_demo.py PepperCarrotKR "Episode 06" --screenshot … --y 3000`): the Korean raw pages (left) and the English output slices (right) show the same strip rows at the same scale, exactly aligned across page/slice boundaries. Two director fixes from looking at the screenshot: the header/caption rows took most of the height (the strips now get the stretch: `addWidget(..., 1)`), and the offscreen screenshot mode showed boxes instead of text because the offscreen Qt platform has no font database (the demo sets `QT_QPA_FONTDIR` to `C:\Windows\Fonts` on Windows).

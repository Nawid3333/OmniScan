# U3b — Reader core: strip view, side-by-side compare view, library service

**Owner:** GLM builder · **Branch:** `U3b` · **Worktree:** `V:\OmniScan-wt\U3b` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read `docs/DECISIONS.md` section "Desktop app" (PySide6, **native widgets, no WebEngine**, Qt-free service layer, strip-space alignment), `src/omniscan/core/paths.py` (`SeriesPaths`, `ChapterPaths`, `list_chapters`, `list_images`, `natural_key`), `src/omniscan/core/schemas.py` (`IngestArtifact`/`SourceFile`, `SlicesArtifact`/`Slice`, `ExportArtifact`/`ExportFile`, `Artifact.save/load`), `src/omniscan/web/app.py` (how the web API reads the same artifacts) and `tests/conftest.py`.

## Why
The desktop app (PySide6) is the product the owner wants. Its most important widget is the **reader / side-by-side viewer**: the raw chapter on the left and the translated output on the right, with **synchronised or independent scrolling** — also the base of the debugger and of manual review by translation groups. This card builds the viewer core and the Qt-free data layer behind it; the main window, settings and model manager come in other cards (U3a…).
Key fact that makes exact sync possible: `ingest` scales every raw page to the common strip width and records where it lands (`SourceFile.y0/y1`, strip space); `slice` cuts the strip into slices (`Slice.y0/y1`, strip space) and `export` writes one output image per non-filtered slice (`ExportFile.slice_index`). So **raw tiles and output tiles share one coordinate system — strip y** — and "linked scrolling" simply means "same strip y on both sides"; filtered slices show up as gaps on the output side.
PySide6 is an optional dependency (extra `gui`, already installed in the shared `.venv`; **never run `uv sync`**, use `uv run --frozen …` or `.venv\Scripts\python.exe`). Tests run headless: `QT_QPA_PLATFORM=offscreen`.

## Files you may create / modify
- `src/omniscan/gui/__init__.py` — exactly `"""Desktop GUI (PySide6)."""` (create; if it exists leave it)
- `src/omniscan/gui/services/__init__.py` — exactly `"""Qt-free service layer: plain data in, plain data out."""` (create; if it exists leave it)
- `src/omniscan/gui/services/library.py` (create; **must not import PySide6**)
- `src/omniscan/gui/strip_view.py`, `src/omniscan/gui/compare_view.py` (create)
- `tests/gui/__init__.py`, `tests/gui/conftest.py`, `tests/gui/test_strip_view.py`, `tests/gui/test_compare_view.py`, `tests/unit/test_gui_library.py` (create)
- `scripts/gui_compare_demo.py` (create)
- `docs/reports/U3b.md` (create)
Anything else is off-limits (especially `pyproject.toml`, `uv.lock`, `src/omniscan/core/**`, `cli.py`).

## Part 1 — `gui/services/library.py` (Qt-free)
```python
@dataclass(frozen=True, slots=True)
class Tile:
    y0: int                                   # strip-space rows [y0, y1)
    y1: int
    path: Path | None                         # None for gaps
    label: str                                # file name, or e.g. "filtered slice 4"
    kind: Literal["image", "filtered", "missing"]

@dataclass(frozen=True, slots=True)
class ChapterView:
    series: str
    chapter: str
    strip_width: int
    strip_height: int
    raw: tuple[Tile, ...]                     # raw pages in strip space
    output: tuple[Tile, ...]                  # output slices in strip space (empty when there is no output)
    has_output: bool

def list_series(cfg: Config) -> list[str]
def list_chapter_names(cfg: Config, series: str) -> list[str]
def load_chapter_view(cfg: Config, series: str, chapter: str) -> ChapterView
```
- `list_series`: names of the sub-directories of `cfg.paths.library_root` **and** of `cfg.paths.output_root` (union), skipping names that start with `_` or `.`, sorted with `natural_key`; a missing root contributes nothing.
- `list_chapter_names`: `SeriesPaths.from_config(cfg, series).chapters()`; if the library has none, the chapter folders under `output_root/<series>` (same ordering/skip rules via `list_chapters`).
- `load_chapter_view` (never raises for missing/invalid artifacts — it falls back; it raises `FileNotFoundError` only when neither the raw dir nor the output dir exists):
  - **Raw tiles.** If `ingest.json` loads (`IngestArtifact.load`), use `strip_width`/`strip_height` from it and one `Tile(f.y0, f.y1, raw_dir / f.name, f.name, "image")` per `SourceFile` whose `filtered` is false, in `index` order. Otherwise (missing/invalid): `list_images(raw_dir)` stacked from y=0; `strip_width` = width of the first image; each tile's height = `round(h * strip_width / w)` (headers only, via Pillow `Image.open(...).size`); `strip_height` = the total. No images → empty tuple, `strip_width=strip_height=0`.
  - **Output tiles.** If `slices.json` **and** `export.json` load: for each `Slice` in `index` order: if an `ExportFile` with that `slice_index` exists and its file exists in `output_dir` → `Tile(s.y0, s.y1, output_dir / name, name, "image")`; elif `s.filtered` → `Tile(s.y0, s.y1, None, f"filtered slice {s.index}", "filtered")`; else → `Tile(s.y0, s.y1, None, f"missing slice {s.index}", "missing")`. Otherwise fall back to `list_images(output_dir)` stacked from y=0 with heights scaled to the strip width as above (when there is no strip width yet, use the first output image's width). `has_output` = at least one tile of kind `image`.
  - The chapter's `strip_width`/`strip_height` are those of ingest or slices (whichever loaded; ingest wins), else from the fallbacks.

## Part 2 — `gui/strip_view.py`
```python
class StripView(QAbstractScrollArea):
    strip_y_changed = Signal(float)           # strip y of the viewport's top edge
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None, *, max_cached_images: int = 32) -> None
    def set_tiles(self, tiles: Sequence[Tile], strip_width: int, strip_height: int) -> None   # resets scroll to 0 and fits the width
    def tiles(self) -> tuple[Tile, ...]
    def zoom(self) -> float                    # widget pixels per strip pixel
    def set_zoom(self, zoom: float, *, emit: bool = True) -> None       # clamped to [0.05, 4.0]; keeps the strip y of the viewport top
    def fit_width(self, *, emit: bool = True) -> None                   # zoom = viewport width / strip_width (margin 0)
    def strip_y(self) -> float
    def set_strip_y(self, y: float, *, emit: bool = True) -> None       # clamped to [0, max_strip_y()]
    def max_strip_y(self) -> float             # max(0, strip_height - viewport_height / zoom)
    def tile_at(self, strip_y: float) -> Tile | None
```
Behaviour (exact):
- The vertical scroll bar works in **widget pixels**: range `0 … max(0, round(strip_height*zoom) - viewport_height)`, value `round(strip_y*zoom)`; `strip_y() == value / zoom`. Moving the bar (mouse wheel, drag, keys) emits `strip_y_changed` once per change. `set_strip_y(..., emit=False)` moves the bar **without** emitting (no feedback loops between linked views).
- The strip is drawn horizontally centred when narrower than the viewport; a horizontal scroll bar appears only when `strip_width*zoom > viewport_width`.
- `paintEvent` paints only tiles that intersect the viewport. An `image` tile is drawn by scaling its image into the target rectangle `(x_off, y0*zoom - value, strip_width*zoom, (y1-y0)*zoom)` — the **tile's strip range decides the size, not the image's own pixel size**. Images are loaded lazily with `QImage(str(path))` and kept in an LRU (`max_cached_images` entries); a file that fails to load is drawn like a `missing` tile with its label. `filtered`/`missing` tiles are drawn as a hatched rectangle (`Qt.BrushStyle.BDiagPattern`) with the label centred. Background outside tiles = the palette's `Base` colour. Use `SmoothTransformation` for scaling.
- Ctrl + wheel zooms by a factor 1.15 per notch around the viewport top (calls `set_zoom`); plain wheel scrolls (default behaviour).
- `tile_at(y)` returns the tile with `y0 <= y < y1` (None in a gap or outside the strip); `set_tiles` copes with an empty list (blank viewport, zero range).
- Resizing the widget keeps `strip_y()` and the zoom unless fit-width is active (`fit_width` sets a flag that a manual `set_zoom` or Ctrl+wheel clears; while it is set, a resize refits the width).

## Part 3 — `gui/compare_view.py`
```python
type SyncMode = Literal["linked", "independent"]

class CompareView(QWidget):
    sync_mode_changed = Signal(str)
    left: StripView                            # raw
    right: StripView                           # output
    sync_checkbox: QCheckBox                   # text "Linked scrolling", checked by default
    fit_button: QPushButton                    # text "Fit width"

    def __init__(self, parent: QWidget | None = None) -> None
    def set_chapter(self, view: ChapterView) -> None
    def sync_mode(self) -> SyncMode
    def set_sync_mode(self, mode: SyncMode) -> None
    def strip_y(self) -> float                 # of the master view
```
- Layout: a horizontal row on top (`sync_checkbox`, `fit_button`, a stretch, a label `chapter_label` with `"<series> — <chapter>"`), below it a `QSplitter(Qt.Horizontal)` with the two `StripView`s, each under a small caption label (`"Raw"` / `"Output"`); when `has_output` is false the right side shows the caption `"Output (not translated yet)"` and an empty strip.
- **Linked mode** (default): the view that emitted `strip_y_changed` / `zoom_changed` last is the *master*; the other one is updated with `set_strip_y(y, emit=False)` / `set_zoom(z, emit=False)`, so both always show the same strip rows at the same scale. `fit_button` calls `fit_width()` on both. **Independent mode**: neither follows the other. Switching back to `linked` re-aligns the non-master view to the master (the view scrolled/zoomed most recently; initially the left one) and emits `sync_mode_changed("linked")`. The checkbox and `set_sync_mode` stay consistent (toggling either updates the other without a second signal).
- `set_chapter` gives each view its tiles (`view.raw` / `view.output`) and the chapter's `strip_width`/`strip_height`, fits the width on both, and scrolls both to 0.

## Part 4 — `scripts/gui_compare_demo.py`
`uv run --frozen python scripts/gui_compare_demo.py SERIES CHAPTER [--screenshot OUT.png] [--size 1400x900] [--y STRIP_Y]`: loads `get_config()`, builds the `ChapterView`, shows a `CompareView`. With `--screenshot` it sets `QT_QPA_PLATFORM=offscreen` (before importing Qt), resizes the window, scrolls to `--y` (default 0), processes events, saves `widget.grab()` to the PNG and exits 0; without it, it runs the normal event loop. Unknown series/chapter → message on stderr, exit 1. (The director uses the screenshot mode to look at real output.)

## Definitions
- **Strip space** = the coordinates of `ingest.json` / `slices.json`: x is the strip width, y in rows.
- Gap = an output tile of kind `filtered` or `missing`.

## Acceptance tests (must exist and pass, offscreen)
`tests/gui/conftest.py`: `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")` at the very top (before any Qt import), a session-scoped `qapp` fixture (`QApplication.instance() or QApplication([])`), and every test module starts with `pytest.importorskip("PySide6")`.
**Library (`tests/unit/test_gui_library.py`, no Qt)**
1. `list_series` unions library and output roots, skips `_reference_en`/`.hidden`, natural order (`Series 2` before `Series 10`), missing roots → `[]`. `list_chapter_names` natural order (`Chapter 2` < `Chapter 10`), falls back to the output folders.
2. Full artifacts: build a tmp config (`Config(paths=…)` — see how other tests do it), write `ingest.json` (3 raw files, 800 px strip width; y-ranges 0–1000, 1000–2500, 2500–3000, one extra file with `filtered=True`), `slices.json` (4 slices: 0–1200, 1200–2400, 2400–2700 filtered, 2700–3000), `export.json` (files for slice 0, 1, 3) and the image files; `load_chapter_view` → 3 raw tiles (filtered file skipped), 4 output tiles with kinds `image, image, filtered, image`, correct y-ranges/labels/paths, `strip_width == 800`, `strip_height == 3000`, `has_output`.
3. A slice that is neither exported nor filtered → `missing` tile (also when the export entry exists but the file is gone).
4. Fallback without artifacts: 3 raw images (800×1000, 800×600, 400×400) → tiles `[0,1000)`, `[1000,1600)`, `[1600,2400)` (the last scaled ×2), `strip_width == 800`; output fallback from `list_images(output_dir)`; no output dir → `has_output` False and empty tuple; corrupt `ingest.json` behaves like a missing one; neither dir exists → `FileNotFoundError`.
**StripView (`tests/gui/test_strip_view.py`)** — tiles are solid-colour PNGs written to `tmp_path` (red 0–100, green 100–200, blue 200–300, strip 60 px wide) plus a gap tile 300–400:
5. At zoom 1 with a 60×100 viewport (`resize` + `show`, process events) `grab().toImage()` pixel at the middle of the viewport is red; after `set_strip_y(100)` it is green, after `set_strip_y(250)` (viewport shows 250–350) the top half is blue and the lower half is not blue (hatched gap); `strip_y()` returns what was set; `set_strip_y(-5)` → 0; `set_strip_y(10_000)` → `max_strip_y()`.
6. `set_zoom(2.0)` doubles the drawn size (a colour boundary that was at widget y=100 is now at 200) and keeps the strip y of the top edge; zoom clamps to `[0.05, 4.0]`; `fit_width()` on a 120 px wide viewport gives zoom 2.0.
7. Signals: moving the scroll bar programmatically (`verticalScrollBar().setValue`) emits `strip_y_changed` exactly once with `value / zoom`; `set_strip_y(y, emit=False)` emits nothing; `set_zoom` emits `zoom_changed` once (and none with `emit=False`).
8. `tile_at`: inside each tile, at the boundaries (`y0` inclusive, `y1` exclusive), in the gap tile (returns the gap tile), below the strip (`None`).
9. An image that is not a valid file is drawn as a hatched/missing tile without raising; more tiles than `max_cached_images` loads at most that many images at a time (spy on the loader or inspect the cache size); `set_tiles([], 0, 0)` shows a blank view without errors.
10. Ctrl + wheel zoom (`QTest.mouseClick` is not needed: send a `QWheelEvent` with `Qt.ControlModifier` through `QApplication.sendEvent`) changes the zoom by 1.15; a plain wheel event scrolls.
**CompareView (`tests/gui/test_compare_view.py`)**
11. `set_chapter` with a `ChapterView` (raw 0–300 as above, output tiles with a gap) shows the tiles in both views, both at strip y 0, same zoom; `chapter_label` text is `"S — Chapter 1"`; `has_output=False` → the caption is `"Output (not translated yet)"`.
12. Linked: `left.set_strip_y(150)` → `right.strip_y() == 150`; scrolling the right view moves the left one; `set_zoom` on one changes the other; the grabbed pixel colours of both views at the same widget y show the **same strip row** (left red-tile row ↔ right tile of that row).
13. Independent: `sync_checkbox.setChecked(False)` (and `set_sync_mode("independent")`) → moving one view leaves the other unchanged; switching back re-aligns to the master (the view moved last) and emits `sync_mode_changed("linked")` exactly once; the checkbox and `sync_mode()` stay consistent.
14. `fit_button` click fits both views; no signal feedback loop (count the `strip_y_changed` emissions of each view for one scroll: master emits once, the follower zero).
**Demo script**
15. Run `scripts/gui_compare_demo.py` through `subprocess` against a tmp library/output built like test 2 (point `OMNISCAN_PATHS__LIBRARY_ROOT`, `OMNISCAN_PATHS__OUTPUT_ROOT`, `OMNISCAN_PATHS__WORK_ROOT` at the tmp dirs) with `--screenshot out.png --size 900x600`: exit code 0, the PNG exists and is 900×600 (check with Pillow); an unknown series exits 1.

## Out of scope
Main window, menus, settings, model manager, run/queue views, editing tools, region overlays (boxes/text on the strip — a later card builds on `StripView`), caching to disk, animations, theming, packaging.

## Commands to run before finishing
```bash
uv run --frozen pytest tests/unit/test_gui_library.py tests/gui -q
uv run --frozen pytest -q -m "not gpu"
uv run --frozen ruff format . && uv run --frozen ruff check .
uv run --frozen pyright
```

## Report
`docs/reports/U3b.md`: Changes, Tests (commands + results), Deviations, Questions. **Commit early** (`U3b: WIP library service`) once Part 1 and its tests pass, again after `StripView`; the final commit is `U3b: strip view, compare view, library service`. You have 150 tool calls in total. If anything is unclear: stop, write the question under Questions, commit, and end.

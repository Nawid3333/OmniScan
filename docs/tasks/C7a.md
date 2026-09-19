# C7a — Typeset layout engine: fit English text into a box (pure, no rendering)

**Owner:** GLM builder · **Branch:** `C7a` · **Worktree:** `V:\OmniScan-wt\C7a` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read `LayoutItem`, `FontRole`, `BBox`, `RGB` in `src/omniscan/core/schemas.py`, `fonts/README.md`
(the OFL fonts you will use in tests), and how PIL measures text: `ImageFont.FreeTypeFont.getlength(text)` returns the advance
width in pixels of a single line.

## Goal
The typesetter (later cards) has to place an English line into a speech bubble. This card builds the **decision engine**, without
drawing anything: given a text, a target rectangle and a font, find the largest font size at which the greedily word-wrapped text
fits, and produce a `LayoutItem`. It also computes the rectangle that fits inside an elliptical/irregular bubble outline. Pure
Python + PIL font metrics; CPU only; no image is created or saved; no GPU; no network.

## Files you may create / modify
- `src/omniscan/typeset/__init__.py` (create, empty)
- `src/omniscan/typeset/fonts.py`, `src/omniscan/typeset/fit.py` (create)
- `tests/unit/test_typeset_fonts.py`, `tests/unit/test_typeset_fit.py` (create)
- `docs/reports/C7a.md` (create)
Nothing else. Do not modify `src/omniscan/core/**` or `fonts/`.

## Part 1 — `typeset/fonts.py`
```python
DEFAULT_FONT_FILES: dict[FontRole, str]   # exactly the table below

def fonts_dir() -> Path
def default_font_path(role: FontRole) -> Path
def load_font(path: Path, size_px: int) -> ImageFont.FreeTypeFont
```
- `DEFAULT_FONT_FILES`: `dialogue` → `ComicNeue-Bold.ttf`, `thought` → `PatrickHand-Regular.ttf`, `shout` → `Bangers-Regular.ttf`,
  `narration` → `ComicNeue-Regular.ttf`, `free` → `ComicNeue-Bold.ttf`, `sfx` → `Bangers-Regular.ttf`.
- `fonts_dir()`: the directory named by the environment variable `OMNISCAN_FONTS_DIR` if set, else `<repo root>/fonts` computed as
  `Path(__file__).resolve().parents[3] / "fonts"` (a development default; packaging will replace it later — say so in the docstring).
- `default_font_path(role)` = `fonts_dir() / DEFAULT_FONT_FILES[role]`; a file that does not exist →
  `FileNotFoundError(f"font not found: {path}")`.
- `load_font(path, size_px)`: `ImageFont.truetype(str(path), size_px)`, cached with `functools.lru_cache(maxsize=64)` on
  `(path, size_px)`; `size_px < 1` → `ValueError`; a missing file → the same `FileNotFoundError` text as above.

## Part 2 — `typeset/fit.py`
```python
class Measurable(Protocol):
    def getlength(self, text: str) -> float: ...

FontFactory = Callable[[Path, int], Measurable]

def wrap_words(text: str, font: Measurable, max_width: float) -> tuple[list[str], bool]
@dataclass(frozen=True, slots=True)
class Fit:
    size_px: int
    lines: list[str]
    width: int        # ceil of the widest line's advance; 0 for no lines
    height: int       # len(lines) * line_height(size_px, line_spacing)
    overflow: bool
def line_height(size_px: int, line_spacing: float) -> int
def fit_text(text: str, max_w: int, max_h: int, font_path: Path, *, min_px: int = 14, max_px: int = 48,
             line_spacing: float = 1.15, font_factory: FontFactory = load_font) -> Fit
def inscribed_box(bubble: BBox, polygon: Sequence[tuple[int, int]] | None, *, margin_px: int = 6) -> BBox
def layout_region(region_id: str, text: str, target: BBox, *, role: FontRole = "dialogue", font_path: Path | None = None,
                  min_px: int = 14, max_px: int = 48, line_spacing: float = 1.15,
                  align: Literal["center", "left", "right"] = "center", color: RGB = (0, 0, 0), stroke_px: int = 0,
                  stroke_color: RGB = (255, 255, 255), font_factory: FontFactory = load_font) -> LayoutItem
```
Definitions (exact):
- `wrap_words(text, font, max_width)`: `words = text.split()` (any whitespace). Empty → `([], False)`. Greedy: start a line with the first
  word; add the next word (joined by one space) while `font.getlength(candidate) <= max_width` (equal fits); otherwise close the line and
  start a new one with that word. A **single word** whose own `getlength` exceeds `max_width` still forms its own line and sets the returned
  flag `too_wide = True` (no hyphenation, no splitting). Returns `(lines, too_wide)`.
- `line_height(size_px, line_spacing)` = `round(size_px * line_spacing)` (Python `round`, never below 1).
- `fit_text`: validate `1 <= min_px <= max_px`, `max_w >= 1`, `max_h >= 1`, `line_spacing > 0` else `ValueError`. Normalise `text` with
  `" ".join(text.split())`. Empty → `Fit(size_px=max_px, lines=[], width=0, height=0, overflow=False)`. Otherwise for `size` from `max_px` down to
  `min_px` (integers, step 1): `font = font_factory(font_path, size)`, `lines, too_wide = wrap_words(text, font, max_w)`,
  `height = len(lines) * line_height(size, line_spacing)`; the first size with `not too_wide and height <= max_h` wins:
  `Fit(size, lines, width=ceil(max(getlength(line))), height, overflow=False)`. If no size fits, return the result for `min_px` with
  `overflow=True`.
- `inscribed_box(bubble, polygon, margin_px)`:
  - `polygon is None` → the bubble box shrunk by `margin_px` on every side.
  - otherwise let `w, h` be the bubble box size and `(cx, cy)` its centre. For `s` in `[1.00, 0.99, ..., 0.20]` (step 0.01, first accepted wins)
    take the rectangle centred at `(cx, cy)` with half-sizes `(s*w/2, s*h/2)`; it is accepted when 24 sample points on its boundary (the four
    corners plus 5 evenly spaced interior points per side, i.e. 4 + 4*5) all lie inside the polygon (even-odd rule; points exactly on the outline count as outside).
    Shrink the accepted rectangle by `margin_px` on every side and round to ints with `round`. If no `s` is accepted, use `s = 0.20`.
    In every case the result must keep `x1 >= x0` and `y1 >= y0` (clamp the margin so a tiny bubble degenerates to a zero-size box at its centre, never to an invalid `BBox`).
- `layout_region(...)`: `path = font_path or default_font_path(role)`; `fit = fit_text(text, target.width, target.height, path, ...)`;
  the text block box is centred inside `target`: `x0 = target.x0 + (target.width - fit.width) // 2`, `y0 = target.y0 + (target.height - fit.height) // 2`,
  `BBox(x0, y0, x0 + fit.width, y0 + fit.height)` (it may exceed `target` when `overflow` is true — do not clamp). Return
  `LayoutItem(region_id=..., font_role=role, font=path.name, size_px=fit.size_px, lines=fit.lines, box=that box, align=align, color=color,
  stroke_px=stroke_px, stroke_color=stroke_color, overflow=fit.overflow)`.

## Acceptance tests (CPU only; deterministic — most use a fake font)
Fake font used by tests 5–9: `class Fake: def __init__(self, size): self.size = size; def getlength(self, text): return len(text) * self.size * 0.5`
and `factory = lambda path, size: Fake(size)`.
**Fonts:**
1. `DEFAULT_FONT_FILES` has exactly the six roles above; `default_font_path("dialogue")` is `fonts_dir()/"ComicNeue-Bold.ttf"` and exists; with
   `OMNISCAN_FONTS_DIR` monkeypatched to an empty tmp dir it raises the documented `FileNotFoundError`; with it pointing at a tmp dir containing a
   copied `ComicNeue-Bold.ttf` it resolves there.
2. `load_font` returns a font whose `getlength("Hello") > 0`, is cached (same object for the same arguments), differs per size, and
   `load_font(path, 0)` raises `ValueError`; a missing path raises the documented error.
**Wrapping:**
3. `wrap_words("aa bb cc dd", Fake10, 55)` (with `Fake10.getlength = len(t) * 10`) → `(["aa bb", "cc dd"], False)`; a line that is exactly `max_width`
   wide fits; `"abcdefghijkl mm"` at width 50 → `(["abcdefghijkl", "mm"], True)`; empty and whitespace-only → `([], False)`; tabs/newlines count as spaces.
4. `line_height(40, 1.0) == 40`, `line_height(30, 1.15) == 34` (34.5 rounds to 34), `line_height(1, 0.1) == 1`.
**Fitting (fake font):**
5. `fit_text("hello world", 100, 100, p, min_px=10, max_px=40, line_spacing=1.0, font_factory=factory)` → `Fit(40, ["hello", "world"], 100, 80, False)`.
6. Same with `max_h=60` → `Fit(30, ["hello", "world"], 75, 60, False)`.
7. `max_w=20, max_h=100` → overflow: `Fit(10, ["hello", "world"], 25, 20, True)` (min size, a word wider than the box).
8. Empty/whitespace text → `Fit(max_px, [], 0, 0, False)`; input `"  hello \n world  "` equals the result for `"hello world"`.
9. Validation: `min_px=0`, `max_px < min_px`, `max_w=0`, `max_h=0`, `line_spacing=0` each raise `ValueError`. A generous box returns `max_px`.
**Fitting (real fonts, ComicNeue-Bold and Bangers from `fonts/`) — property tests over 8 English sentences × 6 boxes (from 60×30 to 400×200):**
10. When `overflow` is false the real measurements agree: every line's `getlength <= max_w`, `height <= max_h`; the size is optimal:
    `size_px + 1` (when `<= max_px`) does not fit under the same rule (wrap again with the real font at that size); lines joined by a space equal the normalised text.
11. Monotonic: enlarging the box in both dimensions never lowers `size_px`. When `overflow` is true, `size_px == min_px`.
**Inscribed box:**
12. `polygon=None`: `inscribed_box(BBox(100,100,300,200), None, margin_px=6)` → `BBox(106,106,294,194)`; a tiny bubble `BBox(10,10,14,14)` with margin 6 →
    a valid zero-size `BBox` at its centre `(12,12,12,12)`.
13. Ellipse polygon (48 points of the ellipse inscribed in `BBox(0,0,400,200)`, computed in the test with `math.cos/sin` and rounded), `margin_px=0`:
    width in `[0.66*400, 0.72*400]` and height in `[0.66*200, 0.72*200]`, centred on `(200,100)` (± 1 px); with `margin_px=10` it is 20 px narrower and shorter (± 1).
14. Rectangle polygon = the bubble box's four corners: the result is (almost) the whole box (`s` stays ≥ 0.98); a concave "L" polygon never yields a
    rectangle with a sample point outside it (assert with your own point-in-polygon in the test).
**Layout item:**
15. `layout_region("r0001", "Are you okay? The dungeon just opened!", BBox(100,100,300,200), role="dialogue")` (real font) returns a valid `LayoutItem`:
    `font == "ComicNeue-Bold.ttf"`, `font_role == "dialogue"`, `size_px` between 14 and 48, `box` centred in the target (± 1 px on each axis), `lines` non-empty;
    with `role="shout"` the font is `Bangers-Regular.ttf`; `align`, `color`, `stroke_px`, `stroke_color` are passed through; an impossible fit
    (a 10 × 10 target with a long sentence) has `overflow=True`, `size_px == 14` and a `box` larger than the target (not clamped).
16. The whole suite, ruff and pyright are green.

## Out of scope
Drawing, glyph rasterisation, stroke geometry, compositing, hyphenation, per-line width adaptation to the bubble outline, font-role
classification, colour detection, the LLM "condense" path, any CLI or artifact writing.

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_typeset_fonts.py tests/unit/test_typeset_fit.py -q
uv run pytest -q
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
Write `docs/reports/C7a.md` (Changes, Tests, Deviations, Questions) and commit `C7a: typeset layout engine`.
If anything above is unclear: stop, write the question under Questions, commit what you have, and end.

# C7b — Typeset stage: `omniscan typeset` (regions + English text + fill colours → `layout.json`)

**Owner:** GLM builder · **Branch:** `C7b` · **Worktree:** `V:\OmniScan-wt\C7b` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md`, `docs/ARCHITECTURE.md` (the "Render pass" section) first. Then read the merged layout engine `src/omniscan/typeset/fit.py` and `fonts.py` (C7a: `layout_region`, `inscribed_box`, `load_font`),
`TypesetConfig` in `src/omniscan/core/config.py` (already present), `Region`, `FinalLine`, `FinalArtifact`, `InpaintArtifact`, `LayoutItem`, `LayoutArtifact`, `RegionsArtifact`, `BBox`, `RGB` in `src/omniscan/core/schemas.py`,
`src/omniscan/translate/prompts.py` (`translatable`), and `src/omniscan/slicer/stage.py` (a stage without a GPU group).

## Goal
For every region that has a final English line, decide **where** the text goes, **which font role** it uses, and **what colour** it has, run the C7a fitting engine, and write `layout.json` (`LayoutArtifact`).
Nothing is drawn here (a later card renders and composites). Pure Python, CPU only; no image is created.

## Files you may create / modify
- `src/omniscan/typeset/plan.py`, `src/omniscan/typeset/stage.py` (create)
- `src/omniscan/cli.py` (modify — ONLY replace the `typeset` stub with the real command and remove `"typeset"` from `_STUB_COMMANDS`)
- `tests/unit/test_typeset_plan.py`, `tests/unit/test_typeset_stage.py` (create); `tests/unit/test_cli.py` (modify — only what the stub removal requires)
- `README.md` (status row `typeset` → `working — omniscan typeset (layout.json; needs ocr.json, final.json, inpaint.json)`; drop `typeset` from the "Not implemented yet" table) and `docs/USER_GUIDE.md`
  (a `### omniscan typeset` subsection in the style of the others, the `layout.json` row of the folder table, every `[typeset]` config key in the config table) — `tests/unit/test_docs.py` must stay green
- `docs/reports/C7b.md` (create)
Do not modify `src/omniscan/core/**` (`TypesetConfig` is finished) or the C7a modules (import from them).

## Part 1 — `typeset/plan.py`
```python
def luminance(rgb: RGB) -> float
def ellipse_polygon(box: BBox, points: int = 48) -> list[tuple[int, int]]
def target_box(region: Region, cfg: TypesetConfig) -> BBox
def plan_layout(regions: Sequence[Region], lines: Mapping[str, str], fills: Mapping[str, RGB], cfg: TypesetConfig,
                *, font_factory: FontFactory = load_font) -> list[LayoutItem]
```
Definitions (exact):
- `luminance(rgb)` = `0.299*r + 0.587*g + 0.114*b`.
- `ellipse_polygon(box, points)`: centre `cx = (x0+x1)/2`, `cy = (y0+y1)/2`, radii `a = width/2`, `b = height/2`; the points `(round(cx + a*cos(t)), round(cy + b*sin(t)))` for `t = 2*pi*i/points`, `i = 0..points-1` (clockwise on screen because y points down).
- `target_box(region, cfg)`:
  - `kind == "bubble_text"` and `region.bubble_bbox` is not None: `ell = inscribed_box(region.bubble_bbox, ellipse_polygon(region.bubble_bbox), margin_px=cfg.margin_px)`; return `ell` unless `region.bbox` has a **strictly larger** area
    than `ell`, in which case return `region.bbox` (the space the original text used lies inside the bubble; the inscribed ellipse box is safe for any bubble shape).
  - every other case (`free_text`, `sfx`, or a `bubble_text` without `bubble_bbox`): `region.bbox` grown on each side by `round(width * cfg.free_grow)` horizontally and `round(height * cfg.free_grow)` vertically, with `x0`/`y0` clamped to `>= 0`.
- Role by kind: `bubble_text` → `"dialogue"`, `free_text` → `"free"`, `sfx` → `"sfx"`. (`watermark` regions are never laid out: `translatable` drops them.)
- Colours: text colour = `region.text_color` when it is not None; otherwise for `bubble_text`: `(0,0,0)` when `luminance(fills[region.id]) >= 128` or no fill is known for the region, else `(255,255,255)`; for `free_text` and `sfx`: `(255,255,255)`.
  Stroke: `bubble_text` → `stroke_px=0`; `free_text` → `cfg.stroke_free_px`; `sfx` → `cfg.stroke_sfx_px`; `stroke_color` = `region.stroke_color` when set, else `(0,0,0)`.
- `plan_layout(regions, lines, fills, cfg)`: iterate `translatable(regions)` (already sorted); take `text = lines.get(region.id, "")`; skip regions whose text is empty after `.strip()`; else
  `layout_region(region.id, text, target_box(region, cfg), role=..., min_px=cfg.min_px, max_px=cfg.max_px, line_spacing=cfg.line_spacing, color=..., stroke_px=..., stroke_color=..., font_factory=font_factory)`. Result in that order.

## Part 2 — `typeset/stage.py` and CLI
```python
class TypesetStage:
    name: ClassVar[str] = "typeset"
    version: ClassVar[int] = 1
    gpu_group: ClassVar[str | None] = None
```
- `inputs(ctx)` = `[ocr.json, final.json, inpaint.json]` (artifact paths); `outputs(ctx)` = `["layout.json"]`; `config_subset(cfg)` = `cfg.typeset.model_dump()`.
- `run(ctx, models)`: a missing input → `FileNotFoundError("<file> missing — run the <ocr|judge|inpaint> stage first")`; `regions = RegionsArtifact.load(ocr.json).regions`; `lines = {l.region_id: l.text for l in FinalArtifact.load(final.json).lines}`;
  `fills = {i.region_id: i.fill for i in InpaintArtifact.load(inpaint.json).items if i.fill is not None}`; `items = plan_layout(...)`; `LayoutArtifact(items=items).save(layout.json)`;
  metrics (all `float`): `items` (laid out), `overflow` (items with `overflow` true), `skipped` (translatable regions without a usable text).
- `omniscan typeset SERIES [--chapter/-c CHAPTER]... [--force]`: same options and help style as `slice`; runs only `[TypesetStage()]` through `_run_stages`.

## Acceptance tests (CPU, no downloads; use the fake font factory `def factory(path, size): return Fake(size)` with `Fake.getlength(text) = len(text) * size * 0.5` for tests that must be independent of real fonts)
1. `luminance((128,128,128)) == 128.0` (within 1e-9), `luminance((0,0,0)) == 0`, `luminance((255,255,255)) == 255` (within 1e-9); `(127,127,127)` gives `127.0`.
2. `ellipse_polygon(BBox(0,0,400,200))` has 48 points; the first is `(400, 100)`; the point at index 12 is `(200, 200)` (t = π/2 → bottom on screen); all points lie inside the box; changing `points` changes the count; the polygon is closed-consistent (no duplicate consecutive points).
3. `target_box` for a `bubble_text` region with `bubble_bbox=BBox(0,0,400,200)` and a small `bbox=BBox(150,80,250,120)` → the inscribed ellipse box for `margin_px=6`, i.e. `BBox(66, 36, 334, 164)`; with a large `bbox=BBox(20,20,380,180)` (area 57 600 > 34 304) → that `bbox` is returned;
   with `bbox` of exactly the same area as the ellipse box → the ellipse box; a `bubble_text` without `bubble_bbox` → `bbox` grown by `free_grow` (`BBox(90,70,110,90)` with `free_grow=0.10` → `BBox(88,68,112,92)`); a `free_text` region near the origin (`BBox(2,2,52,32)`) → `x0`/`y0` clamped at 0: `BBox(0, 0, 57, 35)`; an `sfx` region behaves like `free_text`.
4. Role/colour/stroke table (one test per case with `plan_layout` and the fake factory): bubble on a `(255,255,255)` fill → dialogue, colour `(0,0,0)`, stroke 0; bubble on a `(24,24,32)` fill → `(255,255,255)`; a bubble with no fill known → `(0,0,0)`; `free_text` → role `free`, colour white, stroke `cfg.stroke_free_px`, stroke colour black;
   `sfx` → role `sfx`, stroke `cfg.stroke_sfx_px`; `region.text_color=(10,20,30)` and `region.stroke_color=(1,2,3)` override the defaults.
5. `plan_layout` skips regions with an empty/whitespace/missing line and `watermark` regions; keeps `translatable` order (`r2` before `r10`); every item's `region_id`, `font` name (`ComicNeue-Bold.ttf` for dialogue/free, `Bangers-Regular.ttf` for sfx — use the real font-name mapping of C7a with the fake factory) and `box` come from `layout_region`; a text too long for its box yields `overflow=True` at `size_px == cfg.min_px`.
6. Real fonts: a `bubble_text` region on a white 400×200 bubble with the sentence `"Are you okay? The dungeon just opened!"` gives one item whose `box` lies inside the bubble's inscribed ellipse box and whose size is between `min_px` and `max_px` (uses the real `load_font`, so the OFL fonts in `fonts/` must resolve).
7. Stage (CPU config as in `tests/unit/test_stages_ingest_slice.py`, hand-written `ocr.json`, `final.json`, `inpaint.json`): writes a loadable `layout.json`, manifest `typeset` `done`, second run `skipped`; changing `typeset.max_px` or any of the three inputs re-runs it; a missing input → recorded failure with the documented message; metrics `items/overflow/skipped` as defined.
8. `omniscan typeset S` (CLI runner) runs only the typeset stage, prints one line per chapter, exit 0; `--force`/`--chapter` work; unknown series → exit 2; `omniscan slice S` unaffected; `tests/unit/test_docs.py` and the whole suite stay green.

## Out of scope
Rendering glyphs, compositing, font-role detection beyond the kind table (shout/thought/narration), per-line width adaptation to the bubble outline, colour sampling from the image, LLM condensing of overflowing text, changing `TypesetConfig`.

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_typeset_plan.py tests/unit/test_typeset_stage.py tests/unit/test_cli.py tests/unit/test_docs.py -q
uv run pytest -q
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
Write `docs/reports/C7b.md` (Changes, Tests, Deviations, Questions) and commit `C7b: typeset stage and omniscan typeset`. If anything above is unclear: stop, write the question under Questions, commit what you have, and end.

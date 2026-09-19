# E2b — `omniscan eval`: score the later Pepper&Carrot episodes too (`<text>` boxes), character-weighted recall, page-level OCR chrF

**Owner:** GLM builder · **Branch:** `E2b` · **Worktree:** `V:\OmniScan-wt\E2b` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first, then the merged card `docs/tasks/E2.md`, the report `docs/reports/E2.md` (Deviations, Questions, review addendum), and the code it produced: `src/omniscan/eval/{truth,metrics,score}.py`, the `eval` command in `src/omniscan/cli.py` and the four `tests/unit/test_eval_*.py` files. This card **extends** them; every existing behaviour and test stays unless a line below says otherwise.

## Why (measured on 2026-09-20)
`omniscan eval PepperCarrotKR` over all 33 Korean episodes found truth boxes in only 16 of them: episodes 1–15 (and a few boxes in 29) store their text as `flowRoot` boxes, episodes 16–37 store it as `<text>` elements with line `<tspan>`s and **no box**, so the parser saw nothing (`truth_boxes 0`, precision printed as 0.0). Also, recall per box is dominated by hundreds of tiny SFX boxes, and `missed` lists huge stylised SFX first. This card fixes all three.
The real SVGs to look at (read only, never copy): `V:\OmniScan\data\translated-check\PepperCarrotKR\Episode 22\truth\kr\E22P05.svg` (a `<text>` with `text-anchor:middle` and several line tspans) and `Episode 06\truth\kr\E06P01.svg` (flowRoot).

## Files you may create / modify
- `src/omniscan/eval/truth.py`, `eval/score.py`, `eval/metrics.py` (only if needed), and the `eval` command block in `src/omniscan/cli.py` (print format only)
- the four existing `tests/unit/test_eval_*.py` (extend; existing assertions must keep passing unless they contradict Part 4)
- `docs/USER_GUIDE.md` (the `### omniscan eval` section: new fields and the approximation note), `docs/reports/E2b.md` (create)
Nothing else; do not touch `core/**`.

## Part 1 — `<text>` boxes in `truth.py`
`TruthBox` gets a new field `approx: bool = False` (last, defaulted; existing constructions stay valid). For every SVG element `text` (any namespace prefix) that is **not** inside a `flowRoot`, build one approximate box:
- **Lines:** the child `tspan`s whose attribute `{http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd}role` equals `"line"`, in document order; if there is none, one line from the element's own text. A line's text is `"".join(el.itertext()).strip()`; empty lines are skipped; an element without any non-empty line is skipped (not counted as dropped).
- **Position of a line:** the `x` and `y` attributes of its tspan, else of the `text` element (a whitespace/comma separated list → the first number); both missing → the line is skipped.
- **Font size:** the CSS property `font-size` (a number followed by optional `px`) from the tspan's `style`, else from the text element's `style`, else `16.0`.
- **Anchor:** the CSS property `text-anchor` from the tspan's style, else the element's style, else `start` (values `start`, `middle`, `end`).
- **Line width:** `w = font_size * sum(1.0 if unicodedata.east_asian_width(ch) in ("W", "F") else 0.5 for ch in text)`.
- **Box of a line in the element's coordinates:** `left = x - w` (end), `x - w/2` (middle), `x` (start); `right = left + w`; `top = y - font_size`; `bottom = y + 0.25 * font_size`. The element's box is the union of its lines' boxes.
- Then exactly the same pipeline as for a `flowRoot` box: the `transform` of the element and of its ancestor groups composes (reuse your code), the four corners → min/max, `sx`, `crop_top`, clamping, dropping of boxes with zero area, strip mapping and the floor/ceil rounding. `TruthBox.approx = True`, `lines` = the line texts.
- Boxes from `flowRoot` and from `text` elements share one list; **document order is the order of appearance in the file** (walk the tree once and handle both kinds).
- `load_english_pages` must now include the `<text>` lines too (same walk).
Golden values (SVG width 2000, height 3000, `file_width=1000, file_height=1480, file_y0=0, scale=1.0`, so `sx = 0.5`, `crop_top = 20`; namespace `xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"`):
1. `<text style="font-size:20px;text-anchor:middle" x="1000" y="500" transform="translate(100,50)"><tspan sodipodi:role="line" x="1000" y="500" style="font-size:40px">안녕하세요</tspan><tspan sodipodi:role="line" x="1000" y="540" style="font-size:40px">ab</tspan></text>` → `BBox(500, 235, 600, 280)`, lines `("안녕하세요", "ab")`, `approx=True` (line 1: width 200, 900..1100, y 460..510; line 2: width 40, 980..1020, y 500..550; union 900..1100 × 460..550, translated by (100, 50), then × 0.5 and the top crop).
2. `<text x="100" y="200" style="font-size:30px"><tspan sodipodi:role="line" x="100" y="200">가나</tspan></text>` → `BBox(50, 65, 80, 84)` (start anchor, width 60).
3. `<text style="text-anchor:end" x="300" y="100"><tspan sodipodi:role="line" x="300" y="100" style="font-size:10px">abcd</tspan></text>` → `BBox(140, 25, 150, 32)` (end anchor, width 20).
4. A `text` without tspans (`<text x="10" y="50" style="font-size:20px">가</text>`) yields one box; a text whose lines are all empty yields nothing; a `text` inside a `flowRoot` is not turned into a second box; one SVG with both a flowRoot and a text gives two boxes in file order.

## Part 2 — character-weighted recall and a better `missed` list (`score.py`)
- New `EvalReport` field `recall_chars: float | None` = (sum of `len(normalize(box.text))` over **detected** usable truth boxes) / (the same sum over **all** usable truth boxes); `None` when the denominator is 0. It appears in `to_json` next to `recall`.
- `missed` is now sorted by `len(normalize(text))` **descending**, ties by bbox area descending, then by `(page, y0, x0)`; still at most 10. (This replaces the "largest area first" rule of E2.)
- **Precision** is `None` when the chapter has no usable truth boxes (it was 0.0); the CLI prints `n/a` for it.

## Part 3 — page-level OCR chrF (`score.py`, CLI)
New `EvalReport` fields `ocr_chrf_mean: float | None` and `ocr_chrf_pages: int`. For every page number `n` that has usable truth boxes: reference = the texts of the page's usable truth boxes (`box.text`) joined by `" "` in document order; hypothesis = the `text` of every non-watermark region whose center row lies in the page's strip rows `[y0, y1)` (same rule as the translation score), joined by `" "` in `(bbox.y0, bbox.x0)` order; the page score is `chrf(hypothesis, reference)` (an empty hypothesis scores `0.0`). `ocr_chrf_mean` is the mean over these pages, `None` when there are none. This score does not depend on the box assignment, so it also works when boxes are approximate.

## Part 4 — CLI output
Per chapter (extends the E2 block; keep the first line and the `translation` line as they are):
```
  detection    recall 0.357 (56/157)  chars 0.712  precision 1.000 (67/67 regions)
  OCR          CER macro 0.749  micro 0.113  (56 boxes)  page chrF 0.831 (11 pages)
```
`chars` is `recall_chars`; every `None` prints `n/a`; when `truth_boxes - ignored_boxes == 0` print `  detection    n/a (no usable truth boxes)` and omit the OCR line. The first line gains the number of approximate boxes when there are any: `..., 157 truth boxes (3 ignored, 0 dropped, 12 approximate)`. `worst CER` and `missed` lines stay. `--json` and `eval.json` contain the new fields (`recall_chars`, `ocr_chrf_mean`, `ocr_chrf_pages`, `approx_boxes`).

## Acceptance tests (no reads from `data/`)
1. **Truth:** the four golden cases of Part 1 (box, lines, `approx`), plus: `text-anchor` given only on the tspan; a tspan without `x`/`y` falling back to the element's; `font-size` without `px`; a list-valued `x="10 20 30"` uses the first number; a `transform` on an ancestor `<g>` shifts a text box; a mixed SVG keeps file order; `load_english_pages` includes the text lines.
2. **Recall by characters:** truth boxes with 10, 2 and 1 usable characters where only the 10-character box is detected → `recall == 1/3`, `recall_chars == 10/13`; no usable boxes → both `None` and `precision is None`.
3. **Missed order:** boxes with 2, 9 and 9 characters (the two 9s with different areas) undetected → order: 9 chars big area, 9 chars small area, 2 chars; more than 10 → truncated to 10.
4. **Page OCR chrF:** a two-page ingest where page 1's regions read exactly the truth text → `1.0`, page 2 has truth but no region → `0.0`, mean `0.5`, `ocr_chrf_pages == 2`; regions of a watermark kind are ignored; a page without truth boxes is not counted; regions are joined in `(y0, x0)` order (assert with a reference whose chrF differs for the swapped order using a text pair where order matters, or assert the joined hypothesis through a small helper you expose).
5. **CLI:** the two new lines with the golden numbers of a small scenario, `n/a` for a chapter without truth boxes, the `approximate` count on the first line, the new JSON keys; all E2 CLI tests still pass (update only assertions that Part 4 changes: precision `None`/`n/a`, the `missed` order).
6. **Real data check for the report** (do not commit its output): `uv run omniscan eval PepperCarrotKR -c "Episode 22"` and `-c "Episode 16"` must now show a non-zero number of truth boxes; paste both printed blocks and the number of chapters (of 33) that have usable truth boxes when you run the whole series (`uv run omniscan eval PepperCarrotKR --json`, count the lines with `truth_boxes > 0`).
7. `tests/unit/test_docs.py` and the whole suite stay green.

## Out of scope
Better box geometry (real font metrics), matching English boxes to source boxes, changing the assignment rule, using the scores inside the pipeline.

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_eval_truth.py tests/unit/test_eval_metrics.py tests/unit/test_eval_score.py tests/unit/test_eval_cli.py tests/unit/test_docs.py -q
uv run pytest -q
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
`docs/reports/E2b.md`: Changes, Tests (commands + results), the real-data blocks, Deviations, Questions. **Commit early** (`E2b: WIP text boxes`) once Part 1 passes; the final commit is `E2b: eval covers text elements, char recall, page chrF`. You have 150 tool calls in total. If anything is unclear: stop, write the question under Questions, commit, and end.

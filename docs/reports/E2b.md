# E2b — `omniscan eval`: `<text>` boxes, character-weighted recall, page-level OCR chrF

## Summary

The truth parser now also turns every Inkscape `<text>` element (line `tspan`s, no box geometry)
into one approximate `TruthBox` (`approx=True`), so episodes 16–37 of PepperCarrotKR — which store
their text as `<text>` instead of `flowRoot` — are scored. Scoring adds a character-weighted recall
(`recall_chars`), a text-length-sorted `missed` list, a page-level OCR chrF (raw OCR text against
the source-language truth) and `precision = None` for chapters without usable truth boxes; the CLI
prints the new fields. All 33 chapters of the series now have usable truth boxes (previously 16).

## Changes

- `src/omniscan/eval/truth.py`: `TruthBox.approx: bool = False` (last, defaulted). `parse_svg` walks
  the tree once and handles `flowRoot` **and** `text` elements in document order. For a `text`
  element (never recursed into from a `flowRoot`, so never a second box there): lines are the child
  `tspan`s with `sodipodi:role="line"` (else the element's own text); each line falls back from its
  tspan's `x`/`y`/style to the element's (list-valued attributes → first number; unpositioned lines
  skipped); `font-size`/`text-anchor` from the tspan style, else the element's style, else
  `16.0`/`start`. Line box from anchor (`start`/`middle`/`end`) and width
  `font_size · Σ(1.0 wide / 0.5 other per char)`; the element's box is the union of its lines'
  boxes; then the exact same transform → page (sx, top crop, clamp, drop-on-zero-area) → strip
  (scale/offset, floor/ceil) pipeline as a `flowRoot` box, shared through one `emit` helper.
  `load_english_pages` picks the `<text>` lines up automatically (it reuses `parse_svg`).
- `src/omniscan/eval/score.py`: `EvalReport` gains `approx_boxes` (usable approx boxes), `recall_chars`
  (next to `recall`), `ocr_chrf_mean`/`ocr_chrf_pages` (after the CER fields). `missed` sorts by
  `len(normalize(text))` desc, then bbox area desc, then `(page, y0, x0)`. `precision` is `None` when
  the chapter has no usable truth boxes (was `0.0` when regions existed). Page OCR chrF: per page
  with usable truth boxes, reference = the page's truth texts in document order, hypothesis = the
  non-watermark OCR texts centred in the page's strip rows joined in `(y0, x0)` order (new
  `_ocr_hypothesis`/`_ocr_page_scores` helpers); empty hypothesis scores 0.0.
- `src/omniscan/cli.py`: only `_eval_block` — header line gains `, N approximate` when non-zero;
  detection line gains `chars`; OCR line gains `page chrF X (N pages)`; a chapter without usable
  truth boxes prints `  detection    n/a (no usable truth boxes)` and omits the OCR line.
- `docs/USER_GUIDE.md`: `### omniscan eval` section rewritten (new fields, approximate-box note).
- Tests: `tests/unit/test_eval_truth.py` (+16), `test_eval_score.py` (+9, 1 updated),
  `test_eval_cli.py` (+2, 2 extended).

## Tests

```
uv run pytest tests/unit/test_eval_truth.py tests/unit/test_eval_metrics.py
             tests/unit/test_eval_score.py tests/unit/test_eval_cli.py
             tests/unit/test_docs.py tests/unit/test_cli.py -q
  → 85 passed
uv run pytest -o addopts="-q"
  → 2747 passed, 6 warnings in 146.87s (the warnings are pre-existing torch/starlette
    deprecations; note `-ra -q` from pyproject swallows the summary line when piped)
uv run ruff format . && uv run ruff check .
  → all clean (4 files reformatted, then 321 unchanged)
uv run pyright
  → 0 errors, 0 warnings
```

All four Part-1 goldens reproduce exactly (`BBox(500,235,600,280)` middle/two-line/transform,
`BBox(50,65,80,84)` start, `BBox(140,25,150,32)` end, the no-tspan/empty/mixed cases), plus every
E2 golden (both strip mappings, transforms, metrics, S1 scenario, CLI block).

## Real data (manual check, not committed)

`uv run omniscan eval PepperCarrotKR -c "Episode 22"` (mixed episode: 6 `flowRoot` + 130 `<text>`
elements in its kr SVGs → 135 boxes emitted, 1 skipped as empty):

```
PepperCarrotKR/Episode 22: 12 pages, 121 truth boxes (2 ignored, 12 dropped, 121 approximate)
  detection    recall 0.322 (39/121)  chars 0.348  precision 0.800 (40/50 regions)
  OCR          CER macro 0.419  micro 0.568  (39 boxes)  page chrF 0.286 (12 pages)
  missed: p11 [0,17002,1200,17030] "David Revoy님이 Craig Maloney님의 도움으로 제작한 헤" … p11 [15,16904,1187,16940] "2017/05 - www.peppercarrot.com - 그림 및 이야" … p11 [16,19182,1181,19217] "여러분도 www.patreon.com/davidrevoy에서 후추와 당근" … p02 [361,2902,823,3030] "초록색 버튼은 참가자에게 점수를 주고, 빨간색 버튼은 점수를 깎습니다. " … p06 [729,9689,1200,9791] "아, 관객들이 이 "찬란한" 광경에 매료되지 않은 듯하군요! 이제 스피룰"
  worst CER: p11 2.76 "소프트웨어 : 리눅스 민트 18.1에서 크리타 3.1.3, 잉크스케이프 " … p06 2.00 "우우!" … p11 1.58 "대사 수정: Nicolas Artance, Valvin, Craig Ma" … p04 1.00 "이제 대회를…" … p06 1.00 "우우우!"
```

`uv run omniscan eval PepperCarrotKR -c "Episode 16"`:

```
PepperCarrotKR/Episode 16: 8 pages, 73 truth boxes (1 ignored, 0 dropped, 73 approximate)
  detection    recall 0.260 (19/73)  chars 0.299  precision 0.792 (19/24 regions)
  OCR          CER macro 0.319  micro 0.381  (19 boxes)  page chrF 0.124 (8 pages)
  missed: p06 [8,10127,1191,10157] "2016년 4월 - www.peppercarrot.com - 그림 및 이" … p02 [538,2483,1151,2605] "정말 대모들한테서 배우는 게 없어, 그 분들이 하는 일이라곤 내 집에 눌" … p06 [105,10171,1095,10195] "라이선스: 크리에이티브 커먼즈 저작자표시 4.0, 소프트웨어: 우분투에서" … p02 [199,2047,514,2191] "개인적으로, 나는 아무리 친한 친구라도 중재를 도와주지는 않아. 이런 일" … p06 [174,9557,547,9675] "일이 원하는 대로 안 풀릴 땐 목욕만큼 좋은 게 없지! 우리 류머티즘에도"
  worst CER: p04 1.00 "허허!" … p05 1.00 "괴... 괴물이다!" … p05 1.00 "출발!" … p06 0.80 "그라비타스 스피랄리스!" … p06 0.61 "David Revoy님이 Craig Maloney님의 도움으로 제작한 헤"
```

Whole series (`uv run omniscan eval PepperCarrotKR --json`, output deleted after counting):
**33 of 33 chapters** have usable truth boxes (E2 found them in 16). Totals: 1839 truth boxes,
1287 approximate, 718 detected; mean box recall 0.384, mean character recall 0.542; 247 scored
OCR-chrF pages, none without a score. Sanity cross-check: Episode 22's 135 emitted boxes = 6
`flowRoot` + 130 `<text>` elements minus 1 with no non-empty line, and 12 dropped are text elements
whose estimated union box falls completely outside its page (clamped to zero area).

## Deviations

1. **The "no usable truth boxes" condition.** The card phrases it as `truth_boxes - ignored_boxes
   == 0`, but the report's `truth_boxes` already excludes ignored boxes (E2 semantics, printed as
   "N truth boxes (M ignored…)"), so that formula would also fire for e.g. 2 usable + 2 ignored
   boxes. Implemented as `report.truth_boxes == 0`, which matches the card's message "no usable
   truth boxes" and Part 2's "precision is None when the chapter has no usable truth boxes".
2. **`approx_boxes` counts usable approx boxes** (ignored punctuation-only boxes excluded), like
   every other count on the header line; a punctuation-only `<text>` box therefore does not inflate
   the approximate count.
3. **Nested/unknown-value fallbacks** (unspecified by the card): a `text` element nested inside
   another `text` element gets its own box too (walk recurses into `text`); an anchor value other
   than `start`/`middle`/`end` (e.g. `inherit`) and an unparseable `font-size` fall back to
   `start`/the next source (element style, then 16.0). `load_english_pages` loses lines whose
   tspan/element have no position — same walk as the boxes, per the card.
4. **Dropped approximate boxes on real data.** Episode 22 has 12 `<text>` boxes outside their page
   (mostly p11's rotated/stylised SFX near the credits). They are counted in `dropped` together
   with flowRoot drops, as the card's shared-pipeline rule implies; no separate counter was added.

## Questions

1. Series-wide, `chars` (0.542 mean) lifts box `recall` (0.384 mean) by +0.16, as the E2 review
   intended. In individual episodes the lift can be small (E16: 0.260 → 0.299, E22: 0.322 → 0.348)
   because the largest undetected blocks there are the long credit/license strips, which are
   `<text>`-based and still missed. Acceptable as the new steering number, or should undetected
   long strips be matched fuzzily (out of scope here)?
2. Episode 22's p11 worst-CER entries are big credit blocks whose read is much longer than the
   truth (`cer > 2`); with the smallest-containing-box rule they swallow neighbouring regions. The
   E2 review accepted that rule; just confirming macro-CER distortion by oversize reads is still
   accepted.
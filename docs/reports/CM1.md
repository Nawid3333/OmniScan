# CM1 — chapter matching across two chapter sets

## Changes

- `src/omniscan/match/` (new package, four modules):
  - `align.py` — Needleman-Wunsch-style **global alignment** over a score matrix (`align(score, gap)`
    -> `Alignment(matches, only_a, only_b, score)`). Maximizes the total of matched pair scores minus
    a linear per-gap penalty; a pair whose score is `FORBIDDEN` (-1e9) can never be matched, which is
    how the caller forces dissimilar pairs into gaps ("never force a wrong pairing"). Row-wise numpy
    with linear gaps telescoped into a `np.maximum.accumulate` running max, so a row is O(len_b)
    vector work; traceback from the stored table with a 1e-9 predecessor tolerance (the telescoped
    form round-trips floats — 1e-9 sits far above the ~1e-12 round-off and far below any meaningful
    score gap). The same function runs at page level and at chapter level — one algorithm, no
    special cases for prologues, ads or offsets.
  - `pages.py` — page fingerprints, reusing `omniscan.filter.hashing.dhash` / `hamming` (no second
    hashing scheme): `chapter_hashes(dir)` (dHash per page, reading order), `similarity_matrix`
    (per chapter pair, [p, q]), and `similarity_blocks` — the full chapter×chapter×page×page
    similarity tensor in blocks (uint64 XOR + `np.bitwise_count` popcount, batched so one block's
    XOR intermediate stays ≤ 128 MB, `BLOCK_ELEMENTS` cap). No torch import anywhere (card
    constraint); no new dependency (numpy already provided `np.bitwise_count`).
  - `chapters.py` — orchestration + artifact: `match_chapters(dir_a, dir_b, thresholds) ->
    ChapterMapping`. Chapter reading order comes from the existing `list_chapters` (skips `_`- and
    `.`-prefixed folders, natural sort); page order from `list_images`. Folder names, chapter numbers
    and counts are never used for matching — only for reading order and artifact naming.
  - `cli.py` — `match_app` Typer sub-app, registered in `src/omniscan/cli.py` as
    `omniscan match chapters DIR_A DIR_B [--out PATH] [--force] [--page-similarity F] [--page-gap F]
    [--chapter-gap F] [--min-quality F] [--review-quality F] [--json]`.
- Scoring (decided per the card's "decide and justify", validated by the scenarios):
  - **Page level**: a page pair may align only at dHash similarity ≥ `page_similarity` (0.75);
    everything else must be a gap (`page_gap` 0.25 per skipped page). A match contributes its
    similarity; the DP maximizes matches + gaps. Gating (instead of a bare deadzone) is what keeps
    dHash's ~0.5 random-similarity baseline from stitching unrelated pages together: an unrelated
    page pair can never be taken, only skipped.
  - **Chapter-pair quality** (the confidence signal): sum of aligned page similarities divided by the
    larger page count — 1.0 when every page of both sides matched perfectly, ~0.05 for unrelated
    chapters (a stray accidental page match per ~25 cross pairs), ~0.8 with one absorbed ad page.
  - **Chapter level**: cost = 1 − quality, gated at `min_quality` (0.3; below that a pair is
    FORBIDDEN, not merely expensive), gap = `chapter_gap` (0.6). 0.6 > 0.5 means a no-signal pair
    (cost 1.0) can never beat two gaps (1.2), so a chapter with no true counterpart is left
    unmatched even at series boundaries; a weak-but-real pair (quality ≥ 0.3) only wins when the
    alternative is a cascade of displacements, which costs more. Wrong pairings are structurally
    expensive: matching chapter i to the wrong partner displaces a near-perfect pairing (≈0.9), so
    the global optimum keeps correct pairs.
  - **Review signal**: `quality < review_quality` (0.5) → `review: true`, and each matched pair
    carries `runner_up` — the strongest *other* partner it could have had on either side (None when
    nothing else was even allowed). The CLI prints the weakest matches first.
  - **Pre-filter**: before any page-level DP, a valid upper bound (each page's best partner on the
    other side, averaged over both directions) prunes pairs that provably cannot reach
    `min_quality`. Measured at 100×100 chapters this bound sits ≈ 0.65–0.73 for unrelated pairs
    (dHash's high random baseline), so it prunes little — but the full matrix is cheap: 10,000
    page-level DPs took 1.0 s, so no stronger (approximate) pre-filter was needed; the bound stays
    because it is nearly free and bounds the XOR intermediates' usefulness in extreme sizes.
- `src/omniscan/cli.py` — one new import + `app.add_typer(match_app, name="match")`. No core/
  contract change was needed (the mapping artifact subclasses `core.schemas.Artifact` from outside
  core for atomic save + `schema_version`, without editing it).
- `tests/fixtures/chapter_sets.py` (new) — synthetic chapter sets generated in code (six random
  sinusoid waves + 4–7 panel blocks + a focal disk per page; optional lettering boxes). The same
  art seed at a different size (96×144 vs 72×108), JPEG quality (92 vs 85) and lettering placement
  models "same page, raw vs official translation"; different seeds model different pages. Measured
  on 300 seeds: same page hamming ≤ 14/64 (worst similarity 0.78), different pages median 32,
  1st percentile 16 — the default `page_similarity` 0.75 sits in that gap with ~2 bits of margin on
  the close side. (Everything is seeded: a scenario outcome is deterministic, never flaky.)
- `tests/unit/test_match_align.py`, `test_match_pages.py`, `test_match_chapters.py`,
  `test_match_cli.py` (new; one file per new module + CLI), `docs/USER_GUIDE.md` — new
  `### omniscan match chapters` section (usage, flags, artifact, hand-correction).

## Tests

Commands (final state):

```
uv run --frozen pytest tests/unit/test_match_*.py -q    # 33 passed
uv run --frozen pytest -q -m "not gpu"                  # 3650 passed, 24 deselected, 1 xfailed
uv run --frozen ruff format . && uv run --frozen ruff check .   # all clean
uv run --frozen pyright                                 # 0 errors, 0 warnings
```

The card's six synthetic scenarios (each asserted on the exact mapping, `tests/unit/test_match_chapters.py`,
fixtures: raw 96×144 q92 vs translated 72×108 q85 with shifted lettering, different folder names everywhere):

1. **Clean 1:1 series** (4+4 chapters, 6 pages each, names "Chapter 001…" vs "ch_01…") — all four
   pairs matched in order, quality 0.90–0.94, no review flags, page mapping identity
   (page_a = page_b for all six pages), `b_of`/`a_of` lookups correct.
2. **Chapter only in A** (raw-only prologue in front) — prologue in `unmatched_a`; the four real
   chapters still paired correctly.
3. **Chapter only in B** (English-only ad chapter mid-series, folder "ad_page" between ch_02 and
   ch_03) — ad chapter in `unmatched_b`; the four real chapters paired correctly.
4. **Numbering offset throughout** (B has one extra early chapter, so B's numbering runs one ahead)
   — A "Chapter 001" ↔ B "ch_02" … A "Chapter 004" ↔ B "ch_05"; B's extra chapter flagged unmatched.
5. **Ad page inside a chapter pair** (extra page mid-chapter on B: 7 pages vs 6) — pair still
   matched with `pages_a 6 / pages_b 7`, page mapping (0,0),(1,1),(2,3),(3,4),(4,5),(5,6), quality
   ≈ 0.79, not flagged review.
6. **Near-duplicate-but-distinct chapters not confused** (B's ch_04 reprints chapter 3's last page
   as a recap) — chapter 3 kept its own partner (quality 0.90); chapter 4 matched its own (6 of its
   7 pages, quality > 0.7); the shared-page wrong pairing (quality ≈ 0.13) was below `min_quality`
   and therefore not even eligible.

Additional behavior tests: empty chapter folders come out unmatched on both sides; a chapter-set
directory with no chapter folders raises (CLI exit 2); all-empty chapter sets succeed with
everything unmatched; artifact save/load round-trips (pydantic, `schema_version`, strict fields) and
hand-edits revalidate; runner-up recording; block-wise similarity batching agrees with per-pair
matrices (including a zero-page chapter on one side, whose padding must not leak) and respects the
block cap; CLI: writes + summarises, `--json` (stdout is then only the artifact JSON), default
`--out chapter-match.json` in the cwd, refuse-overwrite without `--force` (file untouched), clean
exit 2 on missing/chapter-less directories, threshold flags recorded in the artifact.

### Realistic-size timing (synthetic, CPU only, warm file cache; cold first run in parentheses)

| Size | `match_chapters` (median of 3) | one side's page hashing |
|---|---|---|
| 100 chapters × 20 pages/side (4,000 pages) | **2.5 s** (cold 15.0 s) | 0.61 s for 2,000 pages ≈ 3,300 pages/s |
| 200 chapters × 20 pages/side (8,000 pages) | **8.0 s** (cold 29.9 s) | 1.23 s for 4,000 pages |

Both runs returned a perfect 1:1 mapping (0 unmatched) with quality 0.88–0.95. Cost structure:
page hashing dominates and scales linearly (~3 ms/page on these small synthetic pages; real
800–1500 px pages decode several times slower — expect roughly a minute for 200 × 20 real chapters
per side); the chapter×chapter matrix including all page-level DPs was 1.03 s for 100×100 (≈1.0 µs
per DP cell, 10,000 pairs × 20×20 cells), scaling ~quadratically in chapter count — 200×200 ≈ 4×
the matrix work. No approximate pre-filter beyond the (nearly free) validity bound was needed at
these sizes.

### Live CLI sample (offset series + A-only prologue + B-only ad chapter, `--review-quality 0.93`)

```
match: 4 matched, 1 unmatched in A, 2 unmatched in B -> ...\mapping.json
match:   A only: ...\raw/Chapter 000
match:   B only: ...\trn/ch_01
match:   B only: ...\trn/ad
match:   review: A 'Chapter 003' <-> B 'ch_04' quality 0.90
match:   review: A 'Chapter 002' <-> B 'ch_03' quality 0.92
...
```
(JSON artifact: `matched` entries with `a`, `b`, `quality`, `pages_a`, `pages_b`, `pages`
(page-index triples), `runner_up`, `review`; `unmatched_a`/`unmatched_b`; `thresholds`.)

## Deviations

- The card suggested page-count/early-signature pruning "if a naive full chapter×chapter cost matrix
  is too slow". It is not: the full matrix (all 10,000 page-level DPs) measured ~1 s at 100×100, so
  no approximate pre-filter is applied — the only prune is a provably-valid upper bound, which at
  realistic dHash baselines rarely fires (measured: unrelated-pair bound 0.65–0.73) but costs
  ~nothing. Numbers in the table above.
- Page alignment gates dissimilar pages *out* (similarity ≥ threshold required for a match) rather
  than scoring every pair with a deadzone: with dHash's ~0.5 random baseline an ungated DP would
  prefer any ≥0.4-similarity page over a gap, which is exactly the "forced wrong pairing" the card
  forbids. Gating means a heavily degraded but correct page falls out of the page map (counted as an
  unmatched page) instead of dragging a wrong pairing in — the conservative direction for a
  human-reviewed artifact.
- Empty chapter folders are kept in the alignment as permanently-unmatchable (they surface in
  `unmatched_*`) instead of being silently skipped — the honest answer for a malformed folder.
- `match chapters` refuses to overwrite an existing mapping without `--force` (the file is meant to
  be hand-edited; a silent overwrite would destroy manual corrections). Extra flag beyond the
  card's sketch, documented in the guide.

## Questions

- None blocking. One note for whoever wires this into O1c/reader later: `quality` is computed with
  the larger page count as denominator, so a pair with several absorbed ad pages tops out below 1.0
  by design — compare pairs via `review`, not via a "quality == 1.0" expectation.
- Threshold defaults (0.75 / 0.25 / 0.6 / 0.3 / 0.5) are justified by the synthetic separation
  measurements above; when the first real chapter sets go through, the knob to move first is
  `--page-similarity` (raise toward 0.8+ if different raw/official art scans accidentally align, or
  lower if genuinely shared art falls below the gate — the artifact records what a run used).
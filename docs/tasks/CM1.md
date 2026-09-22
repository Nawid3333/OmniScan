# CM1 — chapter matching across two independently-sourced chapter sets

**Owner:** GLM builder · **Branch:** `CM1` · **Worktree:** `V:\OmniScan-wt\CM1` · run with `--max-turns 300`
Read `CLAUDE.md` first. Then `src/omniscan/filter/hashing.py` (`dhash`, `similarity`, `hamming` — reuse these, do not reimplement perceptual hashing), `src/omniscan/core/paths.py` (`list_images`, `SeriesPaths`), `src/omniscan/ingest/__init__.py` (how a raw chapter folder is read today), and look at `data/raws/PepperCarrotKR` next to `data/translated-check/PepperCarrotKR` for a sense of chapter-folder conventions — but this card must **not** assume folder names line up between two sources; that is exactly the problem it solves.
This card is a goal, not a spec: you investigate, decide, measure and implement; the acceptance evidence at the bottom is what counts.

## Why
The owner is about to supply real chapter sets for training/testing the pipeline: a raw-language series and an independently-sourced English reference for the same series. Two chapter sets of "the same" series are not guaranteed to line up 1:1 — one side may have an extra prologue or an ad/insert chapter, chapter numbering may differ, folder names will not necessarily match, and one side may simply have more or fewer chapters. The reader/debugger and the qualification suite (O1c) both need a correct chapter-to-chapter mapping before any page-level comparison means anything; nothing in the codebase computes one today — page N of one source is blindly assumed to correspond to page N of the other.

## Goal
Given two directories, each holding one subfolder per chapter (each chapter a folder of page images; arbitrary and possibly differing folder names; arbitrary and possibly differing chapter counts), compute a chapter alignment: which chapter of A corresponds to which chapter of B, with a confidence signal, and an explicit flag for any chapter that has **no** good match on the other side (a prologue, a promo/ad chapter, a chapter only released in one language, etc.). Never force a wrong pairing just to fill a slot.

## Method (a strong starting point, not a fixed spec — investigate, measure, decide)
- **Page fingerprint**: page art (backgrounds, panel layout, character poses) is usually shared between a raw scan and its official translation even though the text differs — `omniscan.filter.hashing.dhash`/`similarity` (already used for promo-page matching, CPU/PIL, resolution-independent by construction) is a natural, language-independent signal for "is this the same underlying page". Reuse it; do not build a second hashing scheme without a measured reason it is needed.
- **Aligning two ordered sequences that may have insertions/deletions on either side** is a solved problem — the same idea as diffing two versions of a text file, or sequence alignment. A classic global-alignment/edit-distance algorithm over a similarity/cost matrix handles "extra chapter only in A", "extra chapter only in B", and "chapters offset by a constant" uniformly, with no special-casing. Apply the idea at two levels: once at the **page** level (within one candidate chapter pair, to score how well that pair's pages line up — tolerant of one side having an extra ad page mid-chapter) and once at the **chapter** level (across the whole series, using each pair's page-level score as its chapter-pair cost). Decide the exact scoring function, gap cost and match-confidence threshold yourself; justify the choice with the synthetic scenarios below and say so in the report. `numpy`/`scipy` are already dependencies — use them if they help; do not add a new dependency without explaining why the stdlib/numpy approach was insufficient.
- Chapters with no confident match on either side must come out clearly flagged in the output, not paired and not silently dropped — a human (the owner) is expected to eyeball the unmatched list before trusting the mapping.

## Files you may create / modify
- `src/omniscan/match/` (new package: your module layout), a CLI command under the existing `omniscan` app (e.g. `omniscan match chapters DIR_A DIR_B [--out PATH]` — your exact flags, document them in the report and `docs/USER_GUIDE.md`)
- `tests/unit/test_match_*.py` (new), `docs/USER_GUIDE.md` (a short new section), `docs/reports/CM1.md`
Do not edit `src/omniscan/core/**` (director-owned) — if a schema/contract change there would genuinely help, stop and describe it in the report instead of editing it.

## Constraints
- CPU-only; no GPU needed (page `dhash` is already CPU/PIL) — do not import torch for this.
- Must stay practical on directories with **hundreds** of chapters, not just a handful — measure and report a realistic-size timing (~100 chapters/side, ~20 pages/chapter, synthetic). If a naive full chapter×chapter cost matrix is too slow at that size, an approximate pre-filter (e.g. page-count-based candidate pruning, or a cheap early signature) is fine — measure it and justify it in the report.
- Never assume folder names, chapter numbers or chapter counts match between the two directories, and never assume the two directories have the same number of chapters.
- Output is a mapping artifact written to disk (your choice of format — TOML or JSON; human-readable and human-editable, since the owner will want to eyeball and correct it by hand), plus a CLI summary: matched count, unmatched-on-each-side count, and the lowest-confidence matches worth a second look.
- Tests use synthetic fixtures generated in code (small solid-colour/simple-shape images, in the style of `tests/fixtures/images`) — never real manga/manhwa content, per the repo's fixture rule.

## Acceptance evidence (put it in `docs/reports/CM1.md`)
1. Synthetic test fixtures you construct in code covering, each with an asserted-correct outcome:
   - a clean 1:1 series (same chapters, same order);
   - a chapter present only in A (a raw-only prologue);
   - a chapter present only in B (an English-only ad/insert);
   - a chapter-numbering offset throughout (A's chapter 3 == B's chapter 4, because B has one extra early chapter);
   - a page-level insertion inside an otherwise-matching chapter pair (one extra ad page mid-chapter on one side);
   - near-duplicate-but-distinct chapters that must not be confused with each other.
2. The realistic-size timing measurement described above.
3. `uv run --frozen pytest -q -m "not gpu"` stays green; a new test file per new module.
4. `ruff format . && ruff check .` clean; `pyright` clean.
5. `docs/USER_GUIDE.md`: a short section on `omniscan match chapters` — what it does, when to use it, how to read and hand-correct the output mapping.

## Out of scope
Wiring the mapping into the GUI reader/compare view (there is no assembled main window yet — card U3c is still open) or into the qualification suite (O1c) automatically — this card only produces the mapping artifact and the CLI to make and inspect it. Fuzzy chapter-title/number parsing from folder names (the algorithm must work from page content alone; folder names may *optionally* be used as a secondary hint, never as the primary signal). Any downloading or site-specific acquisition — out of scope entirely; this card only ever operates on two directories that already exist on disk.

## Commands to run before finishing
```bash
uv run --frozen pytest tests/unit/test_match_*.py -q
uv run --frozen pytest -q -m "not gpu"
uv run --frozen ruff format . && uv run --frozen ruff check .
uv run --frozen pyright
```

## Report
`docs/reports/CM1.md`: Changes, Tests (commands + results, each synthetic scenario and its result, the timing measurement), Deviations, Questions. Commit early (`CM1: WIP alignment`), final commit `CM1: chapter matching across two chapter sets`. You have 300 tool calls. If anything is unclear, stop, write the question in the report, commit, and end.

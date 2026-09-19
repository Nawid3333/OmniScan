# C5a — Glossary post-check and candidate agreement (pure functions for the judge)

**Owner:** GLM builder · **Branch:** `C5a` · **Worktree:** `V:\OmniScan-wt\C5a` (created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. Then read `src/omniscan/glossary/match.py` (`find_terms`, `Match`, `term_present`),
`src/omniscan/translate/prompts.py` (`source_text`, `translatable`), and the `Region`, `GlossaryEntry` classes in
`src/omniscan/core/schemas.py`.

## Goal
Two small, deterministic, dependency-free building blocks for the judge (C5b, written later by the director):
1. **Post-check:** given a Korean source line and an English line, report which *locked* glossary terms occur in the source
   but whose English target is missing from the English line.
2. **Agreement:** decide whether several candidate translations of one line are "the same" (so the judge can skip lines
   where all models agree and spend tokens only where they disagree).
No LLM, no I/O, no CLI, no GPU. Everything is unit-tested on CPU.

## Files you may create / modify
- `src/omniscan/translate/postcheck.py`, `src/omniscan/translate/agree.py` (create)
- `tests/unit/test_translate_postcheck.py`, `tests/unit/test_translate_agree.py` (create)
- `docs/reports/C5a.md` (create)
Nothing else. Do not modify `src/omniscan/core/**`, `glossary/**` or the existing `translate/*` files (import from them).

## Part 1 — `translate/postcheck.py`
```python
@dataclass(frozen=True, slots=True)
class TermViolation:
    entry_id: int
    source: str      # entry.source (never the alias that matched)
    expected: str    # entry.target

def normalize_for_check(text: str) -> str
def check_locked_terms(source: str, target: str, entries: Sequence[GlossaryEntry]) -> list[TermViolation]
def check_regions(regions: Sequence[Region], texts: Mapping[str, str],
                  entries: Sequence[GlossaryEntry]) -> dict[str, list[TermViolation]]
```
Definitions (exact):
- `normalize_for_check(text)` = `unicodedata.normalize("NFKC", text).casefold()`. Nothing else (no whitespace or punctuation change).
- `check_locked_terms(source, target, entries)`:
  1. `locked = [e for e in entries if e.status == "locked"]` (proposed and rejected entries are never checked). If empty → `[]`.
  2. `matches = find_terms(source, locked)` (raises `ValueError` for a locked entry without an id — let it propagate).
  3. Distinct entry ids of the matches, in order of first occurrence in `source`. For each such entry `e`:
     a violation `TermViolation(e.id, e.source, e.target)` unless `normalize_for_check(e.target)` occurs (substring) in
     `normalize_for_check(target)`. An entry whose `target` is `""` therefore never violates.
  4. Return the violations in that order.
- `check_regions(regions, texts, entries)`: iterate `translatable(regions)` (from `translate.prompts`; that already sorts and drops
  watermark/empty regions). For each region take `texts.get(region.id)`; **a region without an entry in `texts` is skipped**
  (a missing translation is not a glossary violation). Compute `check_locked_terms(source_text(region), text, entries)`.
  Return a dict `{region.id: violations}` containing only regions with at least one violation, in `translatable` order.

## Part 2 — `translate/agree.py`
```python
def normalize_line(text: str) -> str
def similarity(a: str, b: str) -> float
def agreement(texts: Sequence[str]) -> float
def candidates_agree(texts: Sequence[str], threshold: float) -> bool
```
Definitions (exact):
- `normalize_line(text)`: `unicodedata.normalize("NFKC", text).casefold()`; then replace every character whose
  `unicodedata.category(c)[0]` is `"P"` or `"S"` (punctuation, symbols) by a space; then `" ".join(result.split())`.
- `similarity(a, b)`: `na, nb = sorted((normalize_line(a), normalize_line(b)))` (sorting makes the result independent of
  argument order); if both are empty → `1.0`; else `difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio()`.
- `agreement(texts)`: the **minimum** pairwise `similarity` over all pairs of `texts`; `1.0` when `len(texts) < 2`.
- `candidates_agree(texts, threshold)`: `agreement(texts) >= threshold`. `threshold` outside `[0.0, 1.0]` → `ValueError`.
  (The judge will use `0.9` by default; do not add a config key — that is C5b.)

## Acceptance tests (CPU only)
**Post-check** (entries: `GlossaryEntry(id=1, source="성진", target="Seong-jin", type="person", status="locked")`,
`GlossaryEntry(id=2, source="게이트", target="Gate", type="place", status="locked")`):
1. Source `"성진이가 게이트에 들어간 지"`, target `"Seong-jin went into the gate"` → `[]` (case-insensitive: `gate` satisfies `Gate`).
2. Same source, target `"He went into the portal"` → `[TermViolation(1, "성진", "Seong-jin"), TermViolation(2, "게이트", "Gate")]`
   (order of first occurrence in the source).
3. Only the second term missing → only `TermViolation(2, ...)`.
4. A `proposed` and a `rejected` entry with hits in the source are ignored; with no locked entries → `[]`.
5. An alias hit (`GlossaryEntry(id=3, source="성진", target="Seong-jin", aliases=["진우"], status="locked")`, source
   `"진우가 왔다"`, target `"He came"`) → `[TermViolation(3, "성진", "Seong-jin")]` (`source` is the entry's source, not the alias).
6. The same entry occurring twice in the source yields one violation. A source with no term hit → `[]` even if the target is empty.
7. `SEONG-JIN!` satisfies `Seong-jin`; the full-width target `"Ｇａｔｅ"` satisfies `Gate` (NFKC).
8. A locked entry without an id (`id=None`) and a hit in the source → `ValueError` (from `find_terms`); the same entry when the
   source has no hit → `[]` (`find_terms` only checks the id on a match).
9. `check_regions`: regions `r0001` (text `"성진이가 왔다"`), `r0002` (text `"게이트"`), `r0003` (kind `watermark`, text `"성진"`),
   `r0004` (text `"안녕"`); `texts = {"r0001": "He came", "r0002": "The gate", "r0003": "x"}` (no entry for `r0004`) →
   `{"r0001": [TermViolation(1, "성진", "Seong-jin")]}` (r0002 satisfied, r0003 watermark skipped, r0004 has no text and no term).
   Result order follows `(slice_index, reading_order, natural id)` — include a case where `r10` sorts after `r2`.
**Agreement:**
10. `normalize_line` goldens: `"“Seong-jin… Wait!”"` → `"seong jin wait"`; `"Hello,   World!!"` → `"hello world"`; `"  "` → `""`;
    `"Ａｂｃ ＆ ｄ"` → `"abc d"`; `"It's   fine — really?"` → `"it s fine really"`; `"…"` → `""`; Korean text is returned unchanged.
11. `similarity` goldens (assert with `pytest.approx(rel=1e-4)`):
    `("I can't believe you're here.", "I can't believe you are here.")` → 0.9818; `("Get out!", "get out")` → 1.0;
    `("Run!", "Watch out behind you!")` → 0.1739; `("Seong-jin, wait!", "Sung-jin, wait!")` → 0.8889; `("", "")` → 1.0;
    `("", "Hi")` → 0.0; `("...", "!!")` → 1.0 (both empty after normalisation). `similarity(a, b) == similarity(b, a)` for all of these.
12. `agreement`: `[]` and one text → `1.0`; `["Get out!", "get out", "Run!"]` → 0.2 (the minimum pair); two identical texts → 1.0.
13. `candidates_agree(["Get out!", "get out", "Get out"], 0.9)` is `True`; adding `"Run!"` makes it `False`; the boundary is
    inclusive (`threshold == agreement(texts)` → `True`); thresholds `-0.1` and `1.1` raise `ValueError`; `0.0` and `1.0` are accepted.
14. A hypothesis-free property test over 200 seeded random ASCII strings: `0.0 <= similarity <= 1.0`, symmetry, and
    `similarity(x, x) == 1.0`.

## Out of scope
The judge, prompts, config keys, CLI, story memory, glossary extraction, repair rounds (all C5b/C5c), the web UI, any I/O.

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_translate_postcheck.py tests/unit/test_translate_agree.py -q
uv run pytest -q
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
Write `docs/reports/C5a.md` (Changes, Tests, Deviations, Questions) and commit `C5a: glossary post-check and candidate agreement`.
If anything above is unclear: stop, write the question under Questions, commit what you have, and end.

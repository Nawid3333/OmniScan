# C5a — Glossary post-check and candidate agreement (pure functions for the judge)

## Changes
- `src/omniscan/translate/postcheck.py` (new): `TermViolation` frozen dataclass (`entry_id`, `source` = the
  entry's own source, never the alias; `expected` = entry.target); `normalize_for_check` = NFKC + casefold
  exactly, no whitespace/punctuation change; `check_locked_terms` (locked entries only — proposed/rejected
  never checked; hits and particles come from `glossary.match.find_terms`, so a locked entry without an id
  raises its `ValueError` on a hit only; distinct entry ids in first-occurrence order; a violation unless
  `normalize_for_check(entry.target)` is a substring of `normalize_for_check(target)` — an empty target never
  violates because `""` is a substring of anything, no special case needed); `check_regions` iterates
  `prompts.translatable` (watermarks/empties dropped, reading order), skips regions with no entry in `texts`
  (a missing translation is not a violation), and returns only regions with ≥1 violation, in `translatable`
  order.
- `src/omniscan/translate/agree.py` (new): `normalize_line` (NFKC + casefold, then every char whose Unicode
  category starts with `P`/`S` becomes a space, then whitespace collapse); `similarity` (sorted normalised
  pair makes it order-independent; both empty → 1.0; else `difflib.SequenceMatcher(None, na, nb,
  autojunk=False).ratio()`); `agreement` = minimum pairwise similarity, 1.0 for fewer than two texts;
  `candidates_agree` = `agreement >= threshold` with `ValueError` for thresholds outside [0.0, 1.0] (checked
  before any similarity work). No config key added (that is C5b).
- `tests/unit/test_translate_postcheck.py` (new, 11 tests) and `tests/unit/test_translate_agree.py` (new,
  22 parametrised cases) — cover all 14 card points, including alias-hit reports `entry.source`, duplicate
  occurrence → one violation, `SEONG-JIN!`/`Ｇａｔｅ` NFKC satisfaction, id-less locked entry raising only on a
  hit, `r10` sorting after `r2` in `check_regions` output order, all similarity/agreement goldens with
  `pytest.approx(rel=1e-4)`, inclusive threshold boundary, and a hypothesis-free property test over 200 seeded
  random ASCII strings (range, symmetry, `similarity(x, x) == 1.0`).
- No changes to `src/omniscan/core/**`, `glossary/**`, or existing `translate/*` files (imported from them).

## Tests
```
uv run pytest tests/unit/test_translate_postcheck.py tests/unit/test_translate_agree.py -q
#   33 passed
uv run pytest -q          # 1972 passed, 3 warnings (pre-existing numpy warning in codec test)
uv run ruff check .       # clean
uv run ruff format --check .   # my files clean; only pre-existing docs/tasks/C3.md flagged (see Deviations)
uv run pyright            # 0 errors, 0 warnings (whole project)
```

## Deviations
- `uv run ruff format .` (ruff 0.16.8 now formats Python code blocks inside markdown files) rewrote code
  blocks in `docs/tasks/C3.md` — a file outside this card's list. I reverted that file to HEAD by hand, so
  this card's commit contains only the four card files + this report. Consequence: `ruff format --check .`
  reports `docs/tasks/C3.md` as "would be reformatted" — that is pre-existing repo state, and every builder
  running the mandated command will trip over it again. Repo-level fix needed (see Questions).

## Questions
1. ruff 0.16's markdown code-block formatting vs `docs/tasks/*.md`: should `pyproject.toml` `[tool.ruff]`
   `extend-exclude = ["docs"]` (or per-file ignores) keep hand-written task cards stable, or does the
   director want docs reformatted once repo-wide and then kept ruff-clean?
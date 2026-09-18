# <ID> — <Title>

**Owner:** GLM builder · **Branch:** `<ID>` · **Worktree:** `V:\OmniScan-wt\<ID>` (sibling of the repo, created by `scripts/omni_builder.py`)
Read `CLAUDE.md` first. It overrides nothing below but applies everywhere.

## Goal
One paragraph: what exists after this card that did not exist before.

## Files you may create / modify
- `path/one.py` (create)
- `tests/unit/test_one.py` (create)
Anything not listed is off-limits (especially `src/omniscan/core/**`).

## Interfaces (exact — do not rename or change signatures)
```python
def example(arg: int, *, flag: bool = False) -> list[str]:
    """One-line docstring."""
```

## Definitions (no interpretation needed)
- Term A means exactly …
- Edge case X → behaviour Y.

## Acceptance tests (must exist and pass)
1. `test_...`: given …, expect …
2. …

## Out of scope
- …

## Commands to run before finishing
```bash
uv run pytest tests/unit/test_one.py
uv run ruff format . && uv run ruff check .
uv run pyright
```

## Report
Write `docs/reports/<ID>.md` (sections: Changes, Tests, Deviations, Questions) and commit `<ID>: <summary>`.
If anything above is unclear: stop, write the question under Questions, commit what you have, and end.

Before this is merged, do a rigorous self-review pass. Be critical — assume there are bugs.

1. Re-run and paste exact output: `uv run pytest -q`, `uv run pytest -m gpu -q`, `uv run ruff format --check .`,
   `uv run ruff check .`, `uv run pyright src tests`.
2. Open `docs/tasks/B3.md` and list, one line each, which test function covers which numbered acceptance test
   (1–15). If any numbered test has no corresponding test function, add it now.
3. `git fetch origin main && git rebase origin/main` (resolve any conflicts — the base has moved since you
   started). Then `git diff origin/main..HEAD --stat` and confirm every changed file is in the card's "Files you
   may create / modify" list. If something outside that list changed (e.g. a stray reformat of an unrelated
   file), revert just that file's changes.
4. Re-read your own `convert_to_jpeg` and `stack_layout` implementations against the card's exact definitions
   (rounding rule, EXIF orientation handling, alpha-flatten-to-white, idempotent re-conversion). Fix anything
   that doesn't match precisely.
5. Fix every issue you find directly and commit each fix with a clear message.
6. Append a `## Review` section to `docs/reports/B3.md` ending with exactly one line:
   `VERDICT: READY_TO_MERGE` or `VERDICT: NOT_READY: <short reason>`.

Do not push. Do not touch `main`. Do not modify `src/omniscan/core/**`.

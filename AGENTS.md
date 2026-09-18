# AGENTS.md — OmniScan

Instructions for any AI coding agent or assistant (Claude Code, Codex, Copilot, Cursor, Gemini, …) working in this repository.

**Read `docs/HANDOFF.md` first.** It explains the project, the reading order (`CLAUDE.md`, `docs/CHECKPOINT.md`,
`docs/OPEN_QUESTIONS.md`, `docs/PLAN.md`, `docs/DECISIONS.md`), how the owner likes to work, and the lessons that already cost time.

Non-negotiables (details in `CLAUDE.md`):
- Windows 11 is the primary environment; code and tests must also run on Linux and macOS. Python 3.14 via `uv` (`uv run --frozen ...`).
- Never hard-code `cuda:0` — use `omniscan.gpu.device.resolve_device(cfg.gpu.device)`.
- Never install or import `paddlepaddle`/`paddleocr`; Paddle *models* run through PyTorch/`transformers`.
- Never commit raw manga/manhwa, samples, secrets or `data/`; tests use synthetic fixtures.
- Definition of done: `uv run --frozen pytest -q`, `uv run --frozen ruff format . && uv run --frozen ruff check .`, `uv run --frozen pyright` all clean.
- If a spec is ambiguous, stop and write the question down (in the card's report or `docs/OPEN_QUESTIONS.md`) instead of guessing.

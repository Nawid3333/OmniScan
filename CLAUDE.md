# OmniScan — rules for every agent working in this repo

OmniScan turns raw Korean/Chinese/Japanese manhwa/manga chapters into English releases, GPU end-to-end on AMD ROCm (WSL2).
Full plan: `docs/PLAN.md`. Architecture/contracts: `docs/ARCHITECTURE.md` (once written).

## Environment
- Runs inside WSL2 Ubuntu 26.04, repo at `~/projects/omniscan`. GPU: RX 9070 XT (gfx1201, 16 GB), ROCm 10.0.0.
- Python **3.14**, managed by **uv**. PyTorch comes from AMD's ROCm 10 index (see `pyproject.toml`); never `pip install torch` from PyPI.
- Ollama runs on Windows, reachable at `http://localhost:11434` (mirrored networking).
- **Never** install or import `paddlepaddle` / `paddleocr`: Paddle has no ROCm support. Paddle *models* are used through HF `transformers` (PyTorch).

## Commands
```bash
uv sync                      # install/refresh deps
uv run pytest                # all tests (GPU tests are marked `gpu`)
uv run pytest -m "not gpu"   # CPU-only tests
uv run ruff format . && uv run ruff check --fix .
uv run pyright
uv run omniscan --help
```

## Ownership (do not cross these lines)
- `src/omniscan/core/**` (schemas, stage protocol, manifest, config contracts) is **owned by Claude**. Builder agents must NOT edit it. If a contract is missing or wrong, stop and describe the needed change in `REPORT.md`.
- Each task card (`docs/tasks/<ID>.md`) lists the files you may create/modify. Stay inside that list.

## Definition of done (every task card)
1. All acceptance tests listed in the card exist and pass: `uv run pytest`.
2. `uv run ruff format . && uv run ruff check .` clean, `uv run pyright` has no new errors.
3. Public functions have type hints and a one-line docstring. No dead code, no commented-out code, no TODOs without a card ID.
4. `REPORT.md` at the worktree root: what changed (files), how it was tested (commands + results), deviations from the card, open questions.
5. Commit on the card branch with message `<ID>: <summary>`. Never push, never touch `main`.

## Hard rules
- **If the spec is ambiguous or seems wrong: STOP, write the question in `REPORT.md`, and end.** Do not guess.
- GPU code: tensors stay on the GPU between steps; no `.cpu()` / `.numpy()` inside hot loops; never loop in Python over image rows or pixels.
- Never write intermediate images to disk unless the card says so (JSON/`.npz` artifacts only).
- Never read or print secrets (`~/.config/omniscan/secrets.env`, API keys).
- Never download raw manga into the repo; tests use synthetic fixtures generated in code (`tests/fixtures/`).
- Keep changes minimal and focused on the card. Match surrounding style.

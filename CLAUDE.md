# OmniScan — rules for every agent working in this repo

OmniScan turns raw Korean/Chinese/Japanese manhwa/manga chapters into English releases, GPU end-to-end (developed on Windows 11 + AMD ROCm; the app is meant to run on any OS/GPU that PyTorch supports).
Full plan: `docs/PLAN.md`. Architecture/contracts: `docs/ARCHITECTURE.md`.
**If you are the director (an interactive session continuing the project) read `docs/HANDOFF.md` first**, then `docs/CHECKPOINT.md`.
If you are a builder running a task card, this file plus the card are your contract.

## Environment
- Windows 11 native; repo at `V:\OmniScan` (builder worktrees in `V:\OmniScan-wt\<ID>`). GPU: RX 9070 XT (gfx1201, 16 GB), ROCm 10.0.0.
  Your Bash tool is Git Bash (bash syntax works, paths like `V:/OmniScan/...`). Never rely on Linux-only behaviour (symlinks, `chmod`,
  `fcntl`, `/dev/...`, `/mnt/c`); code and tests must run on Windows, Linux and macOS. Files use LF line endings (`.gitattributes`).
- Python **3.14**, managed by **uv**. PyTorch comes from AMD's ROCm 10 index (see `pyproject.toml`); never `pip install torch` from PyPI.
- GPU choice: use `omniscan.gpu.device.resolve_device(cfg.gpu.device)`; never hard-code `cuda:0` (on this PC `cuda:0` is the integrated GPU and crashes).
- Ollama runs natively on Windows at `http://localhost:11434`.
- **Never** install or import `paddlepaddle` / `paddleocr`: Paddle has no ROCm support. Paddle *models* are used through HF `transformers` (PyTorch).

## Commands
```bash
uv sync --all-extras         # install/refresh deps, including the `gui` extra (PySide6) tests/gui/** needs
uv run pytest                # all tests (GPU tests are marked `gpu`)
uv run pytest -m "not gpu"   # CPU-only tests
uv run ruff format . && uv run ruff check --fix .
uv run pyright
uv run omniscan --help
```

## Ownership (do not cross these lines)
- `src/omniscan/core/**` (schemas, stage protocol, manifest, config contracts) is **owned by Claude**. Builder agents must NOT edit it. If a contract is missing or wrong, stop and describe the needed change in your report.
- Each task card (`docs/tasks/<ID>.md`) lists the files you may create/modify. Stay inside that list.

## Definition of done (every task card)
1. All acceptance tests listed in the card exist and pass: `uv run pytest`.
2. `uv run ruff format . && uv run ruff check .` clean, `uv run pyright` has no new errors.
3. Public functions have type hints and a one-line docstring. No dead code, no commented-out code, no TODOs without a card ID.
4. `docs/reports/<ID>.md`: what changed (files), how it was tested (commands + results), deviations from the card, open questions.
5. Commit on the card branch with message `<ID>: <summary>`. Never push, never touch `main`.

## Hard rules
- **If the spec is ambiguous or seems wrong: STOP, write the question in `docs/reports/<ID>.md`, commit, and end.** Do not guess.
- GPU code: tensors stay on the GPU between steps; no `.cpu()` / `.numpy()` inside hot loops; never loop in Python over image rows or pixels.
- Never write intermediate images to disk unless the card says so (JSON/`.npz` artifacts only).
- Never read or print secrets (`~/.config/omniscan/secrets.env`, API keys).
- Never download raw manga into the repo; tests use synthetic fixtures generated in code (`tests/fixtures/`).
- Keep changes minimal and focused on the card. Match surrounding style.

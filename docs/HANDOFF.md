# Handoff — start here (new AI session or new person)

This file exists so a fresh session — Claude Code, another AI, or a human — can pick the project up without the
original conversation. It says what OmniScan is, what to read, how the owner likes to work, and what already cost us
time. Written 2026-09-19 after roughly two days of building. Keep it current: when something important is learned,
add it here (working agreement, lessons) or in `docs/DECISIONS.md` (design choices and why).

## What OmniScan is
A tool that takes raw Korean (later Chinese/Japanese) manhwa/manga chapters and produces English releases that look
official: slice the long strip, find and read the text, translate with several models plus a judge (glossary-aware,
consistent across hundreds of chapters), remove the original text, letter the English, export. **GPU end-to-end** on the
owner's AMD RX 9070 XT. **One machine, one GPU** (decided 2026-09-24): the project targets Windows 11 + AMD ROCm
(gfx1201) only — the earlier "any OS/any GPU" goal (old plan milestone M13, cpu/cuda/mps portability extras) was
dropped on the owner's explicit instruction; a professional debug/review area is still wanted, just not the
cross-platform packaging that used to come with it.
Repo: `V:\OmniScan` (Windows 11 native) · GitHub `Nawid3333/OmniScan` (private).

## Read in this order
1. **This file.**
2. `CLAUDE.md` — the rules every coding agent in this repo follows (also the builders' contract).
3. `docs/CHECKPOINT.md` — current state, evidence gathered, the half-built card (C3), ordered next steps, the work loop.
3a. `docs/NEXT.md` — the ordered work queue: which builder cards to launch, which to write, what the director does in parallel.
4. `docs/OPEN_QUESTIONS.md` — questions the owner has not answered yet, each with the default we use meanwhile.
5. `docs/PLAN.md` — the master plan (milestones, cards, architecture, backlog). `docs/ARCHITECTURE.md` — contracts.
6. `docs/DECISIONS.md` — why the important choices were made, with the evidence.
7. `docs/benchmarks/*.md` — measurements behind the decisions (codec, translation probe, Windows-native).
8. `docs/USER_GUIDE.md` — what the CLI and web viewer do today. `docs/tasks/` and `docs/reports/` — every builder card and its report.
9. `docs/memory/` — snapshot of the Claude auto-memory (owner preferences, environment facts) for tools that do not load it.

## Working agreement with the owner (learned, not guessed)
- **Roles.** The owner decides direction and supplies accounts/data. The *director* (Claude in an interactive session) writes exact
  task cards, reviews and merges, and does the genuinely hard parts (codec, detection/OCR, translation logic, inpaint, typeset).
  *Builders* are Claude Code CLI sessions running on Ollama Cloud models (`scripts/omni_builder.py`) that implement well-specified cards.
- **Builder policy.** `glm-5.3-flash:cloud` for essentially every card (the owner wants it used *more* than `glm-5.3:cloud`, which costs
  more tokens); `--model glm` only for genuinely hard cards or after flash failed. At most **3 builders at once** (raised from 2 by the owner on 2026-09-19; use `OMNI_SLOTS=2` when a live Ollama check is planned, because the Pro plan allows only 3 concurrent requests). **Never** deepseek/kimi
  as builders — one deepseek run burned ~5M tokens / $26.77 and hit the Ollama Cloud session limit (hard 429, rolling ~5 h window).
- **Keep the builders busy and notice completions yourself.** Launch a builder as a tracked background tool call and review the moment it
  finishes; the owner complained about having to remind the director. Never poll with sleeps.
- **Save the director's tokens:** delegate volume work and routine review; reserve one large Claude verification pass for the end.
- **GPU-first and evidence-based.** No CPU fallback without benchmark evidence; show measured numbers, then decide. The owner accepts
  "measure first" (e.g. the codec decision).
- **Questions:** write unanswered ones to `docs/OPEN_QUESTIONS.md` with a default, and ask 2–3 at natural pauses; record answers in its
  Decisions table. The owner answers in bursts and sometimes sends several short messages in a row (or one by accident — "ignore this").
- **Destructive actions:** ask unless the owner explicitly asked (they explicitly asked for the WSL distro and a scratch folder to be deleted,
  and both were, after an inventory).
- **Legal/safety scope:** no acquisition from paid DRM platforms (Naver Webtoon, Kakao, Lezhin, Ridibooks, Bomtoon, Kuaikan); never commit
  raws; tests use synthetic images; CC-licensed comics (Pepper&Carrot) are fine as extra test material; avoid GPL/AGPL model dependencies.
- **Language:** the owner writes English, sometimes German (Windows UI is German, so tool output such as robocopy is localised).

## The director loop in one paragraph
Write `docs/tasks/<ID>.md` (exact interfaces, numbered acceptance tests, a file allowlist, stop-and-ask rule) → `ruff format` and commit →
`uv run python scripts/omni_builder.py run <ID>` in the background → on completion `git rebase main` in the worktree (every card edits the
README status table: keep both rows), read `docs/reports/<ID>.md` (the "Questions" section has caught real spec bugs every time), run pytest
+ ruff + pyright (+ web checks), fix problems yourself, mutation-check test-only cards, `git merge --ff-only`, push, remove worktree and
branch. Details and gotchas: `docs/CHECKPOINT.md`.

## Looking things up with `ask`
- Don't read big or generated files (logs, JSON artifacts, long reports, huge modules) into your own context — let GLM do it:
  `uv run python scripts/omni_builder.py ask "QUESTION" [--file PATH ...] [--wait]` (also `--model`, `--max-turns`, `--cwd`).
- It starts one short read-only `claude -p` session (`glm-5.3-flash:cloud`, 25 turns) in this repo and prints **only the answer** on stdout; progress and logs go to stderr / `.builder/logs/ask-*.jsonl`.
- Exit codes: `0` answer on stdout · `1` session failed or no answer (see the log) · `3` all builders busy and no `--wait` (retry with `--wait` or read the file yourself).
- It may read anything in the repo (data/, logs, model JSON on purpose) but cannot write, run Bash, or fetch the web; `.env`/secrets/`~/.config/omniscan` are deny-listed (`.builder/ask-settings.json`).
- It uses a builder slot for the duration of the call; pass `--wait` to queue instead of failing with exit 3.

## Lessons that already cost time
- **Vague specs make builders burn tokens** and hide contradictions. Numbered acceptance tests and exact definitions fixed that. Spec bugs the
  builders caught: dHash of solid colours is identical (B6), an unreachable check (B14), the `forced_cut` boundary meaning (B5), bare page
  numbers read as chapters (B21), typer cannot register `list[Literal[...]]` (B22).
- **Passing tests can be vacuous.** Breaking the code on purpose showed the slicer suite did not pin `max_drift` at all (B27) — add mutation
  checks for test-only cards.
- **Ollama specifics:** thinking models need `think:false` (gemma4 once produced empty output after 353 s); cloud models ignore the `format`
  JSON schema and wrap JSON in prose/fences; `*-cloud` models are reachable through the *local* daemon; translategemma ignores the glossary
  unless locked terms are substituted into the source first. See `docs/benchmarks/translation-probe.md`.
- **GPU on this PC:** the integrated GPU is `cuda:0` on Windows and crashes on the first kernel; the RX 9070 XT is `cuda:1`. Use
  `omniscan.gpu.device.resolve_device`; never hard-code `cuda:0`; never call bare `torch.cuda.synchronize()` without selecting the device
  (it timed nothing and printed 45 000 TFLOPS once).
- **rocJPEG hardware decode is not usable** (WSL had no VCN access; not present on Windows) → the CPU `turbo` codec is the baseline.
- **The acquisition subsystem was removed** (card RM1, 2026-09-22): raws are user-supplied (`omniscan import` / the GUI
  import page); the extract.pics client, `omniscan acquire` and the Cloudflare Worker relay no longer exist. Historical
  context: extract.pics' docs were a JavaScript app the fetch tool could not read, and GitHub could not receive its
  webhook — the reasons the relay existed and the client stayed unfinished.
- **Windows:** symlinks need Developer Mode; `uv.lock` needed a `triton` Linux-only override; npm blocks postinstall scripts unless allowed;
  pass long prompts to CLIs on stdin; the PowerShell tool resets its working directory on every call.
- **Shared `.venv` between worktrees:** every `uv run` re-points the editable `omniscan` install at the worktree it ran in (`.venv/Lib/site-packages/_editable_impl_omniscan.pth`). Builders started by `omni_builder.py` now get `PYTHONPATH=<worktree>/src` so concurrent builders cannot import each other's code (since 2026-09-19); when you run `python`/`pytest` by hand from another worktree, use `uv run` there or set `PYTHONPATH` yourself.
- **Never `uv sync --extra <backend>` in a worktree (since P1, 2026-09-23):** a worktree's `.venv` is a junction to the main checkout's, so `uv sync --extra cpu`/`cuda`/`mps` in ANY worktree replaces the shared, real ROCm torch install with a different backend for the whole project — silently, for every worktree and the director's own session, including a live `omniscan serve`. P2's builder actually triggered this (caught only because a running `omniscan serve` happened to lock `omniscan.exe` mid-swap) — it had already downloaded the cpu wheels and was about to install them over the working ROCm build. Use `uv run --no-sync` for a worktree's own suite/lint/type-check runs instead (only the worktree's source differs; the shared environment is already correct) — the same fix already established for the ordinary "worktree's venv exe is locked" problem now also *prevents* a real regression, not just an inconvenience.
- **GPU-only code paths hide bugs from CPU tests — look at real output.** `TurboCodec.decode_into` reused one pinned staging buffer with async copies on CUDA, so every page of a large chapter held the last page's pixels (fixed 2026-09-19, `8c3a32f`; a large-page GPU regression test guards it). It survived ~2 000 CPU tests and three cards; it showed up only when the first exported image was compared with the raw pages, and as "repeated" OCR sentences. After any change that touches decoding or compositing, run the chain on `scripts/make_korean_chapter.py` output and *view* the result; stage versions were bumped so stale artifacts recompute. Also: run `.venv\Scripts\python.exe` with `PYTHONPATH=<worktree>\src` (the shared editable install points at whichever worktree ran `uv` last — I once got "not implemented yet" for real commands).
- **Builders have a hard turn limit (150) — mechanical loops belong in a script.** The first Q2 run did mutation testing by hand (Edit → pytest → restore, 2–3 tool calls per mutant), used all 150 turns, recorded nothing and committed nothing; a run that ends without a commit produces nothing. `scripts/mutate.py` now applies a whole list of mutants (`MUTANTS = [(file, old, new, label)]`), restores files itself and prints KILLED/SURVIVED per mutant. When you write a card, count the tool calls it needs, tell the builder to commit early (`<ID>: WIP …`), and move anything repetitive into a script the director provides. Mutant lists are kept in `tests/mutants/` (see its README) so a refactor can be re-checked in minutes.
- **Ollama Cloud (Pro plan):** 3 concurrent requests, rolling ~5 h session limit, hard 429 when exhausted — the client fails fast on it. As of 2026-09-23, a rate limit during `omniscan run`'s translate/judge stages no longer needs a manual re-run: `TranslationProfile.fallback` switches to a local model for the rest of that pass, and the affected chapter is retried against the cloud automatically on the next run.
- **`git worktree remove --force` can leave an empty directory behind**, which fools `omni_builder.py`'s `if not worktree.is_dir()` check into skipping `git worktree add` — the builder then silently runs in a git-less folder. Always `rm -rf` the leftover directory before retrying.
- **Never run multiple heavy processes at once** (pytest suites, GPU pipelines, real-hardware benchmarks) — besides the driver-crash risk the GPU lock already guards against, CPU contention from redundant concurrent runs can make a healthy process *look* hung for hours, and any timing/benchmark data collected during the overlap is invalid.
- **A local LLM's own runtime settings can silently sabotage measurements.** Ollama's app-level context-length slider (not anything in this repo) applies to *every* model it serves; at 256K it pushed 11 of translategemma:12b's 49 layers onto the CPU with no error, just a ~2.6x slowdown, and it also let two local models stay resident in VRAM at once (measured: 34s → 87s for one profile). `OllamaClient` now sends its own `num_ctx` per request and `VramManager.evict_ollama(keep=...)` enforces one local model at a time — but the lesson generalises: benchmark the actual served config, not just this repo's code, before trusting a timing number.
- **A wrong performance hypothesis is normal — reverse it when real hardware disagrees, three times if needed.** PaddleOCR-VL's slowness was first blamed on lacking batching (O1f); batching turned out to be 3-32x *slower*, confirmed on real hardware three separate ways, before the actual cause (`use_cache: false` inherited from the model's training config) was found. Don't defend an earlier design assumption once measurement contradicts it.

## Environment (verified)
Windows 11, Ryzen 5 7600X, 64 GB, RX 9070 XT 16 GB (+ iGPU), Python 3.14, uv, Node 24, git + `gh` (logged in as Nawid3333), Ollama 0.34.2 on
Windows (local models: translategemma:12b, gemma4:12b, …; cloud models via the same daemon). Paths: repo `V:\OmniScan`, builder worktrees
`V:\OmniScan-wt\<ID>`, data `V:\OmniScan\data\` (gitignored), machine config `C:\Users\limex\.config\omniscan\config.toml`, old Paddle patch
`V:\OmniScan-wsl-backup\`, full transcript of the founding conversation `V:\OmniScan-archive\`.
The old WSL distro was deleted on 2026-09-19; nothing depends on it.

## Starting a new session
Open `V:\OmniScan` in VS Code and start Claude Code (or any other AI) there. Paste:

> You are continuing the OmniScan project as its director. Read `docs/HANDOFF.md`, then `docs/CHECKPOINT.md` and `docs/OPEN_QUESTIONS.md`.
> Confirm the state with `git status`, `git log -5` and `uv run --frozen pytest -q`, then continue with the first item under
> "Next, in priority order" unless I say otherwise. Ask me 2–3 open questions when it fits.

Claude Code also auto-loads `CLAUDE.md` and — because its project memory for this folder was seeded — the notes in
`C:\Users\limex\.claude\projects\V--OmniScan\memory\`. Other tools read `AGENTS.md`, which points back here.

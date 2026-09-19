---
name: omniscan-project
description: "OmniScan = user's manhwa/manga translator project (Windows-native at V:\\OmniScan, ROCm PyTorch, GLM builder via Ollama); where everything lives and how to resume"
metadata:
  node_type: memory
  type: project
  originSessionId: 18b96bfa-62d0-4f52-b6c0-2ca8b649a9c3
  modified: 2026-09-18T22:53:09.108Z
---

OmniScan: raw KO/ZH/JA manhwa/manga → industry-quality English (slicer, promo filter, OCR, multi-candidate translation + judge, inpaint, typeset, SFX later). Started 2026-09-18.

- Plan: `docs/PLAN.md` in the repo (mirror: C:\Users\limex\.claude\plans\okay-let-s-make-a-kind-pinwheel.md)
- Repo: github.com/Nawid3333/OmniScan (private). **Local: `V:\OmniScan` (Windows 11 native)** since 2026-09-19; the old WSL distro was
  deleted at the user's request. Builder worktrees: `V:\OmniScan-wt\<ID>`. Data (sample library/work/output, gitignored): `V:\OmniScan\data\`;
  machine config: `C:\Users\limex\.config\omniscan\config.toml`. Small backup of an old Paddle-build patch: `V:\OmniScan-wsl-backup\` (deletable).
- Split: GLM builder (Claude Code CLI on Ollama, glm-5.3-flash:cloud) builds B-cards via `uv run python scripts/omni_builder.py`; Claude builds C-cards (codec, OCR, translation logic, inpaint, typeset, SFX) and reviews everything. See [[omniscan-env]], [[user-gpu-first]].

**Why:** user wants Claude to delegate volume work to the cheaper Ollama agent and only do heavy parts + review.
**How to apply:** write unambiguous task cards (GLM wastes tokens on vague specs), review diffs before merge. Builder policy is **flash-first, up to 3 agents at once (owner, 2026-09-19)** — see [[user-gpu-first]].

**RESUME HERE:** read `docs/CHECKPOINT.md` in the repo first (current state, evidence, the half-built detection card = branch `C3`, next
steps, the card→review→merge loop and its Windows gotchas). Then `docs/OPEN_QUESTIONS.md` (see [[feedback-open-questions]]) — ask 2–3 at natural pauses.
Real comic test pages (CC BY 4.0 Pepper&Carrot) are in `V:\OmniScan\data\library\PepperCarrot\`, never in the repo. The user still owes 1–2 real
Korean raw chapters. The user wants completions noticed proactively (launch builders as tracked background tool calls, then review at once).

**Product goal (M13, deferred):** one packaged desktop app for all systems (Windows/macOS/Linux), any GPU (CUDA/ROCm/MPS/CPU auto-detect), with a
professional debug/review area (browser views embedded via QWebEngineView), built after the pipeline works end-to-end. No Linux container/WSL inside
the app (measured: everything, incl. PaddleOCR models, runs natively on Windows). `gpu.device = "auto"` (never hard-code `cuda:0`; the iGPU is cuda:0 on this PC).

**Backlog (in docs/PLAN.md, not memory — don't duplicate here):** ad-hoc paste-URL acquisition (B19), export formats (B22, done), model version pinning,
cloud LLM cost/usage tracking, per-chapter QA report, chapter watch, editor edit history, duplicate chapter detection. Check `docs/PLAN.md`'s "Backlog".

---
name: user-gpu-first
description: User insists on GPU end-to-end processing and evidence-based choices; runs sudo commands themselves
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 18b96bfa-62d0-4f52-b6c0-2ca8b649a9c3
  modified: 2026-09-18T22:53:42.848Z
---

User wants the pipeline GPU end-to-end (keep data in VRAM, use the full VRAM, minimal disk IO) and rejected a plain CPU libjpeg-turbo fallback. They accepted a benchmark-decided approach (hybrid GPU codec vs turbo) when shown live evidence that rocJPEG can't run in WSL.

**Why:** they believe CPU paths hammer the CPU and slow the pipeline; they trust measured numbers.
**How to apply:** propose GPU paths first; when a GPU path is impossible, show live test evidence and offer a benchmark rather than silently falling back. For sudo steps, give copy-paste blocks — the user runs them. See [[omniscan-project]].

**Builder agent policy (latest, 2026-09-18 evening; supersedes the earlier "single agent, flash only" rule):** default to **glm-5.3-flash:cloud for essentially every card**; the user explicitly said flash should be used *more* than the full **glm-5.3:cloud** because glm-5.3 is more token-costly. Use glm-5.3 sparingly — only for a card that is genuinely hard/large or that flash already failed on. Up to 3 builders may run concurrently since 2026-09-19 (earlier 2; `OMNI_SLOTS`, default 3, in `scripts/omni_builder.py`; the user asked for faster progress). **Avoid deepseek-v4-pro / kimi**: a deepseek run burned ~5M input tokens / $26.77 over 71 turns and hit the account-wide Ollama Cloud session limit (hard 429, rolling ~5h window) without finishing. Flash cards so far (B2/B6/B9/B14/B15/B22) all landed cleanly, so it is proven sufficient for well-specified cards. The user also wants Claude to spend its own tokens sparingly and delegate heavy work (including routine review/verification passes, not just building) to the Ollama agent — Claude's role is writing precise task cards, merge decisions, and the genuinely hard/risky pieces (GPU codec, core algorithms) — with a big Claude pass reserved for a final whole-repo verification once the milestones are built out.

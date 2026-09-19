# Checkpoint — 2026-09-19, evening (Windows native, `V:\OmniScan`)

**New session? Read `docs/HANDOFF.md` first** (reading order, working agreement with the owner, lessons), then this file, then `docs/NEXT.md` (the ordered work queue with builder cards).
`docs/PLAN.md` is the master plan (see its "Plan revisions" section); this file is the current-state snapshot. Repo: **`V:\OmniScan`**, remote `Nawid3333/OmniScan` on GitHub (`main` is pushed after every merge).
Local data (sample library, work artifacts, outputs) lives in `V:\OmniScan\data\` (gitignored); machine config in `C:\Users\limex\.config\omniscan\config.toml` (paths only; the GPU is chosen by `gpu.device = "auto"`).

## State in one paragraph
Every stage of the pipeline exists or is in review. Merged and reviewed: ingest, slice, promo filter, **detect (C3)**, translate candidate runs, **judge (C5a/C5b)**, **inpaint v1 flat fill (C6a)**, **typeset layout engine and stage (C7a/C7b)**, glossary, watermark regions,
job queue, import, CBZ/PDF packaging, the web debug tool (5 views), synthetic Korean test pages (X1). About 2 500 tests, ruff and pyright clean, all passing on Windows in ~1 minute. **Running builders** (2026-09-19 17:00): **C4a** (OCR stage), **C6b** (LaMa stage), **C7c** (glyph renderer + export).
Card **R1** (`omniscan run`, three passes, queue integration) is written and starts when those three merge; after it: an end-to-end golden test on synthetic Korean pages, then tuning on real Korean raws (question A1 still open — everything real so far is English Pepper&Carrot).

## What works today (CLI)
`doctor`, `import`, `ingest`, `slice`, **`detect`**, `filter run|restore`, `glossary list|export|import`, `watermark add|list|remove`, `translate` (candidate runs; needs `ocr.json`), **`judge`** (`final.json`; needs `ocr.json` + runs), **`inpaint`** (flat fill; needs `ocr.json`), **`typeset`** (`layout.json`; needs `ocr.json`, `final.json`, `inpaint.json`),
`pack` (CBZ/PDF), `serve` (+ `npm run dev` in `webui/`), `queue add|list|run|pause|resume|cancel|retry|clear`. Stubs (exit 2): `acquire ocr export run reference` (`ocr` and `export` are being built). Web views: Slicer, OCR (reads `ocr.json`), Translation, Reader, Filtered.
Run everything with `uv run ...` from `V:\OmniScan`. Nothing writes `ocr.json` yet until C4a merges, so there is no end-to-end run yet.

## Evidence gathered (all reproducible; details in `docs/benchmarks/`)
- **Windows native works** (`windows-native.md`): AMD's `win_amd64` ROCm wheels run on the RX 9070 XT; GPU selection via `omniscan.gpu.device.resolve_device` (the iGPU is `cuda:0` and crashes, the 9070 XT is `cuda:1`; call `torch.cuda.set_device` before MIOpen work).
- **fp16 is unreliable → everything runs fp32** (addendum in `windows-native.md`, `docs/DECISIONS.md`): RT-DETR fp16 works only at batch 8 and fails at batch 1–3 (MIOpen); the PP-OCR detector and LaMa fail in fp16 too. fp32 speeds: RT-DETR 31–65 tiles/s, PP-OCR det 25 ms per 960×704, LaMa 30 ms per 512² (steady state).
- **OCR probe** (`ocr-probe.md`): PP-OCRv5 server text-line detector + Korean recognition through `transformers`: 7/7 lines, recognition CER 0.048 on the detector's own boxes; DB post-processing on the CPU with OpenCV takes 3.4 ms per page (12× faster than a GPU version) → `opencv-python-headless` is a dependency.
- **LaMa probe** (`lama-probe.md`): TorchScript `big-lama.pt` runs on the GPU in fp32; steady state 30 ms per 512² crop, but every new input shape costs a 10–25 s warm-up → one fixed 512² window, warmed once; text on painted art removed convincingly, output identical outside the mask.
- **Detection on real art** (C3 report addendum): on English hand-lettered Pepper&Carrot pages (out of the model's domain) it finds titles and bubbles, misses some, and puts low-confidence boxes on objects; thresholds are untuned until real Korean pages exist.
- **Codec:** rocJPEG cannot run here; CPU `turbo` ≈ 141–148 Mp/s (`codec.md`); the hybrid GPU codec (C2) stays an open decision (question F1).
- **Translation probe** (`translation-probe.md`): gemma4 needs `think:false`; cloud models ignore Ollama's JSON `format`; translategemma ignores the glossary unless locked terms are pre-substituted.
- **Builders on Windows:** `scripts/omni_builder.py` (default 3 concurrent slots since 2026-09-19, `PYTHONPATH` pinned to the worktree) runs `claude` against the local Ollama with `glm-5.3-flash:cloud`.

## Next, in priority order
The ordered queue, card states and review tiers are in **`docs/NEXT.md`**. In short: review/merge C4a, C6b, C7c as they finish → launch **R1** → E2E golden test → live judge check on the real Ollama (needs free request slots) → C5c (story memory + glossary proposals) → web views for the new stages →
real Korean raws (A1) for tuning detection thresholds, OCR and typeset sizes → hybrid codec / portability (P1/P2) / desktop shell (M13) later.

## Waiting on you (the user)
The full list (about 50 questions with my defaults) is **[docs/OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)** — ask 2–3 of them at natural pauses and record answers there. The most pressing ones:
- **A1: 1–2 real Korean raw chapters** (`uv run omniscan import "<folder>" --series Sample --chapter "Chapter 1"`) — blocks tuning; everything real so far was English.
- **D5** (judge only on disagreements — built that way), **E2/E3** (fonts and the lettering look you want), **F9** (glyph rasterisation on the CPU).

## How to work the loop (for the next session)
- **Builders:** `uv run python scripts/omni_builder.py run <ID>` from `V:\OmniScan` (default `glm-5.3-flash:cloud`; `--model glm` only for genuinely hard cards or after flash failed; `resume <ID> <feedback-file> [--max-turns N]`; `smoke`). Launch it as a background tool call so the completion notification arrives; at most 3 at once
  (`OMNI_SLOTS=2` when a live Ollama check is planned). Worktrees go to `V:\OmniScan-wt\<ID>` with a junction to the shared `.venv`. A stuck builder can be stopped (kill the `omni_builder.py run <ID>` python process tree), its session id copied from the log into `.builder/logs/<ID>.session`, and resumed with a feedback file.
- **Card → review → merge:** write `docs/tasks/<ID>.md` (exact interfaces, numbered acceptance tests, file allowlist, stop-and-ask; compute golden values from real code first), `uv run ruff format` it and commit; after the builder finishes: in the worktree `git rebase main` (README/USER_GUIDE/stub-list conflicts are routine: keep both sides' additions,
  remove every implemented command from the stub lists), read `docs/reports/<ID>.md` (Deviations + Questions), run ruff/pyright/pytest **with `PYTHONPATH=<worktree>/src` and `.venv/Scripts/python.exe -m …`** (avoids the shared editable install), mutation-check the logic (a scripted list of ~15–25 mutants; kill survivors with tests), look at real output where it is visual,
  add a "Review addendum (director)" to the report, `git merge --ff-only <ID>` in `V:\OmniScan`, run the suite on `main`, push, remove the worktree (delete the leftover `.venv` junction with `[System.IO.Directory]::Delete(path, $false)` — never recursive) and the branch.
- **Gotchas:** builders' `ruff format .` may reformat older un-formatted cards (`git checkout -- docs/tasks/<file>` before rebasing); a builder whose worktree was cut before a dependency merged may re-implement it (dedupe on review); the PowerShell tool resets its cwd on every call; use `uv run --frozen`; files with quotes are easier to write with the editor tools; symlinks need Developer Mode.
- **Data:** never commit raws or samples (`data/`, `library/`, `samples/` are gitignored); tests use synthetic fixtures (`tests/fixtures/korean_pages.py`).

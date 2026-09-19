# Next session — work plan and builder queue

Written 2026-09-19 (evening) for the director session that continues the project. It replaces the "Next, in priority order"
list in `docs/CHECKPOINT.md`. Read `docs/HANDOFF.md` and `CHECKPOINT.md` first; this file says *what to do and in which order*,
with as much of the volume work as possible handed to the `glm-5.3-flash:cloud` builders.

## The idea in one paragraph
Nothing runs end to end yet because five stages are missing (detect half-built, OCR, judge, inpaint, typeset, export). The
work on each is mostly *typing and testing to a fixed spec* — exactly what the flash builders do well (B29 landed a
six-module feature from one card) — while the director's real value is the spec, the contracts, and looking at real output.
So the rule for the next days: **the director writes cards and checks live results; builders write the code and the tests.**
The bottleneck is card writing, so two cards are already written (C3, C5a) and the first thing to do is start them, then
write the next cards while they run. Goal of the week: a thin **walking skeleton** (every stage in a simple v1 form, runnable
with `omniscan run`) on synthetic Korean pages, then quality upgrades once real Korean raws arrive (question A1).

## Definition of done for the next session
1. C3 (detection) and C5a (post-check/agreement) merged; detections looked at on real pages in the OCR view.
2. The Korean synthetic-page fixture card (X1) and the OCR card (C4a) written; ideally both running.
3. Render-pass contracts (`inpaint.json`, `patches.npz`, export inputs) written into `ARCHITECTURE.md`/`core/schemas.py`.
4. `docs/CHECKPOINT.md` updated; at most 3 builders running at the end, each with a card that is ready to review next time.

## Start of session (director, ~15 min)
```powershell
Set-Location V:\OmniScan
git status; git log -5 --oneline; git fetch
uv run --frozen pytest -q                       # expect ~2000 passed in ~20 s
uv run --frozen omniscan doctor                 # GPU = RX 9070 XT (cuda:1), Ollama up
uv run python scripts/omni_builder.py smoke     # 3-turn connectivity test of glm-5.3-flash:cloud
git add docs; git commit -m "docs: task cards C3, C5a and the next-session plan"   # the loop commits cards before a builder starts
```
Continue the existing C3 branch (the launcher only creates a *new* worktree from `main`; C3 already has the WIP commit):
```powershell
git worktree add V:\OmniScan-wt\C3 -b C3 origin/C3
git -C V:\OmniScan-wt\C3 rebase main            # expect a small conflict in core/config.py (maybe config/default.toml) next to gpu.device="auto": keep both
cmd /c mklink /J V:\OmniScan-wt\C3\.venv V:\OmniScan\.venv
```
Then start the two ready builders as **tracked background calls** (never sleep-poll; the completion notification arrives by itself):
`uv run python scripts/omni_builder.py run C3` and `uv run python scripts/omni_builder.py run C5a` (both `flash`, default).

## Builder queue
Model is `flash` unless stated. At most 3 running. Review tier: **T1** = run checks + one mutation check; **T2** = read the diff of
the logic + the report's Questions; **T3** = also a live GPU/real-page check by the director.

| Wave | ID | What | Depends on | Tier | State (updated 2026-09-19 evening) |
|---|---|---|---|---|---|
| 1 | **C3** | Detection stage, `vision` VRAM group, `omniscan detect` | — | T3 | ✔ **merged** (live check on Pepper&Carrot done; thresholds untuned until real Korean pages) |
| 1 | **C5a** | Locked-term post-check + candidate agreement | — | T1 | ✔ merged |
| 1 | **X1** | Synthetic Korean pages with ground truth (`tests/fixtures/korean_pages.py`) | fonts | T2 | ✔ merged |
| 1 | **C7a** | Typeset layout engine (`typeset/fit.py`) | — | T1 | ✔ merged (25/25 mutants killed) |
| 1b | **C5b** | Judge + `omniscan judge` | C5a | T2 | ✔ merged (23/23 mutants killed); **live check on real Ollama still to do** |
| 2 | **C4a** | OCR stage: PP-OCRv5 line detection + Korean recognition → `ocr.json` | C3, X1 | T3 | **running** (`docs/tasks/C4a.md`) |
| 2 | **C6a** | Inpaint v1: masks + flat fill → `inpaint.json` + `patches.npz` | X1, contracts | T2 | ✔ merged (visual check: bubbles identical to the text-free page) |
| 2 | **C7b** | Typeset stage → `layout.json` | C7a, contracts | T2 | ✔ merged (16/16 mutants killed) |
| 3 | **C7c** | Glyph renderer (`typeset/render.py`) + GPU compositing + `export` stage → `output/<Series>/<Chapter>/0001.jpg…` + `export.json` | C6a, C7b | T3 | **running** (`docs/tasks/C7c.md`) |
| 3 | **C6b** | LaMa stage (`inpaint_lama`): fixed 512² windows, fp32, weights fetched with sha256 check → `patches_lama.npz` | C6a, `docs/benchmarks/lama-probe.md` | T3 | **running** (`docs/tasks/C6b.md`) |
| 3 | **R1** | `omniscan run`: stage order, VRAM group sequencing (vision → local Ollama → torch), resumable, queue `STAGE_TABLE` gets every stage | C4a, C6a, C7b, C7c | T2 (director reviews the sequencing) | **card written** (`docs/tasks/R1.md`); launch after C4a, C6b, C7c merge |
| 3 | **E2E** | Golden test: synthetic Korean chapter → full pipeline with a fake LLM client → output images exist, text removed, English inside boxes | R1 | T1 | — |
| 3 | **C5c** | Story memory (per-chapter summaries in `series.db`) + glossary proposals from OCR text | C5b patterns | T2 | write card |
| 3 | **B11'** | Web views: raw \| mask \| clean slider and a layout overlay (Svelte/TS) | C6a, C7b | T2 | — |
| later | **C4b/c** | PaddleOCR-VL second opinion; zh/ja model packs (config + tests) | C4a, real data | T3 | — |
| later | **C7d** | Polygon fitting, font roles (dialogue/shout/thought/narration), colour matching | C7c | T3 | — |

**Fillers** (well specified, no dependencies — use them whenever a slot would otherwise idle; each needs a short card):
- **B31** duplicate/near-duplicate chapter detection (reuse the promo filter's dHash).
- **B32** `doctor` checks: every model named in `translation_profiles.toml` exists in Ollama; Hugging Face models are pinned to a
  `revision` and their files verified (plan backlog "model management").
- **B33** `acquire` plumbing everything except the extract.pics request/response shapes: `sources.toml`, resumable downloader,
  non-chapter image filter, DRM-domain warning (question A4 still open).
- **Q1…** *Mutation-review cards* (new card type): a builder copies a merged module to scratch, applies N listed mutations
  (flip a comparison, off-by-one, drop a guard), runs the module's tests and reports which mutants survived, then adds
  tests that kill them. Started with the modules the director has not mutation-checked (slicer bands/cuts, filter, glossary matcher,
  queue). This moves the "break it on purpose" review from the director to a builder.
- **P1** (portability, needs care → use `--model glm` or the director): make the torch backend selectable in `pyproject.toml`
  (ROCm / CUDA / CPU / MPS via `uv` extras + `conflicts`), so a CPU-only Linux/macOS CI job can `uv sync` — today
  `torch[device-gfx1201]==…+rocm10.0.0` is pinned unconditionally, so nothing except this PC can install the project.
  Then **P2**: GitHub Actions matrix (Windows + Linux + macOS, CPU tests only). Not before the skeleton runs, but decide the shape early
  because it changes `pyproject.toml`/`uv.lock`.

## Director's own work (in parallel with the builders)
| # | Task | Why the director |
|---|---|---|
| D1 | Fetch OFL fonts into `fonts/` with their licence files: a Korean text font (Nanum Gothic or Noto Sans KR) for tests and a few comic-lettering fonts for typeset (e.g. Bangers, Comic Neue, Patrick Hand). Commit. | Licence check; needed by X1/C7a |
| D2 ✔ | **OCR probe on the GPU** — done, `docs/benchmarks/ocr-probe.md` (detector fp32-only, OpenCV post-processing on the CPU, CER 0.048). Transformers 5.17 has `pp_ocrv5_server_det`, `pp_ocrv5_mobile_det`, `pp_ocrv6_small_det`/`medium_det` and matching rec models, and the `PaddlePaddle/PP-OCRv5_*_det_safetensors` repos exist; the server-det image processor has `post_process_object_detection`. Load det + `korean_PP-OCRv5_mobile_rec`, run on a bubble crop rendered with the Korean font, look at boxes/scores/timing. Result decides the C4a card (which det model, crop padding, line grouping). | Unknown territory → measure first |
| D3 | **Render-pass contracts**: what `inpaint` writes (`inpaint.json`: per region method / fill colour / patch box; `patches.npz`: cleaned crops), what `typeset` writes (`layout.json` exists), what `export` reads; add `InpaintArtifact` to `core/schemas.py` and document in `ARCHITECTURE.md` (glyph rasterisation is CPU/FreeType into small patches, compositing on the GPU — see question F9). | `core/**` is director-only; every later card depends on it |
| D4 | LaMa probe: can a PyTorch/TorchScript LaMa (FFT ops) run on this ROCm-Windows build, and how fast on a 512² crop? | Decides C6b and whether flat-fill is enough for v1 |
| D5 | Live look at C3 output: copy a chapter's `regions.json` to `ocr.json`, open the OCR view, tune `detect.threshold`/`tile_px` on the Pepper&Carrot pages. | Judgement on real pixels |
| D6 | Review and merge whatever finished (loop in `CHECKPOINT.md`: rebase, read the report's *Questions*, pytest+ruff+pyright, mutation check, `merge --ff-only`, push, remove worktree). | Keeps builders unblocked |

**Order of card writing** (so a slot is never idle): C3 ✔ → C5a ✔ → *(start both)* → D1 → X1 → C5b → D2 → C4a → D3 → C6a, C7a.

## Questions to ask the owner tomorrow (2–3, at the first pause)
1. **A1 — real Korean raws:** 1–2 chapters for `omniscan import "<folder>" --series Sample --chapter "Chapter 1"`. Blocks tuning of
   detection/OCR on real Korean; until then everything is synthetic or English.
2. **D5 — judge default:** run the cloud judge on every line, or only where candidates disagree / a locked term is violated? The
   card C5b assumes *only on disagreements* (cheaper, and Ollama Cloud has a rolling ~5 h limit).
3. **E2/E3 — the look:** free OFL fonts are fine to start? Can you show 2–3 pages of an official English release whose lettering
   you want to match? Steers C7a/C7c (font sizes, stroke, alignment).
(Also open: **F9**, glyph rasterisation on the CPU — see `docs/OPEN_QUESTIONS.md`.) Record answers in the Decisions table there.

## Rules for the builders this week (from `HANDOFF.md`, restated)
- `flash` for everything; escalate a card to `--model glm` only after flash failed twice on it; never deepseek/kimi except `kimi`
  for a TypeScript view where flash struggles and only after asking. ≤ 3 running (owner allowed 3 on 2026-09-19), and **use OMNI_SLOTS=2 or stop starting builders when a live
  cloud-translation check is planned** (they share the Ollama Cloud limit); a hard 429 means wait, then `resume`.
- Cards follow `docs/tasks/_TEMPLATE.md` and the B29/C3 level of detail: exact signatures, golden values computed from real code
  (run them before writing the card), numbered acceptance tests, file allowlist, stop-and-ask. A vague card costs more tokens than it saves.
- Test-only or logic cards get a mutation check before merge (or a **Q** card afterwards).
- Nothing touches `src/omniscan/core/**` except the director.

## Risks to watch
- **Flash on GPU code** (C3's model wrapper): the card pins the exact preprocessing and includes a preprocessing-equivalence test
  against the Hugging Face processor; if that test fails the builder must stop and ask instead of loosening it.
- **Rebase conflicts** on `core/config.py` (C3 vs the `gpu.device = "auto"` change) — small, resolve by keeping both.
- **OCR detection** may not work as hoped on tall bubble crops (D2 will tell); fallback is grouping the comic detector's own
  `text_bubble` boxes and running recognition on them directly.
- **LaMa on ROCm-Windows** is unproven (D4); v1 ships with flat fill only and flags the rest.
- **No real Korean data yet** — synthetic fonts render cleaner than scans; every tuning number is provisional until A1 is answered.
- **Ollama Cloud limits** — builders and cloud translation share them; check `ollama ps`/usage before a big run.

## Not tomorrow
The hybrid GPU codec (C2), the Qt desktop shell (M13), relay deployment, reference mode (C12), SFX replacement, LAN mode. They stay
in `docs/PLAN.md`; none of them blocks the walking skeleton.

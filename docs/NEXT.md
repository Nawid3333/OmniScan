# Next session — work plan and builder queue

Rewritten 2026-09-19 (late evening), after the walking skeleton was finished. Read `docs/HANDOFF.md` and `docs/CHECKPOINT.md` first; this file says
*what to do next and in which order*, with as much of the volume work as possible handed to the `glm-5.3-flash:cloud` builders.

## Where we are
The whole pipeline exists and runs: `omniscan run SERIES` takes raw chapters through ingest → slice → detect → OCR → translate (3 models) → judge → inpaint → LaMa → typeset → export in three passes.
It is protected by ~2 650 tests plus an **end-to-end golden GPU test** (`tests/unit/test_e2e_synthetic.py`: real detector/OCR/LaMa models on synthetic Korean pages, fake LLM; it fails when pages are mixed, reordered or coloured wrongly — proven by putting the old codec race back).
What is **not** done: sound effects (the detector finds none), everything tuned on real Korean raws (question A1 — all real data so far is English Pepper&Carrot), story memory, web views for the render stages, portability (the torch pin is ROCm-only), `acquire`.
So the next phase is **quality on real pages and the missing features**, not plumbing. The rule stays: **the director writes cards and looks at real output; builders write the code and the tests.**

## Start of session (director, ~10 min)
```powershell
Set-Location V:\OmniScan
git status; git log -5 --oneline; git worktree list      # expect: clean main, no worktrees left
uv run --frozen pytest -q                                 # ~2 650 passed, ~2–3 min including the GPU tests (one is the ~50 s golden test)
uv run --frozen omniscan doctor                           # GPU = RX 9070 XT (cuda:1), Ollama up
uv run python scripts/omni_builder.py smoke               # 3-turn connectivity test of glm-5.3-flash:cloud
```
Then: ask the owner the questions at the bottom (A1 first), write the next card while a builder runs the previous one. Launch builders as **tracked background calls** (the completion notification arrives by itself; never sleep-poll).

## Ordered queue
Model is `flash` unless stated. At most 3 builders running (`OMNI_SLOTS=2` when a live Ollama check is planned). Review tiers: **T1** = checks + mutation run; **T2** = read the logic diff + the report's Questions; **T3** = also look at real output.
Every builder has a hard limit of **150 tool calls**: a card must fit (count the calls), must say "commit early (`<ID>: WIP …`)", and anything repetitive belongs in a script the director provides (see `scripts/mutate.py`).

| # | ID | What | Tier | State |
|---|---|---|---|---|
| 1 | **A1/T1** | **Tune on real Korean pages.** Legal data is in place: Pepper&Carrot Korean/English/text-free for 5 episodes (`data/raws/PepperCarrotKR`, `data/translated-check/PepperCarrotKR/…`; more episodes are one download loop away, see CHECKPOINT). Build an eval script (OCR CER and detection recall against the English original text is not available per box — use the Korean page text via the `en`↔`kr` pairing by reading order, or hand-label one episode), then tune `detect.threshold`, `ocr.drop_conf`, typeset sizes; fix the misses listed in CHECKPOINT. Commercial-style long strips (owner's own legally obtained chapters via `omniscan import`) remain the real target | T3 (director) | started: first real run done; **E2** (`omniscan eval`: detection recall/precision, OCR CER and translation chrF against the SVG text layers) and **X2** (scan artefacts for the synthetic pages) are running with glm-5.3-flash; OCR of all 33 episodes is precomputed in `data/work/` |
| 2 | **J1** | Live judge check on the real Ollama with more data than KoreanDemo (needs free request slots: run with `OMNI_SLOTS=2`); check `docs/OPEN_QUESTIONS.md` D5 in practice (judge only on disagreement) | director | not started |
| 3 | **C5c** | Story memory (per-chapter summaries in `series.db`, fed to translate/judge prompts) + glossary proposals (recurring proper nouns found in OCR text → `proposed` entries) | T2 | card to write (patterns: `translate/judge*.py`, `glossary/store.py`) |
| 4 | **B11'** | Web views for the render stages: raw \| mask \| clean slider (`inpaint.json`/`patches.npz`) and a layout overlay (`layout.json`); Svelte/TS + API endpoints | T2 | card to write (patterns: existing 5 views in `webui/`, `web/` API) |
| 5 | **M10** | Sound effects: detection of SFX (`sfx` kind exists in schemas and the fixtures), a policy (translate as text, keep, or replace with lettered English) — design first, the director decides | T3 | design needed; blocked on real raws for detection |
| 6 | **Q3…** | More mutation reviews (card type Q, use `scripts/mutate.py`): queue store/worker, CBZ/PDF packaging, web API, translate profiles/prompts, watermark regions, typeset render/fit/plan (director mutation-checked C7*, a builder pass finds more) | T1 | cards are copy-edits of `docs/tasks/Q1.md`/`Q2.md` |
| 7 | **B32** | `doctor` checks: every model named in `translation_profiles.toml` exists in Ollama; Hugging Face models pinned to a `revision` with files verified | T1 | filler, card to write |
| 8 | **B31** | Duplicate / near-duplicate chapter detection (reuse the promo filter's dHash) | T1 | filler, card to write |
| 9 | **B33** | `acquire` plumbing except the extract.pics request/response shapes: `sources.toml`, resumable downloader, non-chapter image filter, DRM-domain warning (question A4 open; the extract.pics docs are a JS app the fetch tool cannot read) | T2 | filler, card to write |
| 10 | **C4b/c** | Second-opinion OCR (PaddleOCR-VL); zh/ja model packs (config + tests) | T3 | after real data |
| 11 | **C7d** | Polygon fitting to the bubble outline, font roles (dialogue/shout/thought/narration), colour matching | T3 | after real data |
| 12 | **P1/P2** | Portability: torch backend selectable in `pyproject.toml` (ROCm / CUDA / CPU / MPS via `uv` extras + `conflicts`), then a GitHub Actions matrix (Windows + Linux + macOS, CPU tests). Today nothing except this PC can `uv sync`. Needs care: use `--model glm` or do it as director | T3 | decide the shape soon, it changes `pyproject.toml`/`uv.lock` |
| 13 | **C2** | Hybrid GPU JPEG codec decision (question F1); CPU `turbo` is ~145 Mp/s and not the bottleneck so far | director | open |
| later | **M13** | Desktop shell (Qt), relay deployment, reference mode (C12), LAN mode | — | `docs/PLAN.md` |

Merged and reviewed so far (all with mutation checks; details in `docs/reports/<ID>.md`): C3 detect, C4a OCR, C5a post-check/agreement, C5b judge, C6a flat inpaint, C6b LaMa, C7a fit, C7b typeset stage, C7c render + export, X1 Korean fixtures, R1 `omniscan run`,
Q1 (129 slicer/ingest mutants, 23 test gaps closed), Q2 (115 filter/glossary/importer mutants, 19 gaps closed), E1 golden test.

## Card checklist (what made cards succeed)
- Follow `docs/tasks/_TEMPLATE.md` and the C3/R1/E1 level of detail: exact signatures, **golden values computed from real code first**, numbered acceptance tests, file allowlist, stop-and-ask, commands to run, report sections.
- Tell the builder which existing files to imitate (a similar stage, a similar test) — it copies patterns well and invents badly.
- GPU code: pin the exact preprocessing and add an equivalence test; flash gets GPU details wrong (C3 needed a rescue after the fp16 detour).
- After the builder: `git rebase main` (README / USER_GUIDE / stub-list conflicts are routine: keep both sides), ruff/pyright/pytest with `PYTHONPATH=<worktree>\src`, then `scripts/mutate.py` with a director-written list, look at real output, write the review addendum, `merge --ff-only`, push, remove worktree + branch.
- Builders that hit the turn limit have usually finished the work but not committed: check `git status` in the worktree before deciding anything.

## Risks to watch
- **No real Korean data yet** — synthetic fonts render cleaner than scans; every tuning number (detect threshold, `ocr.drop_conf`, typeset sizes, the golden-test thresholds) is provisional until A1 is answered.
- **Detector on real art**: on Pepper&Carrot it misses some text and puts low-confidence boxes on objects; on Korean pages it has never been run.
- **Ollama Cloud limits** — builders and cloud translation share a rolling ~5 h limit (3 concurrent requests); a hard 429 fails fast, wait then `resume`.
- **fp16 is unreliable on this ROCm-Windows stack** (MIOpen); everything runs fp32. Re-test when torch/ROCm updates.
- **Editable-install flip** between worktrees: use `PYTHONPATH=<tree>\src` and `.venv\Scripts\python.exe -m …`.
- **A killed `scripts/mutate.py` run** leaves a mutant in place; the next `check`/`run` restores it automatically (or `git checkout -- src`).

## Questions to ask the owner (2–3, at the first pause; record answers in `docs/OPEN_QUESTIONS.md`)
1. **A1 — real Korean raws:** 1–2 chapters for `omniscan import "<folder>" --series Sample --chapter "Chapter 1"`. Blocks tuning of detection/OCR/typesetting on real pages; nothing else is as valuable right now.
2. **E2/E3 — the look:** free OFL fonts (Comic Neue, Bangers, Nanum, …) fine to start? Can you show 2–3 pages of an official English release whose lettering you want to match? Steers C7d.
3. **D5 — judge default:** built as "only where candidates disagree or a locked term is violated" (cheaper; Ollama Cloud has a rolling limit). OK?
(Also open: **F9**, glyph rasterisation on the CPU — small FreeType patches, compositing on the GPU; and about 45 more with my defaults in `docs/OPEN_QUESTIONS.md`.)

## Not next
The hybrid GPU codec (C2), the Qt desktop shell (M13), relay deployment, reference mode (C12), SFX replacement, LAN mode stay in `docs/PLAN.md`; none of them blocks quality work on real pages.

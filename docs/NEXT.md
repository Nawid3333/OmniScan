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
| 1 | **A1/T1** | **Tune on real Korean pages.** Legal data is in place: Pepper&Carrot Korean/English/text-free for 5 episodes (`data/raws/PepperCarrotKR`, `data/translated-check/PepperCarrotKR/…`; more episodes are one download loop away, see CHECKPOINT). Build an eval script (OCR CER and detection recall against the English original text is not available per box — use the Korean page text via the `en`↔`kr` pairing by reading order, or hand-label one episode), then tune `detect.threshold`, `ocr.drop_conf`, typeset sizes; fix the misses listed in CHECKPOINT. Commercial-style long strips (owner's own legally obtained chapters via `omniscan import`) remain the real target | T3 (director) | started: first real run done; **E2** (`omniscan eval`: detection recall/precision, OCR CER and translation chrF against the SVG text layers) is **merged** (`omniscan eval`, 21/21 mutants killed after 2 tests). **First baseline (2026-09-20, all 33 episodes, untuned):** only 16 episodes have scoreable truth (episodes 16–37 keep their text in `<text>` elements without boxes → card **E2b**, written, launch when a slot frees); on those: pooled recall 0.40 (dominated by tiny/stylised SFX boxes), micro CER per chapter 0.02–0.5 for the dialogue-heavy ones (episodes 7/8/9: 0.02–0.04, episode 6: 0.11, episodes 2/5/11: 0.5–0.6), chrF of the translation 0.40 on episode 6;  **X2** (scan artefacts for the synthetic pages) is merged (16/16 mutants killed after 3 added tests; on scan-like pages OCR still finds all 17 regions at CER 0.029, confidence 0.864 vs 0.904 clean; `KoreanScan` sample in `data/raws/`); OCR of all 33 episodes is precomputed in `data/work/` |
| 2 | **J1** | Live judge check on the real Ollama with more data than KoreanDemo (needs free request slots: run with `OMNI_SLOTS=2`); check `docs/OPEN_QUESTIONS.md` D5 in practice (judge only on disagreement) | director | not started |
| 3 | **C5c** | Story memory (per-chapter summaries in `series.db`, fed to translate/judge prompts) + glossary proposals (recurring proper nouns found in OCR text → `proposed` entries) | T2 | card to write (patterns: `translate/judge*.py`, `glossary/store.py`) |
| 4 | **B11'** | Web views for the render stages: raw \| mask \| clean slider (`inpaint.json`/`patches.npz`) and a layout overlay (`layout.json`); Svelte/TS + API endpoints | T2 | card to write (patterns: existing 5 views in `webui/`, `web/` API) |
| 5 | **M10** | Sound effects: detection of SFX (`sfx` kind exists in schemas and the fixtures), a policy (translate as text, keep, or replace with lettered English) — design first, the director decides | T3 | design needed; blocked on real raws for detection |
| 6 | **Q3…** | More mutation reviews (card type Q, use `scripts/mutate.py`): queue store/worker, CBZ/PDF packaging, web API, translate profiles/prompts, watermark regions, typeset render/fit/plan (director mutation-checked C7*, a builder pass finds more) | T1 | cards are copy-edits of `docs/tasks/Q1.md`/`Q2.md` |
| 7 | **B32** | `doctor` checks: every model named in `translation_profiles.toml` exists in Ollama; Hugging Face models pinned to a `revision` with files verified | T1 | filler, card to write |
| 8 | **B31** | Duplicate / near-duplicate chapter detection (reuse the promo filter's dHash) | T1 | filler, card to write |
| 9a | **U2a → U2b** | Model catalog + `omniscan models` manager (U2a), then loaders read the downloaded folders (U2b) | T2 | U2a card written (`docs/tasks/U2a.md`), launch when a slot frees |
| 9b | **U6** | Updater core: newest app release from GitHub Releases, download + SHA-256 check, staged folder | T2 | card written (`docs/tasks/U6.md`), launch when a slot frees |
| 9 | **B33a/b** | `acquire`: **B33a** plumbing (image-list downloader, filters, DRM refusal, `sources.toml`) — ✔ **merged** (the builder caught a wrong row in the card; a review fix makes remembered rejections apply only while filters are on); **Q3** mutation review of it (card written, launch when a slot frees); **B33b** the `omniscan acquire` command + extract.pics client (needs the API docs as text, question A4) | T2 | B33a done, Q3 queued, B33b waits for A4 |
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

## Owner spec of 2026-09-20 → `docs/PRODUCT_SPEC.md` (read it), card queue
The owner's long list (OCR engines/models on all hardware, model UI + updates + qualification, slicer strategies + compare, promo filter tiers, pipeline preview/gates, reader + side-by-side + manual tools, covers, downloader, GitHub automation) is captured with decisions and evidence in **`docs/PRODUCT_SPEC.md`**. Key evidence: **PP-OCRv6 is en+zh only and scores far below the v5 Korean model on Korean pages** (page chrF 0.105 vs 0.475 on episode 6), so "v6 biggest as default" applies per language where the qualification suite says it wins; the catalog will carry every size so the user can choose.
| # | ID | What | State (2026-09-20, late evening) |
|---|---|---|---|
| S1 | **H1** | Hardware detection (`omniscan hardware`) + per-model compatibility | ✔ merged |
| S2 | **O1a** | Catalog of all OCR models/sizes (34 entries, format `hf`, `scripts/hf_catalog.py`) | ✔ merged (live: HF download + `verify --deep`) |
| S3 | **O1b** | OCR engines: model ids for `ppocr` (v5/v6, any size), `manga_ocr` crop reader | ✔ merged (live: 91 JA regions in 3.8 s, page chrF 0.686) |
| S3b | **O1d** | `paddleocr_vl` engine (fp32, ~3.6 s/region, more accurate on stylised lettering) | card written; run started 2026-09-20 but stopped by the weekly Ollama limit (no commits) — re-run when GLM is available |
| S4 | **O1c** | Qualification suite `scripts/qualify_ocr.py` (ko/cn/ja; page chrF primary — the box-level CER is untrustworthy for JA/ZH until the eval's truth assignment is checked) + `model-watch` Action | after O1d |
| S5 | **F2a → F2b → F2c** | **The promo filter is not wired into the pipeline** (finding 2026-09-20): F2a integrates tier 2 (file-level in `ingest`, slice-level in `slice`, `filter.json` = user overrides, config `filter.*`); F2b = tier 1 post-OCR text patterns (Discord/Patreon/"translated by"/URLs, global + per series, restorable); F2c = tier 3 position/shape heuristics + batch over all chapters | F2a card next |
| S6 | **S2** | Slicer strategies + `slice-compare` + per-series `series.toml` | ✔ merged (follow-up: `slice-compare` should mark a failing strategy instead of crashing) |
| S7 | **P1** | Runner gates: `omniscan run --step` with one-chapter preview | ✔ merged (live-tested) |
| S8 | **U3b → U3a → U3c** | (U3a ✔ merged 2026-09-20; U3c goal-style card written, `--max-turns 300`, launch next) | Desktop app (PySide6, native widgets): **U3b** strip view + side-by-side viewer ✔ merged; **U3a** models view (card written); **U3c** main window, library, reader integration, `omniscan gui`, settings (needs a user-config writer, director) | in progress |
| S9 | **L1** | Library covers/metadata via AniList/MangaDex/Jikan (no LLM), user upload. Shapes measured 2026-09-20: AniList GraphQL (30 req/min), MangaDex `includes[]=cover_art` + `uploads.mangadex.org/covers/<id>/<file>` (HEAD not allowed), Jikan flaky (504) | card to write |
| S10 | **B33b/c/d** | extract.pics client, page-run selection, chapter selection, completeness, plan, `omniscan acquire plan|run|check` | ✔ merged, live-tested on a CC-BY page (4 of 100 free credits used) |
| S11 | **G1** | `omni_builder.py ask` (read-only GLM lookups) | ✔ merged |
| S12 | **Q5 / Q6** | Mutation reviews: Q5 hw/hf/`series_config` (running), Q6 acquire + slicer strategies (card written) | Q5 ✔ merged (88 mutants, 9 gaps closed); Q6 waiting |
| S13 | **G2** | **Steady GPU load (owner: coil whine):** background GPU warm-up thread (hides the 9–14 s first-conv stall), typeset fit loop 2.5× fewer measurements (identical layouts) | ✔ merged; measured detect 15–17 s → 3.1 s, typeset 5–7 s → 1.5 s (the slice stage now waits 8 s for the warm-up; ~48 s of a 67 s run are outside the stage timers) |
| S14 | **G3** | Steady GPU load part 2 (goal-style card `docs/tasks/G3.md`, `--max-turns 300`): timeline of the ~48 s outside the stage timers, fix the biggest idle gaps (17 s LaMa load, start-up, warm-up placement); outputs bit-identical | WIP commit in `V:\OmniScan-wt\G3` (`scripts/measure_run.py`); stopped by the weekly Ollama limit — recreate the `.venv` junction, then `omni_builder.py resume G3 <feedback.md>` |

## Working rule (owner, 2026-09-19): use the GLM builders more, the director does less by hand
The director writes cards (exact interfaces, golden values), launches builders, merges and looks at real output; **implementation, tests, docs and routine mutation reviews go to the builders** (keep 3 slots busy; write the next card while they run; when a slot frees, launch the next card at once). One-off work that needs this machine (packaging local model caches, GitHub uploads) is the exception.

## Models, releases and updates (owner decisions 2026-09-19)
- **Models are pulled by the program, and the settings screen lists every model** (size, purpose, status) so the user decides what to download. The four vision/OCR/inpaint files are mirrored unchanged (all Apache-2.0) as assets of the GitHub release **`models-v1`** of `Nawid3333/OmniScan` (published: detector 159 MB, PP-OCR det 79 MB, Korean rec 22 MB, LaMa 206 MB, `manifest.json` with SHA-256); the LLMs come from Ollama (too big for a repo). The catalog file `config/models.toml` and the manager are card **U2a**; making the pipeline load from the downloaded folders is **U2b** (after U2a).
- **The mirror is in a PRIVATE repository**, so a program on someone else's PC gets 404 from it. The manager therefore falls back to the upstream sources (Hugging Face, the LaMa GitHub release) automatically. For the mirror (and app updates) to work for everybody the repository, or a separate public `OmniScan-releases` repository, must be public — owner decision needed.
- **App updates come from GitHub Releases built by GitHub Actions** (tags `vMAJOR.MINOR.PATCH`; assets `omniscan-<os>-<arch>.zip` + `SHA256SUMS`); "Update" in the program looks for a newer app release, downloads and verifies it. Card **U6** builds the check/download/verify part (it ignores non-app tags such as `models-v1`); U4 (PyInstaller/Nuitka packaging + the release workflow, `.github/workflows/release.yml`) and U5 (swapping the running program) follow the packaging decision. "Like FastStream" in the owner's message is not a reference I recognise — read as "update from GitHub releases"; ask if a specific app was meant.

## Self-contained desktop app (M13) — owner wish of 2026-09-19: "people just run the exe, everything is inside"
Design (details in `docs/PLAN.md` M13; open questions B4/B11/B12): PySide6 shell around the existing pipeline and the web views (`QWebEngineView`), packaged with PyInstaller or Nuitka. What "everything inside" costs, measured or estimated:
- **Torch runtime**: CPU wheel ~0.2 GB, CUDA ~3–4 GB, ROCm ~4–5 GB (per vendor, so per-vendor builds or a runtime picked at install time).
- **Vision models**: detector ~0.2 GB, PP-OCRv5 (det + Korean rec) ~0.3 GB, LaMa ~0.2 GB — easily bundled.
- **The translator is the hard part**: today it needs Ollama (a separate install) and 3 models (translategemma-12b ~8 GB, gemma4-12b ~8 GB, a cloud model). A truly self-contained app must embed a local LLM runtime (llama.cpp / `llama-cpp-python` with GGUF weights, one 4–8 GB model) or use a cloud API with the user's key. That replaces `llm/ollama.py` behind the `ChatClient` protocol (the pipeline already only sees that protocol).
- Recommendation: one installer per platform that contains the app + the matching torch runtime + the small vision models, and downloads the LLM weights once on first run (progress dialog, checksum) — or an "offline bundle" that includes them (~10–14 GB). Needs the owner's decision (B4/B11/B12) before the cards are written.
- Cards to write once the pipeline is tuned (not now): **U1** embedded LLM runtime behind `ChatClient` (+ model manager), **U2** device/runtime selection + first-run model downloader, **U3** PySide6 shell (project list, run/queue view with progress, embedded review views, settings), **U4** PyInstaller/Nuitka packaging + CI matrix (Windows/macOS/Linux), **U5** installer + auto-update. U1 and U2 can start earlier and are useful even for the CLI.

## Acquisition (downloader) — scope decided 2026-09-19
`acquire` is being built as **source-agnostic plumbing**: B33a (running) = ordered, resumable image-list downloader + non-chapter filter + DRM-platform refusal + `sources.toml`; B33b (next) = the `omniscan acquire` command and the extract.pics client (needs the API docs as text, question A4). There is no site-specific code and no HTML crawler in the project, the tool prints a "you are responsible for the rights to this content" notice, and Claude does not run it against unlicensed aggregator sites (see `docs/DECISIONS.md`).

## Risks to watch
- **No real Korean data yet** — synthetic fonts render cleaner than scans; every tuning number (detect threshold, `ocr.drop_conf`, typeset sizes, the golden-test thresholds) is provisional until A1 is answered.
- **Detector on real art**: on Pepper&Carrot it misses some text and puts low-confidence boxes on objects; on Korean pages it has never been run.
- **Ollama Cloud limits** — builders and cloud translation share a rolling ~5 h limit (3 concurrent requests) **and a weekly usage limit** (hit 2026-09-20 ~20:00: every builder got `429 … reached your weekly usage limit`, the runs retry 10× and exit with code 1). Nothing can be built by GLM until the week resets or the owner adds credits / upgrades (ollama.com/settings, ollama.com/upgrade). A hard 429 fails fast: `resume` after the limit is gone. F2a (WIP in `V:\OmniScan-wt\F2a`, uncommitted tests) and G3 (WIP commit) are waiting to be resumed.
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

# OmniScan — Build Plan (Manhwa/Manga Translator)

> **Current state and how to resume: [docs/CHECKPOINT.md](CHECKPOINT.md).** This file is the plan; that one is the snapshot.

## Context
Goal: drop in raw Korean/Chinese/Japanese chapters (`<Series>/Chapter N/`) and get back English slices that look like an official release. The pipeline removes and inpaints the original text, letters English into the bubbles, keeps names consistent across hundreds of chapters, and later replaces SFX.
Greenfield project. It runs **GPU end-to-end** on the RX 9070 XT, using as much VRAM as possible. It was planned for WSL2 but is now built and run on **Windows 11 natively** (see "Plan revisions" below; where the text below still says WSL, that is the founding plan).
Raws come in through an **acquisition layer** (the extract.pics API plus a webhook relay in the repo). **Chapters that are already translated** in a series act as a reference baseline for translating the rest.
*(Removed 2026-09-22, card RM1: the acquisition layer was deleted — raws are now exclusively user-supplied via `omniscan import` / the GUI import page. Acquisition-related text below is historical.)*
Work is split: a **GLM builder agent** (Claude Code CLI on your Ollama, **glm-5.3-flash:cloud**) builds the well-specified parts. **Claude (Opus)** writes the task cards, builds the hard parts, and reviews and merges everything.

## Plan revisions (2026-09-19) — these win over the older text below
Status by milestone (details in [CHECKPOINT.md](CHECKPOINT.md); the ordered work queue is [NEXT.md](NEXT.md)):

| Milestone | Status |
|---|---|
| M0 environment | done — moved to Windows-native; the WSL steps (1, 2, 5) and B17 "Docker for ROCm on WSL" are obsolete |
| M1 foundation | done (contracts, `VramManager`, scaffold, `doctor`) |
| M2 codec / ingest / slicer / viewer | done except C2 (hybrid GPU codec): the CPU `turbo` codec is the baseline until a real chapter is profiled |
| M2b acquisition | superseded 2026-09-22 (RM1): the relay was built but never deployed, `omniscan acquire` was built and then removed — raws are user-supplied (`omniscan import`) |
| M2c import | done (B21) |
| M3 promo filter | done (B6, B30) |
| M4 detect + OCR | C3 half built on branch `C3`; C4 not started; OCR view done (B7) |
| M5 translation | Ollama client, glossary, candidate runs, review view done (B8, B9, B29, B10); judge (C5) not started |
| M5b reference mode | not started |
| M6 inpaint, M7 typeset + export | not started (Reader view B12 and CBZ/PDF packaging B22 exist; the export stage does not) |
| M8, M9, M10, M11, M13 | not started (B14 page-mode slicer and B15 watermark regions exist as modules) |
| M12 | job queue done (B23); LAN mode not started |

Changes to the plan:
1. **Windows-native**, no WSL and no Linux container (`docs/benchmarks/windows-native.md`, `docs/DECISIONS.md`).
2. **Builders implement almost everything.** The "Who builds what" table is superseded. The director owns the contracts
   (`core/**`), the exact algorithm specs, live tuning on real pages and review; builders implement pieces sized like card B29
   (exact signatures, golden values, fake-model tests) — including what used to be "Claude cards". Splits: **C3** detect ·
   **C4** → C4a OCR core, C4b second-opinion reader, C4c zh/ja packs · **C5** → C5a post-check + agreement, C5b judge,
   C5c story memory + glossary proposals · **C6** → C6a mask + flat fill, C6b LaMa · **C7** → C7a layout engine, C7b renderer +
   export, C7c polygon fitting / font roles / colour matching. Test fixtures (X1, synthetic Korean pages) and mutation-review
   cards (Q) are builder work too.
3. **Walking skeleton before quality.** Every stage first ships in a simple v1 and `omniscan run` chains them on synthetic Korean
   pages; LaMa, second-opinion OCR, polygon fitting and SFX come after. Real Korean raws (question A1) replace the synthetic fixtures for tuning.
4. **Detection as built:** tiles of min(1280, strip width) with 0.5 overlap, resized to 640² (the model's training size).
   Detect writes `regions.json` only — no text masks (C6 derives them from OCR line boxes) and no cut validator (a region crossing
   a cut only matters when export re-cuts slices).
5. **OCR:** transformers 5.17 ships PP-OCRv5/v6 *detection* and recognition model classes and `PaddlePaddle/*_det_safetensors`
   repos exist, so both halves can run through PyTorch. C4 starts with a GPU probe; the fallback is running recognition directly
   on the detector's `text_bubble` boxes.
6. **Render-pass contract:** stage outputs stay JSON / `.npz`. `inpaint` writes `inpaint.json` + `patches.npz` (cleaned crops),
   `typeset` writes `layout.json`, `export` decodes the strip once, composites patches and glyphs onto it, cuts the slices and
   encodes. Glyph rasterisation (FreeType) runs on the CPU into small patches; compositing is on the GPU (question F9).
7. **Judge economics:** the judge only sees lines where candidates disagree (agreement below 0.9) or a locked term is violated,
   with one repair round for violations (question D5).
8. **Portability (P1+P2 done 2026-09-23, reverted 2026-09-24):** `pyproject.toml` briefly selected the
   torch backend per machine via `[project.optional-dependencies]` (`rocm-gfx1201` / `cuda` / `cpu` /
   `mps`) with `[tool.uv.conflicts]` refusing more than one at a time, and `.github/workflows/ci.yml` ran
   a real `ubuntu-latest`/`windows-latest`/`macos-latest` matrix on the `cpu` extra (5 iterative CI rounds
   to get green — the AMD rocm10 wheel index mirroring torch's whole pure-Python dependency closure and
   winning "first index wins" search over PyPI was the recurring root cause, fixed by explicit `pypi`
   source pins). **The owner decided (2026-09-24) to drop the "any OS/any GPU" goal entirely and target
   only this machine** (Windows 11 + AMD ROCm gfx1201): the `cuda`/`cpu`/`mps` extras, `[tool.uv.conflicts]`
   and the non-ROCm index entries were removed from `pyproject.toml`; `rocm-gfx1201` is now the project's
   only backend. **`.github/workflows/ci.yml` still references the removed `cpu` extra and needs a decision**
   (fix it to sync `rocm-gfx1201` and run on `windows-latest` only, or remove it) — flagged, not done, see
   `docs/OPEN_QUESTIONS.md`. Everything else portability-related (B2–B13 in `docs/OPEN_QUESTIONS.md`, the
   old M13 "universal app" framing below) is likewise moot.
   See `pyproject.toml`'s `[tool.uv.sources]` comments for the full empirically-verified account.
9. **QA loop:** every logic card gets a mutation check; mutation-review cards hand that job to builders; one large Claude
   verification pass at the end (question F7). Every Hugging Face model is pinned by `revision` in config once validated
   (`DetectConfig.revision` is the pattern), and a stage's `version` is bumped when its model changes.

## Verified environment (WSL2 era, checked live 2026-09-18 — historical)
| Item | State |
|---|---|
| GPU | RX 9070 XT, gfx1201, 16 GB VRAM (+ iGPU), driver 32.0.31041.1004 |
| CPU / RAM | Ryzen 5 7600X (12 threads), 64 GB (WSL currently gets 30 GB, the default) |
| WSL | Ubuntu 26.04.1, kernel 6.18, systemd on, `/dev/dxg` present, **no `/dev/dri`** |
| ROCm in WSL | **ROCm 10.0.0 already installed** (`amdrocm*10.0 10.0.0-4`, librocdxg 1.2.2, rocJPEG 1.7.0, rocDecode 1.9.0). **`rocminfo` sees gfx1201.** |
| Python | system 3.14.4, `uv` installed; **no torch yet** |
| Missing in WSL | node (Windows npm leaks in through PATH), gh, Claude Code CLI |
| Ollama (Windows) | 0.34.2. Local: translategemma:12b, gemma4:12b, gemma4:31b (19 GB, too big), qwen3.8:27b… Cloud: gemma4:31b-cloud, **glm-5.3:cloud, glm-5.3-flash:cloud**, kimi-k3, deepseek-v4-pro, qwen3.5:397b… Cloud **plan: Pro** (3 concurrent requests, 5-hour session and weekly limits) |
| Ollama ↔ WSL | **Unreachable from WSL** (NAT mode, no `.wslconfig`) → fixed in M0 with mirrored networking |
| GLM via Ollama | `/v1/messages` (Anthropic-compatible) **works with tool calls** for both GLM models (~3 s). glm-5.3-flash: 321B MoE / 18B active, 1M context, tools + vision. glm-5.3: 753B, 1M context, tools. |
| GitHub | `gh` on Windows logged in as **Nawid3333** (repo scope) |

**rocJPEG live test (ctypes, on your GPU):** `HARDWARE` → *"Failed to initialize the VA-API JPEG decoder"* (needs `/dev/dri`; ROCDXG passes compute only, not the VCN media engine). `HYBRID` → `NOT_IMPLEMENTED`. **Real rocJPEG can't run in WSL today.** → see Codec strategy.

## Key decisions
- **Python 3.14** (already on the system; ROCm 10 wheels cover 3.11–3.14). **No PaddlePaddle framework** (it has no ROCm support). Paddle **models** run via PyTorch/HF transformers safetensors: PP‑OCRv6 det/rec (zh/ja), `PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors` (exists, no conversion needed), and PaddleOCR‑VL‑1.5 as the second-opinion reader.
- **Modular, in-process:** each stage is its own package with a strict contract, CLI subcommand and tests. The GPU stages share one worker process (one ROCm context, tensors stay in VRAM).
- **CLI first, then the browser debug/review tool** (Firefox; usable by scanlation groups), then the manual editor, then LAN/group mode.
- **Translation = candidate runs + judge.** OCR writes `ocr.json` once. Runs happen at any time: translategemma:12b (local), gemma4:12b (local), gemma4:31b-cloud. Optional extra CJK-strong cloud candidates: glm-5.3, kimi-k3, deepseek-v4-pro. **Judge = gemma4:31b-cloud** (picks, merges or rewrites, using the raw text, candidates, glossary and panel image).
- **Glossary:** auto-proposed, you lock entries, locked entries are enforced and post-checked. **Honorifics kept.**
- **Promo filter:** example-driven; results moved to `_filtered/` (never deleted) with a review panel. Watermark inpainting is optional and comes later.
- **SFX:** full replacement is the goal; it's a late milestone.
- **Korean manhwa first**, then Chinese, then Japanese.
- **Acquisition:** ~~extract.pics API (your key stays local) → webhook → **Cloudflare Worker relay** kept in the repo and deployed by a GitHub Action → pushed to OmniScan over a WebSocket. There's no polling of extract.pics.~~ *(Removed 2026-09-22, card RM1 — raws are user-supplied via `omniscan import`.)*
- **Reference mode:** translated chapters of the same series are aligned with their raws. Terms that stay consistent across **≥ 3 reference chapters are auto-locked**; the rest are proposed. The reference chapters also feed a style guide, few-shot examples, story memory and a quality benchmark.
- **Builder model: glm-5.3-flash:cloud** (your choice: more token-efficient, slower). It escalates to glm-5.3:cloud only for a card that fails review twice.
- **Assumptions to confirm:**
  - Uniform bands **≥ 50 px** are cut points, and taller ones are allowed too. Say if 150 px is a hard maximum.
  - Platform end-cards are only filtered if you add them to the examples.

## Orchestration: Claude (Opus) + GLM builder
**Setup (M0):**
1. `.wslconfig` → `networkingMode=mirrored`, so WSL `localhost:11434` reaches Windows Ollama.
2. Install the Claude Code CLI in WSL (native installer).
3. Wrapper `~/.local/bin/omni-builder` runs `claude` with:
   - `ANTHROPIC_BASE_URL=http://localhost:11434`, `ANTHROPIC_AUTH_TOKEN=ollama`, `ANTHROPIC_API_KEY=""`
   - `ANTHROPIC_MODEL=glm-5.3-flash:cloud` (**default**; token-efficient, has vision for UI screenshot checks), `ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5.3-flash:cloud`, `CLAUDE_CODE_SUBAGENT_MODEL=glm-5.3-flash:cloud`
   - `CLAUDE_CODE_MAX_OUTPUT_TOKENS=32000` (the GLMs think out loud), `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, `API_TIMEOUT_MS=600000` (flash is slow)
   - Escalation: `omni-builder --model glm-5.3:cloud` only after 2 failed review rounds on a card.
4. **Concurrency:** the Pro plan allows 3 concurrent requests → the wrapper takes an `flock` slot, **max 2 builders at once**, 1 slot kept for tests. While a big cloud translation run is active → 1 builder.

**Per task card loop:**
1. Claude writes `docs/tasks/<ID>.md` from a template: goal, files, **exact signatures**, **explicit definitions (no ambiguity; the probe showed GLM burns most of its tokens resolving vague specs)**, acceptance tests, out-of-scope, commands, report format.
2. `git worktree add ~/projects/omniscan-wt/<ID> -b <ID>` → the builder runs headless in the background:
   `omni-builder -p "$(cat docs/tasks/<ID>.md)" --output-format stream-json --verbose --max-turns 150 --permission-mode acceptEdits --settings .builder/settings.json` → log `.builder/logs/<ID>.jsonl`.
3. The builder must finish by writing `REPORT.md` (changes, tests run, open questions). **If the spec is unclear, it stops and asks in REPORT.md instead of guessing.**
4. Claude reviews: REPORT.md, `git diff main...<ID>`, `uv run pytest`, `ruff`, `pyright`, plus a code review.
   - Up to 2 fix rounds via `omni-builder --resume <session>`; otherwise Claude fixes it directly.
   - Then a PR via `gh pr create` → squash-merge → push.
5. **Guardrails (`.builder/settings.json`):**
   - Allowed: Read/Edit/Write/Glob/Grep, WebFetch (for doc research), `uv run|sync *`, `git status|diff|add|commit|log *`, `ls/cat/grep/find`.
   - Denied: `git push`, `sudo`, `rm -rf`, `curl|wget`, and edits to `src/omniscan/core/**`.
6. `CLAUDE.md` in the repo: conventions, contract ownership, definition of done, commands.
7. **Pilot = card B1 (scaffold)** calibrates the card style, model choice and turn/token budget before the rest of the work goes out.
8. GLM is also used for **bulk research** (reading library docs and summarizing to `docs/research/`). Claude verifies the key claims.

**Who builds what (so Claude only does the heavy parts):**
| GLM builder (glm-5.3-flash via Ollama) — ~20 cards, most of the code volume | Claude (Opus) — ~12 cards + all reviews |
|---|---|
| B1 scaffold + `doctor` (pilot) · B2 turbo baseline + codec benchmark harness · B3 ingest · B4 slicer · B5/B7/B10/B11 debug views · B6 promo filter · B8 Ollama client · B9 glossary store · B12 export + reader · B13 editor UI · B14 page mode · B15 watermark (fixed) · B16 LAN mode · B17 Docker · B18 Cloudflare relay + GitHub Action · B19 `acquire` downloader (batch + ad-hoc paste-URL) · B20 reference-mode infra + Compare view · B21 local import (loose files/folders → library layout) · B22 export formats (CBZ/PDF) · B23 job queue + notifications | M0 environment · C1 contracts/VRAM manager · C2 hybrid GPU JPEG codec · C3/C4 detection + OCR · C5 translation/judge/glossary logic · C6 inpainting · C7 typesetting · C8 incremental recompute · C9/C10 zh/ja packs · C11 watermark detector · C12 reference alignment + calibration · SFX (M10) |
| Also: bulk research (reading library docs into `docs/research/`), writing tests/fixtures, docs | Also: writing every task card, reviewing every PR, fixing what fails review twice |

The builder starts right after M0: **B1 is the pilot** (it calibrates card style and token/turn budget), then B2/B3/B4 and B18/B19 run on the two builder slots while Claude builds C1/C2.

**Where Claude runs:** after M0, reopen VS Code **connected to WSL** (`code ~/projects/omniscan`) and continue there with Claude Code, so shells are Linux-native. `docs/PLAN.md` = a copy of this plan inside the repo.

## Git & repo
- `gh repo create Nawid3333/OmniScan --private`. The repo lives at `~/projects/omniscan` (WSL ext4, **not** `/mnt/c`). Package name: `omniscan`.
- Install `gh` in WSL + `gh auth login` (you, once) + `gh auth setup-git`.
- Branch per card, PR per card (review history on GitHub), squash merge, no force pushes to `main`.
- pre-commit: ruff format/check.
- `.gitignore`: `models/ work/ output/ library/ samples/ .env* *.npz .builder/logs/`. **Raw chapters are never committed** (copyright); tests use synthetic fixtures.

## Architecture

### Folder layout
```
<library_root>/<Series>/Chapter N/*.jpg|png|webp     ← raws, READ-ONLY (written only by `omniscan import`)
<library_root>/<Series>/_reference_en/Chapter N/…    ← already-translated chapters (optional baseline)
promo_examples/global/*.jpg  promo_examples/<Series>/*.jpg
<work_root>/<Series>/series.db  glossary.yaml
                     Chapter N/manifest.json ingest.json slices.json filter.json regions.json ocr.json
                               translations/<run_id>.json final.json masks/*.npz marks/*.jpg (small judge/debug thumbnails)
<output_root>/<Series>/Chapter N/0001.jpg …          ← final English slices
<output_root>/<Series>/_filtered/Chapter N/…  + report.json
```
**Minimal IO:**
- Slices are **virtual (y-ranges)** while processing. Physical JPEGs are written only at export (or in debug mode).
- Intermediate artifacts are JSON / `.npz` only.
- Filtered raw files are copied byte-for-byte (no re-encode).
- The debug tool draws overlays client-side (SVG over the raw JPEGs), so no preview files are needed.

### Acquisition layer (extract.pics + webhook relay)
> **Superseded 2026-09-22 (card RM1):** the acquisition layer (extract.pics client, webhook relay, `omniscan acquire`) was removed; raws are user-supplied. Kept as historical design record.

**What extract.pics does (from its docs):**
- `POST https://api.extract.pics/v0/extractions` with the API key in the `Authorization` header; batches are supported.
- The webhook URL is set **once per extract.pics project** in the project settings.
- Events: `extraction_done` {id, status done|error, url, images[{id,url}], project_id, batch_id}, `extraction_batch_done` {batch_id, extraction_status{}}, `download_done` {id, status, temporary signed ZIP `url`}.
- **Non-200 responses are retried** (so duplicates are possible). **No signature** is sent, and the URL must be public.

**Why GitHub itself can't be the webhook (verified 2026-09-18):**
- The extract.pics project form has only two fields, `name` and `webhook_url`: no custom headers, no secret, and the body format is fixed (`{type, data}`).
- GitHub's only inbound triggers, `repository_dispatch` and `workflow_dispatch`, need an `Authorization: Bearer` header (tokens can't go in the URL) and a body with `event_type` or `ref`.
- A live test sending the exact extract.pics payload got **401 "Requires authentication"** from `api.github.com/.../dispatches` and **405** from GitHub Pages.
- Codespaces public ports only exist while a codespace is running (idle timeout), so they're not usable either.
- So GitHub **hosts and deploys** the relay code (Actions), and Cloudflare **receives** the webhook.

**Relay (`relay/`, TypeScript Cloudflare Worker + SQLite-backed Durable Object, free tier):**
- `POST /hooks/extractpics/<HOOK_SECRET>`: reject a wrong secret, store the event **idempotently** (key = type + id), answer 200 immediately. Events expire after 14 days.
- `GET /ws` with `Authorization: Bearer <CLIENT_TOKEN>`: WebSocket (hibernation API) that **pushes each event the moment it arrives**. `GET /events?after=<cursor>` replays anything missed while the PC was off. `POST /ack` removes delivered events.
- Deploy: `.github/workflows/relay-deploy.yml` runs wrangler on every push to `relay/**`. It reads `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `HOOK_SECRET` and `CLIENT_TOKEN` from GitHub secrets. Tests use vitest + `@cloudflare/vitest-pool-workers`.
- The relay **never sees your extract.pics key**. OmniScan starts extractions locally; the relay only forwards results.

**`omniscan acquire <Series>` (`src/omniscan/acquire/`):**
1. Reads `sources.toml` (a list of chapter URLs, or a URL template with `{n}` + a range) and submits one extraction batch per group of chapters.
2. Waits on the relay WebSocket (no polling).
3. On `extraction_done`, it downloads the images in page order: directly with the right Referer, or via extract.pics' async ZIP (`download_done`) when direct download is blocked. Downloads are streamed and checksummed, and resume after interruptions.
4. **Non-chapter images are dropped** (logos, thumbnails, ads) using width-consistency and size heuristics, plus the promo filter later.
5. Output goes to `library/<Series>/Chapter N/0001.jpg…`, which is exactly the folder structure the rest of the pipeline expects.
6. `acquire` can also fetch `_reference_en` chapters the same way.

### Reference mode (already-translated chapters as a baseline)
1. **Process** each `_reference_en/Chapter N` through the same vision pass (English OCR = PP-OCRv6 rec en), next to its raw chapter.
2. **Align raw ↔ EN:** match the art between the two strips. Feature matching on text-masked, downscaled art gives a piecewise y-mapping, which tolerates different slicing, crops and credit pages. Bubbles are then paired by IoU in the mapped space → a **parallel corpus** (source line ↔ human English line + confidence) in `series.db`.
3. **Glossary:** the LLM extracts term pairs from the aligned lines. **Consistent in ≥ 3 chapters → auto-locked** (the human translator's choice wins); inconsistent → proposed.
4. **Style guide** (`style.md`, injected into every prompt): honorific usage, per-character voice and register, profanity level, SFX conventions, number and units style.
5. **Few-shot retrieval:** embed source lines (`qwen3-embedding:8b`, already on your Ollama), then retrieve the top-k similar parallel pairs as in-context examples for each chunk.
6. **Story memory:** summaries of the reference chapters (built from the English text) give chapter 41+ full story context. The tail of the last reference chapter seeds continuity.
7. **Calibration benchmark:** translate held-out reference chapters blind with every profile and judge setup. Score them against the human version (LLM pairwise judge + chrF), then **pick the best profiles and judge prompt per series**.
8. **Compare view** in the web tool: raw | human EN | OmniScan output, side by side.

### Stages (package + CLI subcommand + JSON artifact each)
0. **acquire** (removed 2026-09-22, card RM1): raws come from `omniscan import` / the GUI import page.
1. **ingest:** discover series/chapters (natural sort); convert non-JPEG to JPEG q95 4:4:4 (EXIF-rotate, flatten alpha on white, CMYK → RGB).
2. **vision pass (per chapter, all vision models resident, pixels decoded once into VRAM):**
   - a. codec decodes **straight into the preallocated chapter strip** (so stitching is free). Mismatched widths are resized on the GPU to the dominant width.
   - b. **promo pre-check (file level)** on each file's region of the strip; matches are removed from the strip on the GPU and the raw file is copied to `_filtered/`.
   - c. **slicer:** per-row uniformity on the GPU (JPEG-noise tolerant, slow vertical gradients allowed) → cut only inside uniform bands **≥ 50 px** (cut at band centre) → DP picks cuts near the target height (default 3000, range 1500–6000, hard max 15000, never > 65,535) → `blank=true` for fully uniform slices (**they skip OCR, inpainting and typesetting**) → forced cut at the lowest-detail row if no band is found (flagged red).
   - d. **promo filter (slice level):** pHash/dHash + small embedder (SigLIP2/DINOv2) vs the examples.
   - e. **detect:** `ogkalu/comic-text-and-bubble-detector` (RT-DETR-v2, Apache-2.0: bubble / text_bubble / text_free), tiled (see Plan revisions #4; no text masks and no cut validator here).
   - f. **ocr:** PP-OCRv6 det → rec (Korean: korean_PP-OCRv5_mobile_rec) batched on the GPU → lines grouped into regions → reading order → confidence. Low confidence → PaddleOCR-VL-1.5. Writes `ocr.json` + set-of-marks thumbnails for the judge.
3. **text pass:** glossary extraction → translation runs (independent, resumable, throttled) → judge → glossary post-check/repair.
4. **render pass (per chapter; decode again once):** inpaint (flat fill for uniform bubble interiors, LaMa for text over art) → typeset (fit to the bubble polygon, font roles, hyphenation, size search, stroke, colour match; overflow → LLM condense) → composite **edited patches onto the original pixels only** → encode → write output.
- **Cloud streaming mode:** when all translators and the judge are cloud models, vision → translate → render runs per chapter with the pixels kept in VRAM (a single decode).
- **Resumable everywhere:** `manifest.json` = hash of inputs + stage version + config. Also used for the editor's single-slice re-render.

### Codec strategy (GPU-first; a benchmark decides the default)
`gpu/codec/` backends behind one interface `decode_into(files, strip, offsets)` / `encode(strip_range) -> bytes`:
1. **`rocjpeg`:** ctypes/nanobind wrapper, **auto-enabled if `rocJpegCreate(HARDWARE)` succeeds** (a future WSL/Windows driver, or native Linux). Batched decode into HIP memory.
2. **`hybrid` (our decoder, built by Claude):**
   - C++ extension (nanobind + scikit-build-core) using libjpeg-turbo's `jpeg_read_coefficients` does **only the entropy (Huffman) decode**, multithreaded, into one **pinned int16 coefficient arena**.
   - One async H2D copy on a copy stream.
   - The GPU does dequantization + batched 8×8 IDCT (matmul with the DCT basis) + libjpeg-compatible "fancy" chroma upsampling + YCbCr→RGB, **writing straight into the strip**.
   - Handles progressive, 4:2:0/4:2:2/4:4:4 and grayscale.
   - Accuracy test: ≤ 2 levels max diff and > 50 dB PSNR vs libjpeg-turbo.
   - **Hybrid encode** mirrors it: the GPU does colour conversion + FDCT + quantization, and the CPU does only the Huffman step via `jpeg_write_coefficients`.
3. **`gpu_huffman` (stretch):** fully GPU entropy decode (self-synchronizing parallel Huffman decoding). Built only if the benchmark shows the CPU Huffman step is the bottleneck.
4. **`turbo`:** libjpeg-turbo CPU decode + upload. **Benchmark baseline only**, not a pipeline default unless you approve it after seeing the numbers.
- **Benchmark (`scripts/bench_codec.py`)** on your sample chapters:
  - Measures: end-to-end time per chapter into VRAM, MP/s, **CPU %**, H2D bytes, VRAM, peak RAM.
  - Output: `docs/benchmarks/codec.md`.
  - **Default = fastest end-to-end; ties → lowest CPU.**

### GPU / VRAM strategy (use as much of the 16 GB as possible)
- **Budget ≈ 14.5 GB** (desktop headroom, configurable). Batch sizes are **auto-tuned to fill the budget**. `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`.
- **Streams:** read → pinned double-buffers → copy stream → compute stream → D2H/encode stream, so IO, transfer and compute overlap. SAM/ReBAR is enabled on the Windows side; M0 measures pinned H2D bandwidth (ROCDXG lists pinned-memory caveats).
- **Model groups are never co-resident** with the local LLM:
  - Vision (detector + PP-OCR + embedder + optional PaddleOCR-VL ≈ 3–5 GB, plus chapter strips)
  - Inpaint (LaMa ≈ 2–4 GB)
  - Local LLM in Windows Ollama (translategemma:12b ≈ 8 GB + KV)
- `gpu/vram.py` **VramManager:** `acquire(group)` frees the others (`del` + `gc` + `empty_cache`, Ollama `keep_alive:0` checked via `/api/ps`) and logs peak usage per stage. WDDM can silently page VRAM to RAM, so the budget is enforced by us.

### Translation & glossary
- **Profiles:** `{endpoint: local|cloud, model, style: translategemma|chat_json, options}`. Endpoints are Ollama only: local `localhost:11434`, or cloud `https://ollama.com` with `OLLAMA_API_KEY`.
- **chat_json:** chapter script in reading order with region ids, the **relevant glossary subset** (alias/particle-aware), story-so-far summary, the previous chapter's tail, honorifics policy; Ollama `format` JSON schema.
- **translategemma:** its required prompt format. Locked names are **pre-substituted** in the source, then verified.
- **Judge:** set-of-marks slice images (numbered boxes) so the model can infer speaker, gender and tone. It updates story memory and proposes new glossary terms.
- **Glossary entry:** source, target, type, gender/pronouns, aliases, Korean particle forms (이/가/은/는/을/를/의/아/야/도/에게…), notes, `proposed|locked|rejected`, first seen, count. **Post-check:** a locked source term ⇒ its target must appear. Violations → judge repair → otherwise flagged.

### Repo layout (`~/projects/omniscan`)
```
CLAUDE.md  pyproject.toml  uv.lock  .python-version(3.14)  config/default.toml  .builder/settings.json
src/omniscan/core/   config schemas stage manifest paths log         (contracts — Claude only)
src/omniscan/gpu/    vram.py  codec/{base,rocjpeg,hybrid,turbo}.py  csrc/ (nanobind ext)
src/omniscan/{acquire,ingest,slicer,promo,detect,ocr,llm,glossary,translate,reference,inpaint,typeset,export,web}/  cli.py
relay/ (Cloudflare Worker, TypeScript, wrangler.toml, tests)   .github/workflows/relay-deploy.yml
scripts/ bench_codec.py bench_ocr.py rocjpeg_probe.py     docs/ PLAN.md ARCHITECTURE.md tasks/ research/ benchmarks/
tests/ unit/ fixtures/synthetic/ e2e/      fonts/ (OFL defaults + yours)     models/ (gitignored)
```

## Milestones & cards
**Owners:** [B] = GLM builder · [C] = Claude · [You] = sudo / accounts / samples.

### M0 — Environment & orchestration [C + You]
1. [You] `.wslconfig`: `networkingMode=mirrored`, `memory=48GB`, `processors=12`, `swap=16GB`, `sparseVhd=true`. In `/etc/wsl.conf`: `[interop] appendWindowsPath=false`, and add the VS Code bin dir to PATH in `~/.bashrc`. Then `wsl --shutdown`.
2. [You] Run the sudo block: `apt install build-essential cmake ninja-build pkg-config libjpeg-turbo8-dev libraqm0 gh nodejs npm`, then `gh auth login`.
3. [C] uv project with Python 3.14 → `torch[device-gfx1201]==2.13.0+rocm10.0.0`, `torchvision[device-gfx1201]==0.28.0+rocm10.0.0` from the `https://stable.repo.amd.com/rocm/whl-next/` index (pinned in pyproject). Verify:
   - `torch.cuda.is_available()`
   - fp16 matmul TFLOPS
   - `mem_get_info`
   - pinned H2D bandwidth
   - If the pip ROCm doesn't find librocdxg → set `LD_LIBRARY_PATH` to `/opt/rocm` and document it.
4. [C] Ollama from WSL: `curl localhost:11434/api/version`; test `/v1/messages` with glm-5.3.
5. [C] Install the Claude Code CLI in WSL; create the `omni-builder` wrapper + flock slots + `.builder/settings.json`; do a smoke run (`omni-builder -p "print the python version"`).
6. [C] `gh repo create Nawid3333/OmniScan --private`; push `CLAUDE.md`, the task template, `docs/PLAN.md`, `.gitignore`.
7. [C] `scripts/rocjpeg_probe.py` committed (re-run on every driver update).
8. ~~[You] Create a free Cloudflare account → an API token ("Edit Cloudflare Workers") + your account ID → I store them as GitHub repo secrets with `gh secret set`. Put your extract.pics API key in `~/.config/omniscan/secrets.env` (never committed). Once the relay is deployed (M2b), paste its hook URL into the extract.pics project settings.~~ *(Obsolete 2026-09-22, card RM1: the relay was never deployed and the subsystem was removed.)*

### M1 — Foundation
- **C1:** contracts:
  - `core/schemas.py` (pydantic v2, versioned: Chapter, Slice, Region, OcrLine, Candidate, FinalLine, GlossaryEntry, FilterDecision, Layout)
  - `core/stage.py` (Stage protocol, chapter-pass runner, manifest hashing)
  - `gpu/vram.py`
  - the `gpu/codec/base.py` interface
  - `docs/ARCHITECTURE.md`
- **B1 (pilot):** scaffold:
  - ruff / pyright / pytest / pre-commit
  - TOML + env + secrets config
  - typer CLI with every subcommand stubbed
  - rich logging
  - **`omniscan doctor`** (GPU, ROCm, torch, rocJPEG probe, Ollama local + cloud, codec backend)

### M2 — Codec + ingest + slicer + debug viewer v0
- **C2:** hybrid codec (C++ ext + GPU kernels) + rocjpeg wrapper + hybrid encode.
- **B2:** turbo baseline backend + `bench_codec.py` harness (metrics above).
- → **Decision gate:** codec default chosen from the benchmark numbers.
- **B3:** ingest.
- **B4:** slicer to the exact spec. Synthetic tests:
  - never cuts through bubbles or art
  - never uses bands < 50 px
  - tolerates JPEG noise
  - allows gradients
  - slice heights stay in range
  - `blank` flag set correctly
  - forced cuts are flagged
- **B5:** web tool v0 (FastAPI + Svelte/TS):
  - chapter browser
  - **slicer view** (cut lines, bands, forced cuts in red, original file boundaries dashed, slice sizes)
  - raw ↔ slices view with synced scroll

### M2b — Acquisition [B, reviewed by C]
> **Superseded 2026-09-22 (card RM1):** the subsystem described here was built and then removed; raws are user-supplied (`omniscan import`). Kept as historical record.
- **B18:** Cloudflare Worker relay + Durable Object + GitHub Action deploy + vitest suite. *Accept:*
  - duplicate POSTs are stored once
  - a wrong secret returns 404
  - a WebSocket client receives an event within 1 s
  - events missed while offline are replayed via `/events?after=`
- **B19:** `omniscan acquire` (`sources.toml`, batch submit, WebSocket wait, ordered + resumable download, ZIP fallback, non-chapter image filter). *Accept:* a mocked extract.pics + relay end-to-end test writes `Chapter N/0001.jpg…` in page order.
  - **Ad-hoc mode:** `omniscan acquire <Series> --url <chapter_url> [--url <chapter_url> ...]` (or one URL per line piped
    on stdin) runs a single extraction for just those links without editing `sources.toml` first — the quick path
    for "a new chapter just came out, grab it now." Auto-detects the chapter number from the URL/page title when
    possible; falls back to prompting or `--chapter N`. On success it also appends the link to `sources.toml` so
    the series' link history stays complete for later re-runs.
  - **Scope boundary:** acquisition targets scanlation/fan raw aggregators (the kind extract.pics already points
    at), not paid DRM-protected official platforms (Naver Webtoon, Kakao(Page), Lezhin, Ridibooks, Bomtoon,
    Kuaikan) — several of those render pages as obfuscated canvases specifically to block extraction, and bypassing
    that is a different, much riskier problem than this project takes on. `doctor`/`acquire` should recognize and
    warn on a known DRM-platform domain rather than silently failing weirdly.
- **[You]:** paste the hook URL into the extract.pics project → live test on one chapter URL.

### M2c — Local import [B]
- **B21:** `omniscan import <path> [--series NAME] [--chapter N]` — the second acquisition path, for raws you
  already have on disk (a scanlation group's own working files, a manually downloaded archive, a USB drive of
  scans) instead of pulling from the network. Accepts a single folder, a folder-of-folders (auto-splitting into
  chapters by subfolder name via the existing `chapter_number()`/`natural_key()` parsing in `core/paths.py`), or a
  loose flat dump of images (grouped by filename-prefix heuristics, flagged for confirmation when ambiguous rather
  than guessed silently). Copies (never moves, unless `--move` is passed) into
  `library_root/<Series>/Chapter N/`, running the same ingest JPEG-normalisation any other raw goes through.
  *Accept:* importing a nested folder tree reproduces the right `<Series>/Chapter N/` layout; an ambiguous flat
  dump is reported, not guessed; re-importing the same source is a no-op (content-hash de-dup, no duplicate
  files).

### M3 — Promo filter [B]
- **B6:** file-level + slice-level hooks, `_filtered/` + report, `omniscan filter restore`, a **series-wide Filtered review panel** with a Restore button.

### M4 — Detection + OCR [C] (+ view [B])
- **C3:** detector (tiling, NMS across tiles, regions); masks and the cut validator moved out (Plan revisions #4).
- **C4:** OCR engines via transformers; region grouping; reading order; PaddleOCR-VL fallback.
  - `bench_ocr.py`: PP-OCRv5 vs v6 det, fallback rate, and your local Ollama OCR models (glm-ocr, deepseek-ocr) as extra second-opinion candidates.
- **B7:** OCR view (boxes by kind, confidence, disagreements).

### M5 — Translation [C core, B infra]
- **B8:** Ollama client (local + cloud, key, streaming, `format` schema, retry/backoff, rate limits, `keep_alive:0`, `/api/ps`, profiles).
- **B9:** glossary store (SQLite, YAML import/export, CLI, alias + particle matcher).
- **C5:** glossary extraction, run executor (both styles), story memory, judge + set-of-marks, post-check/repair, condense.
  - A/B evaluation of candidate models on a sample chapter.
- **B10:** translation review view (source | candidates | final | rationale, glossary hits, flagged filter).

### M5b — Reference mode [C core, B infra]
- **B20:** `_reference_en` ingestion (incl. via `acquire`), DB tables for parallel pairs, style guide and embeddings, `omniscan reference add|status|rebuild`, and the **Compare view** (raw | human EN | ours).
- **C12:**
  - raw↔EN strip alignment + bubble pairing
  - glossary mining with auto-lock (≥ 3 consistent chapters)
  - style-guide extraction
  - few-shot retrieval (`qwen3-embedding:8b`)
  - story memory from the EN chapters
  - **calibration benchmark** that picks each series' best translation profiles and judge prompt

### M6 — Inpainting [C] (+ view [B])
- **C6:** flat fill + LaMa, crop batching, GPU compositing.
- **B11:** raw | mask | clean slider.

### M7 — Typeset + export → **first end-to-end Korean chapter**
- **C7:** typesetting engine.
- **B12:** export (composite onto the original pixels, codec encode, mirrored tree) + **Reader view** (continuous strip, raw | final side by side).

### M8 — Manual editor
- **B13:** edit OCR text / translation / box / font role, glossary lock, re-render slice.
- **C8:** incremental recompute API.

### M9 — Chinese & Japanese
- **C9:** zh pack.
- **C10:** ja pack (page mode, right-to-left order, vertical text → horizontal English; OCR benchmark incl. manga-ocr).
- **B14:** page-mode passthrough in the slicer.

### M10 — SFX full replacement [C]
SFX classification + reading (PaddleOCR-VL / Gemma 4 vision), onomatopoeia glossary, large-area inpainting (LaMa → diffusion when needed, VRAM-scheduled), stylized rendering.

### M11 — Watermarks
- **B15:** fixed-position mask per series.
- **C11:** random-position detector.

### M12 — Group mode & packaging [B]
- **B16:** LAN mode (auth, edit log, page locks).
- **B17:** Docker for ROCm on WSL (`/dev/dxg` + `/usr/lib/wsl`); user docs; Ollama-only API keys.

### M13 — Unified desktop app (deferred, post first-end-to-end) [C + B]
One packaged application (Windows `.exe`) instead of CLI + separate browser tool. **Scope narrowed 2026-09-24:
Windows + this machine's AMD ROCm GPU only** — the earlier "any GPU / any OS" framing (macOS `.app`, Linux
binary, CUDA/MPS/CPU auto-detect, cross-platform packaging) is dropped along with the rest of the portability
work (see the "Portability" plan revision above).
- **Shell:** PySide6; no C++/Java needed beyond the GPU codec extension already planned for C2.
- **Debug/review area:** not rebuilt from scratch — the browser views from B5/B7/B10/B11 (slicer cuts, OCR
  boxes, translation candidates, inpaint before/after) are embedded via `QWebEngineView`, so that work carries
  over as the desktop app's dedicated debugging panel instead of a separate Firefox tab. Each pipeline stage
  (slice / OCR / translate / inpaint) gets its own reviewable pane in one window, enterprise-tool style, not
  just a log stream.
- **Packaging:** PyInstaller or Nuitka; a Windows build bundling (or first-run-downloading) the ROCm torch
  wheels for this GPU — no per-vendor installer variants needed.
- Scheduled **after** the pipeline works end-to-end on this machine (post-M9/M10).

**Critical path:** M0 → C1 + B1 (pilot) → C2/B2 benchmark gate → B3/B4 → C3/C4 → B8/B9 + C5 → C6 → C7 + B12 (first end-to-end).
**In parallel (builder slot 2):** B18/B19 acquisition, the views (against fixture JSON), then B20.
**After first end-to-end:** C12 reference mode (it needs working OCR + translation).

## Verification
- `omniscan doctor` all green, including Ollama from WSL and the GLM builder smoke run.
- Codec: the accuracy test vs libjpeg-turbo passes; the benchmark table is committed; the default backend is justified by numbers.
- Unit tests: slicer synthetic suite, ingest, promo fixtures, schema round-trips, glossary matcher + post-check, VramManager (never two groups loaded at once).
- **Golden sample:** you put 1–2 raw chapters in `<library_root>/Sample/` → `omniscan run Sample` end to end → every stage inspected in Firefox.
- Per-stage metrics: time, peak VRAM (≤ budget), CPU %, OCR fallback rate, **glossary violations = 0 for locked terms**, overflow count.
- Re-running is a no-op (resumable); editing one line re-renders only that slice.
- Builder loop: every merged card has a PR, a green CI-equivalent (pytest/ruff/pyright) and a REPORT.md.
- ~~Acquisition: the relay vitest suite is green in the GitHub Action. A live extract.pics chapter → event arrives over the WebSocket with no polling → the chapter folder is written in order.~~ *(Superseded 2026-09-22, card RM1: the acquisition subsystem was removed; raws come from `omniscan import`.)*
- Reference mode: on a series with translated chapters, alignment precision is spot-checked in the Compare view. Auto-locked terms match the human translation. The calibration report ranks the profiles, and the chosen setup beats the default on held-out reference chapters.

## Backlog — scoped out for now, don't lose these
Things worth building that don't have a card yet; each gets promoted to a real milestone/card when its
dependencies land. Checked against prior art (`zyddnys/manga-image-translator` is the closest comparable OSS
project — full detect/OCR/translate/inpaint/typeset pipeline, multi-backend translation, 20+ languages; it
validates the overall pipeline shape but is CUDA/Nvidia-first and not GPU-VRAM-budgeted the way this project is).

- ~~**Export formats beyond flat JPEG slices**~~ — **done** (`omniscan pack`, CBZ/PDF; card B22).
- ~~**Job queue + notifications**~~ — **done** (`omniscan queue add/list/run/pause/resume/cancel/retry/clear`,
  webhook notifier; card B23).
- **Model management:** HF model downloads are currently implicit (whatever `transformers`/`huggingface-hub`
  pulls on first use). Needs explicit version pinning, an integrity check surfaced in `omniscan doctor`, and a
  documented rule that **a model upgrade bumps the owning stage's `version`** in `core/stage.py` terms (so the
  resumability hash correctly invalidates old output) — this rule should go in `docs/ARCHITECTURE.md` once C1
  is revisited, it's cheap to write down now so it isn't forgotten.
- **Cost/usage tracking for cloud LLM calls:** directly motivated by burning ~5M tokens / $26.77 on a single
  stuck builder run this session. Translation runs against Ollama Cloud should log token usage and estimated
  cost per chapter/series (the `CandidateRun.usage` field in `core/schemas.py` already has a place to put this
  — it just isn't populated by anything yet), with an optional budget cap that pauses a batch run rather than
  silently spending through a rate limit.
- **Per-chapter QA report:** formalize the metrics already listed under Verification (OCR confidence
  distribution, glossary violation count, typeset overflow count, promo-filter false-positive rate) into one
  `qa.json` per chapter, surfaced as a summary badge in the debug views (B7/B10/B11) instead of only being
  eyeballed in raw JSON.
- **Chapter watch / incremental catch-up:** stale since RM1 removed the acquisition subsystem (raws are now
  exclusively user-supplied via `omniscan import`) — the original framing ("re-run `acquire`'s link discovery")
  no longer applies. If still wanted, it would mean something like "diff a folder of newly-imported chapters
  against what's already in the library," a different, smaller shape than originally scoped.
- **Edit history in the manual editor:** the web viewer's Edit tab (card B33) already lets you click-to-edit a
  region's final text in place, but there's still no undo / version history per region — a bad manual edit
  simply overwrites `final.json`'s line (marked `decision: "manual"`) with no way back except re-running judge.
  Worth a lightweight history (even just N previous values per region) if manual edits turn out to need undo
  in practice.
- ~~**Duplicate/near-duplicate chapter detection**~~ — **done** (`omniscan match duplicates`, reusing CM1's own
  page-hash quality metric self-compared rather than B6's per-file dHash directly; card B31).
- **Privacy:** the packaged app (M13) handles copyrighted raw scans; it should not phone home any telemetry by
  default. Worth stating explicitly once M13 is picked back up.

## Risks / watch items
- The pip-installed ROCm in the torch wheels vs the system ROCm 10 + librocdxg in WSL → verified in M0.3.
- rocJPEG stays unavailable in WSL until AMD exposes VCN through ROCDXG → the hybrid codec + auto-probe on every driver update.
- The hybrid codec is real engineering (C++ ext + GPU kernels); the benchmark may show the CPU Huffman step is the bottleneck → `gpu_huffman` stretch.
- Ollama Cloud Pro limits are shared by the builder and translation runs → flock slots, throttling, resumable runs; watch the 90% usage email.
- GLM verbosity and spec drift → unambiguous cards, the "ask, don't guess" rule, review before every merge.
- Licenses: avoid GPL/AGPL models (comic-text-detector, ultralytics YOLO); commercial fonts are yours to supply.
- extract.pics: the API limits per plan are unverified (the UI shows a 100 images / 100 MB download cap) → acquire splits large chapters into batches. Some sites block direct image downloads → the ZIP download fallback.
- The webhook has no signature → the secret URL path + idempotent storage. Cloudflare free tier (100k requests/day) is plenty.
- Reference alignment can fail on heavily re-edited scanlations (cropped or redrawn panels) → low-confidence pairs are excluded and shown in the Compare view.

# Open questions for the project owner

Questions I (Claude) should ask you at natural pauses, so nothing has to be remembered from chat history.
Each has the **default I will use until you answer**, so nothing is blocked unless it says so.

**How this file works**
- At a natural pause I ask 2–3 relevant questions (not all at once), then record your answer in the *Decisions* table
  at the bottom with the date, and delete the question from the lists.
- A "blocks" entry means real work cannot proceed or cannot be validated without your answer.
- Ordered roughly by how soon the answer matters. Last updated 2026-09-19.

## A. Data and accounts (needed soon)

| ID | Question | Blocks / why it matters | Default until answered |
|---|---|---|---|
| A1 | Can you provide **1–2 real Korean raw chapters** (a folder of images; `omniscan import "<folder>" --series Sample --chapter "Chapter 1"`)? | **Blocks** tuning detection/OCR on real Korean text; everything so far was validated on English/synthetic pages | I keep using the CC BY *Pepper&Carrot* pages (English) |
| A3 | Cloudflare account + API token ("Edit Cloudflare Workers") + account id, your extract.pics API key, and later the relay hook URL (plan M0 step 8) | **Blocks** deploying the relay (B18) and live `acquire` | Relay code stays undeployed |
| A4 | Can you paste the **extract.pics API docs** (request/response shapes)? The site is a JS app I cannot fetch | **Blocks** the concrete extract.pics client in `acquire` (B19); the rest of acquire can be built without it | I build downloader/sources/filter first and leave the client as a stub |
| A5 | Which **sites** do you actually raw-acquire from? | Referer/hotlink rules, non-chapter-image filter heuristics, the DRM-platform warning list | Generic heuristics only |
| A6 | Which **promo / end-card** pages appear in your raws (drop 3–5 examples into `promo_examples/global/` and per series)? | The promo filter is example-driven; with no examples it filters nothing | Nothing filtered |

## B. Platforms, distribution and the universal app

| ID | Question | Blocks / why it matters | Default until answered |
|---|---|---|---|
| B2 | Which **OS + GPU combinations must be first-class** (tested, supported) vs best-effort? Windows: AMD / NVIDIA / Intel / CPU; macOS: Apple Silicon (MPS) / Intel; Linux | Test matrix, wheel choice, how much GPU-specific tuning we do | Tier 1: Windows + AMD/NVIDIA, Linux + AMD/NVIDIA; Tier 2: Apple Silicon (MPS), CPU-only everywhere |
| B3 | Do you have a **Mac** and/or an **NVIDIA machine** (or a friend who does) to test on? | Without a Mac we cannot verify macOS builds or MPS behaviour; CI can only run CPU tests there | CPU-only CI on macOS, no MPS claims |
| B4 | **Packaging model**: one big installer, or a small app that **downloads the matching PyTorch runtime** (CUDA / ROCm / MPS / CPU) on first run after detecting the hardware? | Installer size, first-run internet requirement, how "just works" it feels | Small app + on-demand runtime download (as ComfyUI/LM Studio-style tools do) |
| B5 | **Containers / a Linux instance inside the app** — you asked because of PaddleOCR. Measured: PaddleOCR's models run natively on Windows (PyTorch, no PaddlePaddle framework), so no Linux is needed; and containers are a bad fit for end users (Docker Desktop needs WSL2/Hyper-V, and there is **no Apple GPU passthrough**). Are you OK with native packaging instead? Docker stays as an optional server/dev path (B17) | Architecture of M13 | Native packaging, no embedded container |
| B6 | **App shell**: Qt (PySide6, my recommendation, embeds the existing web views via QWebEngine) vs Tauri/Electron around the web UI | UI rewrite cost, native feel, installer size | PySide6 |
| B7 | Should the **Qt shell skeleton start earlier** (in parallel with the pipeline) since "universal" is the goal, or stay after first end-to-end? | Risk of late platform surprises vs distraction | After first end-to-end, but with CPU CI from now on |
| B8 | **Code signing**: Windows certificate (avoids SmartScreen warnings) and an Apple Developer account (USD 99/yr, needed for notarised macOS apps). Budget/willingness? | Whether normal users can open the app without scary warnings | Unsigned builds, documented workaround |
| B9 | **Auto-update** via GitHub Releases acceptable? | Update UX, hosting | Manual download for now |
| B10 | **Minimum hardware** you want to support (VRAM, RAM) and is CPU-only "works but slow" acceptable? | Model choices (e.g. LaMa vs lighter inpaint), batch sizing | 8 GB VRAM recommended, CPU-only allowed but slow |
| B11 | **Local LLM for end users**: require Ollama to be installed (today), bundle an embedded llama.cpp-style server, or cloud-only? | "Everybody can run it" depends on this; translategemma:12b needs ~8 GB VRAM | Require Ollama (local or cloud); revisit at M13 |
| B12 | **Model distribution**: download from Hugging Face on first run (needs internet) or ship inside the installer? Disk budget (several GB) OK? | Installer size, offline use | First-run download with checksum verification |
| B13 | **Project license and repo visibility** (currently private). Open-source later? Which license? (I already avoid GPL/AGPL models.) | Dependency choices, contributor rules | Private, license undecided, GPL/AGPL avoided |
| B14 | **Telemetry**: none by default and never for page content; opt-in crash reports only — OK? | App design, privacy statement | No telemetry |

## C. Audience, legal and privacy

| ID | Question | Blocks / why it matters | Default until answered |
|---|---|---|---|
| C1 | **Audience**: personal use, your scanlation group, or a public tool? | LAN mode/auth (B16), disclaimers, support burden | Personal + small group |
| C2 | **Legal stance** — confirm: no DRM-platform acquisition (Naver Webtoon, Kakao, Lezhin, Ridibooks, Bomtoon, Kuaikan); tests never contain copyrighted raws; the app ships with a "you are responsible for the material you process" notice | Acquisition scope, docs | As stated |
| C3 | **Cloud LLM privacy**: is it OK to send OCR'd text of copyrighted works to Ollama Cloud models, or must there be a **local-only mode**? | Default profiles, judge choice | Local + cloud both available, cloud profiles enabled per config |
| C4 | **Budget**: any token/cost cap for cloud translation and judging per chapter or per series (Ollama Pro has rolling limits)? | Batch pacing, whether to pause instead of spend | Warn and fail loudly on rate limit; no cap |

## D. Translation and style

| ID | Question | Blocks / why it matters | Default until answered |
|---|---|---|---|
| D1 | **Honorifics**: keep `-nim / -ssi / hyung / noona / sunbae` romanised (current) or localise (Mr./Senior/Brother)? Per-series override? | Prompt text, glossary policy | Keep romanised |
| D2 | **Name romanisation**: Revised Romanization with hyphenated given names (Seong-jin) — or your preference? Family name first or last? | Glossary bootstrap, consistency across chapters | Revised Romanization, given-name hyphen, family name first as in the source |
| D3 | **Profanity / adult content**: keep as written, soften, or per-series? Cloud models may refuse some content — fall back to a local model automatically? | Judge/candidate selection, refusal handling | Keep as written; fall back to local on refusal |
| D4 | **SFX**: translate to English onomatopoeia, keep the original with a small caption, or full stylised replacement (M10)? What order of priority? | Whether SFX is skipped in v1 | Translate candidates only; typeset later |
| D5 | **Judge default**: run the cloud judge on every line, or only on lines where candidates disagree/flags exist (saves tokens)? | Cost, speed | Only on disagreements/flags |
| D6 | **Human review gates**: fully automatic, or a mandatory pause after the first chapter of a series for glossary approval, and before export? | Pipeline shape, UI | Pause after chapter 1 for glossary approval; otherwise automatic |
| D7 | **Reference mode**: auto-lock terms consistent in ≥ 3 reference chapters — right threshold? Whose translation counts as authoritative (official release vs fan)? | Glossary quality | 3 chapters, official release first |
| D8 | **Language order** after Korean: Chinese then Japanese (plan). Do you need Japanese vertical text / right-to-left soon? | C9/C10 scheduling | Korean → Chinese → Japanese |

## E. Output and typesetting

| ID | Question | Blocks / why it matters | Default until answered |
|---|---|---|---|
| E1 | **Output format**: keep webtoon slices (target ~3000 px), one continuous strip, or per-page? JPEG quality / PNG / WebP? CBZ or PDF as the default package? | Export stage, slice height | Slices of ~3000 px, JPEG q95, CBZ on request |
| E2 | **Fonts**: free OFL defaults, or your own (commercial) lettering fonts? Per-series font roles? | Typesetter look; fonts cannot be bundled if commercial | OFL defaults, your fonts pluggable |
| E3 | Can you show **2–3 reference pages of the look you want** (official English release style you like)? | Typesetting targets (size, stroke, alignment) | A neutral webtoon style |
| E4 | **Watermarks**: do you remove source-site watermarks? Which series have fixed-position ones? | B15 usage, C11 priority | Not removed |
| E5 | **Quality bar**: "indistinguishable from an official release" or "clean and readable"? | Inpainting model (LaMa vs diffusion), time per chapter | Clean and readable first, upgrade later |
| E6 | **Speed target**: minutes per chapter you can accept (overnight batches?) | Batch sizing, cloud vs local choices | No target yet |

## F. Process and scope

| ID | Question | Blocks / why it matters | Default until answered |
|---|---|---|---|
| F1 | Should the **hybrid GPU JPEG codec (C2)** come sooner? Evidence so far: CPU `turbo` ≈ 141–148 Mp/s, ~0.4 s / 50 pages | Whether a GPU decoder is worth its complexity | Later: detect/OCR first, profile a real chapter, then decide |
| F3 | How many **series / chapters** do you expect (dozens? hundreds?) | Queue/DB scaling, storage | Tens of series |
| F4 | What **manual edits** do you expect to make most (text, box size, font size, glossary)? | Editor (M8) priorities | Text + font size + glossary lock |
| F5 | Is **LAN / multi-editor mode** (B16: auth, edit log, page locks) needed soon? | Scheduling B13/B16 | Later |
| F6 | **Notifications**: desktop toast (with the app), Discord/webhook, or just logs? | Queue notifier defaults | Log + optional webhook |
| F7 | Builder cost policy: OK to use glm-5.3 (not flash) for genuinely hard cards, and when do you want the **big final verification pass**? | Token spend | Flash first; verification pass after the milestones are built |
| F8 | More **legal test material**: OK to keep using CC-licensed comics (Pepper&Carrot etc.) for non-Korean checks, or do you know CC-licensed Korean webcomics? | Test coverage until real raws arrive | Pepper&Carrot only |

## Decisions (answered)

| Date | Topic | Decision |
|---|---|---|
| 2026-09-18 | GPU | GPU end-to-end; CPU fallbacks only with benchmark evidence |
| 2026-09-18 | Builders | Flash-first (`glm-5.3-flash:cloud`), up to 2 concurrent; avoid deepseek/kimi as builders |
| 2026-09-18 | Product | Long-term: one universal desktop app (any OS, any GPU) with a built-in professional debug/review area (M13) |
| 2026-09-19 | Platform | The shipped app is one executable for all systems (Windows, macOS, Linux); development should not be tied to one OS |
| 2026-09-19 | Environment | Windows 11 native is the primary dev/run environment; the project lives in `V:\OmniScan` (worktrees in `V:\OmniScan-wt`); the WSL distro is retired |
| 2026-09-19 | Data location | Library / work / output live in `V:\OmniScan\data\` (gitignored), configured in `C:\Users\limex\.config\omniscan\config.toml` |
| 2026-09-19 | GitHub | `gh` is logged in on Windows as Nawid3333; PR-based review is possible (default stays: merge locally, push to `main`) |
| 2026-09-19 | GPU choice | `gpu.device = "auto"` picks the strongest discrete GPU (skips integrated GPUs); no hard-coded `cuda:0` |
| 2026-09-19 | Process | Open questions are kept in this file and asked at natural pauses; a checkpoint lives in `docs/CHECKPOINT.md` |

# Running OmniScan without a strong GPU — options and a recommendation

Written 2026-10-02 after the owner asked: "it should run on NVIDIA, Intel and AMD GPUs, and people without a
GPU could maybe use cloud tools; I'm not sure how that should look — explore it." This is the exploration.
Nothing here is built yet except what the first section lists; the decisions it needs are in
`docs/OPEN_QUESTIONS.md` (G1–G3).

## What already works today

| Need | State | Where |
|---|---|---|
| NVIDIA / AMD / Intel / Apple GPU | one torch backend extra per machine (`cuda`, `rocm-gfx1201`, `xpu`, `mps`); `gpu.device = "auto"` picks the strongest device the build can reach; `omniscan doctor` and `omniscan hardware` say what was found | `pyproject.toml`, `omniscan.gpu.device`, `omniscan.hw` |
| No GPU at all | the `cpu` extra: every stage runs, slowly; CI runs the whole CPU suite on Windows, macOS and Linux | `pyproject.toml`, `.github/workflows/ci.yml` |
| Translation without a local LLM | the default profile is already a cloud model (`gemma4:31b-cloud` through Ollama Cloud, reached via the local Ollama daemon, with a local fallback on rate limit); any Ollama model, local or cloud, is one profile table away | `config/translation_profiles.toml`, `omniscan.llm.ollama` |
| Translation with the page image | profiles with `images = true` send the page tiles to a vision model, so a cloud vision model already sees the art | `omniscan.translate.images` |
| Doing a chapter on another machine | a chapter's raws and all its work travel as one `.omniscan` project file (`omniscan project pack` / `unpack`, the Library's *Send chapter…*) | `omniscan.packaging` |
| A GPU box serving a browser | `omniscan serve` is an HTTP API + web Studio; a group can run it on the one PC with a GPU and edit from any browser on the LAN | `omniscan.web.app` |

So the GPU-vendor half of the question is answered by X1 of the roadmap and only needs real-hardware smoke
runs on NVIDIA, Intel and Apple machines. The open half is **where the heavy models run for a user whose PC
cannot run them well**.

## What is heavy, and what is not

Measured on the owner's machine and on CPU (`docs/benchmarks/`): the pipeline's cost sits in four places.

| Stage | Model | VRAM it wants | On a CPU | Could a cloud API do it? |
|---|---|---|---|---|
| detect | comic text-bubble detector (YOLO-class) | ~1 GB | seconds per page, fine | yes, but cheap enough locally |
| OCR | PP-OCRv5 det + rec, or PaddleOCR-VL (8 GB) | 0.5–8 GB | PP-OCR: seconds per page; PaddleOCR-VL: minutes | **yes**: a vision LLM reads a page crop; Ollama Cloud already serves them |
| translate + judge | an LLM (12B local or 31B cloud) | 9–21 GB local, 0 cloud | not usable (minutes per line) | **already cloud by default** |
| inpaint (LaMa) | big-lama | 2 GB | ~1 min per page at full resolution | yes (any GPU box running OmniScan), or skip: flat fills only |
| typeset / export | none (CPU rasteriser, GPU composite) | – | fine | not needed |

The two stages a weak PC really cannot do are the LLM (already cloud) and LaMa (slow but it finishes; the
`inpaint` stage's flat fills work without it, which is what a scanlation cleaner would then fix by hand in the
Studio).

## The options

### A. "Light" local mode (no new infrastructure)
CPU torch, PP-OCRv5 mobile models (0.5 GB, the fastest in the OCR qualification), cloud translation through
Ollama Cloud, LaMa off by default (flat fills, hand cleanup in the Studio). Everything exists; what is missing
is a *preset*: `omniscan doctor` / the first-run wizard recognising "no usable GPU" and setting
`ocr.engine`, `inpaint.lama = false` and the cloud profile in one step, plus honest time estimates
(`omniscan.pipeline.eta` already learns seconds per page per device).
Cost to the user: an Ollama account (free tier or Pro) for the cloud model, nothing else. Privacy: the OCR'd
text (and page tiles when `images = true`) goes to Ollama Cloud — question C3 already covers that.

### B. Cloud vision LLM as the OCR engine
Add an `ocr.engine = "vlm_api"` that sends each region crop (or page tile) to a vision model over Ollama's API —
the same client, the same `*-cloud` models — and parses the text back. OmniScan already has the crop readers
(`ocr/crop_readers.py`) and a local VLM engine (PaddleOCR-VL); a remote one is the same interface with an HTTP
call instead of a forward pass. With A + B a machine needs **no torch model bigger than the detector**, and
a CPU-only laptop finishes a chapter in minutes instead of hours.
Risk: cloud VLM OCR of Korean/Japanese is good but not qualified yet; it must go through the OCR qualification
suite (`omniscan eval`, `docs/benchmarks/ocr-qualification.md`) before it becomes a default. Cost: cloud tokens
per page (an image is ~1–2k tokens).

### C. A remote OmniScan worker (one GPU, many users)
`omniscan serve` already exposes the pipeline over HTTP, and the queue runs jobs. The missing piece is the
desktop app talking to a *remote* server instead of its in-process services: a `[remote] url` setting, and the
GUI's services (`omniscan.gui.services.*`) calling the API when it is set. A group then needs one GPU PC (or
one rented GPU VM) and everyone else runs the light app. This is also the shape of a hosted "OmniScan Cloud"
if the owner ever wants to offer one: the same server, behind accounts.
Cost: a GPU box, or a rented one (a cheap cloud GPU runs the whole pipeline; the app could rent it on demand
later). Work: a thin HTTP client behind the existing service layer, auth for anything beyond a LAN.

### D. Other providers' APIs (OpenAI-compatible, DeepL, Google)
Translation profiles today are Ollama-only. An `endpoint = "openai"` profile type (OpenAI-compatible
`/v1/chat/completions`: that is OpenAI, Mistral, Groq, OpenRouter, LM Studio, vLLM, …) would let a user bring
any key. DeepL/Google are cheaper per character but ignore the glossary, the story and the judge, so they only
make sense as a *candidate* profile the judge compares, never alone. Work: one more client class next to
`OllamaClient` implementing `ChatClient`; the prompts and parsing stay.

### E. Ship a bundled LLM runtime instead of requiring Ollama
Rejected for now: Ollama already runs on all three OSes and all vendors, serves the cloud models through the
same daemon, and the owner's benchmarks are all against it (question B11).

## Recommendation (in this order)
1. **A** now: it is configuration and a doctor/wizard preset, and it makes "works on any PC" true today.
2. **D** (OpenAI-compatible endpoint) right after: small, and it removes the single-vendor dependency for the
   one stage that is cloud by default.
3. **B** when the OCR qualification suite can measure it: it is what makes a CPU-only machine *fast*, not just
   possible.
4. **C** for groups, after the Studio is complete: the server exists; the client is the work.

What the owner needs to decide (G1–G3 in `docs/OPEN_QUESTIONS.md`): whether a hosted OmniScan service is
wanted at all or only "bring your own GPU/key"; which cloud providers to support first; and whether cloud OCR
of copyrighted pages is acceptable under the same terms as cloud translation (C3).

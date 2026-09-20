# GPU load: measurements, causes, fixes (owner report 2026-09-20: coil whine = no steady load)

Coil whine is the sound of the card's inductors when the power draw swings at audible frequencies: long idle gaps, then bursts, or thousands of tiny kernels with CPU work in between. So the goal is **a continuous, boring load**: no multi-second idle stalls, no CPU-bound phases while the GPU waits, and few sync points between batches. Measured on the RX 9070 XT (ROCm 10, Windows, fp32) with `scripts/gpu_duty.py` and the MIOpen log (`MIOPEN_LOG_LEVEL=6`).

## What the GPU does today (one Pepper&Carrot chapter, 11 pages, ~12 000 px strip; plain `omniscan run`, wall clock)
| Stage | wall | finding |
|---|---|---|
| ingest + slice | 1 s | CPU (JPEG decode) |
| detect | **15–17 s** | 3 forwards of **0.13 s** each = 0.4 s of real work; **the first convolution of every process costs 9–14 s** (see below) |
| ocr | 4 s | steady |
| inpaint (flat) | 1.5 s | steady |
| inpaint_lama | 1.3 s here (5 windows × 0.24 s); 20–25 s in a fresh process | model load + warm-up of the fixed window shape (FFT plans, conv init) |
| typeset | **5–7 s, GPU idle** | 96 % of it is `PIL.ImageFont.getlength` (33 000 calls, 155 µs each) in the wrap/fit search — pure CPU work that repeats the same measurements |
| export | 0.8 s | small |
So a chapter is ~30 s of wall time of which the GPU computes for only a few seconds: long idle stalls and CPU-bound phases, then short bursts.

**Do not trust a profiled run.** `scripts/gpu_duty.py` wraps the stages in `torch.profiler`; on this ROCm-Windows build a profiled render pass once produced a **corrupted export** (pages of the strip overlaid with shifted colour channels, 2026-09-20), while the same stages without the profiler, three times, were clean, and `decode_into` alone under the profiler and under heavy concurrent GPU load stayed bit-exact (15/15 each). The profiler's kernel/duty numbers are also inflated (40 s vs 1.3 s for the LaMa stage). Use the tool for the first-call table (`--ops`, wall clock only) and never keep the output of a profiled run.

## First-call stalls (once per process, GPU idle or bursting on tiny benchmark kernels)
`scripts/gpu_duty.py --ops`: first 3×3 convolution **9–14 s** (MIOpen loads the 311 MB Composable-Kernel grouped-conv library, `MIOpenCKGroupedConv_gfx1201.dll`, even for ungrouped convs; `MIOPEN_CK_LIB_PATH`, `MIOPEN_FIND_MODE` and the solver toggles do **not** avoid it), first `rfft2`/`irfft2` 1–3 s each (rocFFT plans, LaMa), first `bmm` 1.3 s (rocBLAS), first `conv_transpose2d` 1.4 s. The MIOpen find-db and kernel cache in `~/.miopen` work (lookups take 0.15 ms); the stall is library initialisation, not tuning. A background thread hides it: a warm-up thread ran the first conv for 10 s while the main thread kept doing CPU work at full speed, and the main thread's own first conv then took 0.01 s (`scratchpad/warm_thread.py`, 2026-09-20).

## What did not help
- **Batching LaMa windows** (B=4, B=8): 1 313 ms and 703 ms per window versus 238 ms at B=1 — MIOpen picks slower kernels for the batched shapes. LaMa at B=1 is already ~100 % busy with back-to-back short kernels.
- `MIOPEN_FIND_MODE=FAST|HYBRID`, `MIOPEN_CK_LIB_PATH=<empty dir>`, disabling the CK solvers by environment variable: same 9–14 s.
- The torch profiler on this build reports per-op device time but no kernel timeline, so gaps below 1 s can only be seen through the Windows GPU-engine counters (`\GPU Engine(*)\Utilization Percentage`, 1 s resolution).

## Fixes (cards)
1. **G2** — start the GPU library warm-up (conv, FFT, GEMM, transposed conv) on a background thread when a GPU pipeline starts, so the stalls overlap with ingest/JPEG decode/model loading; cache text measurements in the typeset fit loop (removes ~5 s of GPU idle per chapter).
2. **G3** — one-batch look-ahead in the detector/OCR loops: queue the next batch's forward pass before the previous batch's results are copied to the host, so the GPU never waits for CPU post-processing.
3. Sequential autoregressive readers (`manga_ocr` beam search, PaddleOCR-VL generation) are launch-bound by nature; O1d batches their generation.
4. A long-lived process (the desktop app, `omniscan queue`) pays the stalls once per session, not once per command.

## If the card still whines
Hardware coil whine also depends on the individual card. Adrenalin → Performance → Tuning: a small power limit (−10 … −15 %) or a fixed maximum clock usually removes it at a few percent of speed; a frame-rate cap does nothing for compute. Report which stage is loudest (`scripts/gpu_duty.py`) and it gets its own card.

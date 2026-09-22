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

## Cross-process GPU exclusivity (owner's driver crashed, 2026-09-22)
Two or three GLM builders run concurrently (`scripts/omni_builder.py`, up to `OMNI_SLOTS`), each in its own git worktree, but there is only one physical GPU. On 2026-09-22, while two builders and a director live-check were all touching the RX 9070 XT at once, one GPU test hit a hard, non-Python process crash (not reproducible on retry), and the owner's GPU driver crashed at the same time. Neither is proof by itself, but concurrent real-hardware access is a known stressor on this ROCm-Windows stack (see "What did not help" above and the earlier `torch.profiler`-corrupted-export finding), so it is now prevented structurally rather than by policy:

- **`omniscan.gpu.lock`** (`src/omniscan/gpu/lock.py`): `gpu_lock()` / `acquire_gpu_lock()` + `release_gpu_lock()` is an OS-level advisory lock on a fixed path in the shared temp directory — the same path from every worktree, so it is a true cross-process, cross-worktree mutex. It auto-releases if the holder exits or crashes (no stale-lock cleanup needed).
- **Wired in automatically**, not by task-card convention: `cli.py`'s `omniscan run` and the single-stage subcommands (`detect`/`ocr`/…), the queue executor (`queue/executor.py`), and every `@pytest.mark.gpu` test (`tests/conftest.py`) all take this lock before touching the real device and release it after. Two builders (or a builder and the director) can run concurrently as before — file-scope separation between cards still matters and each card's file allowlist should keep them non-overlapping — but only one of them is ever actually executing GPU code at a time; the other blocks and logs that it is waiting. `pytest -m "not gpu"` never touches the lock, so the CPU suite stays exactly as fast as before.
- Ad-hoc scripts that touch the real GPU outside these paths (`scripts/gpu_duty.py`, a one-off measurement script, a manual live check) should wrap the GPU-touching part in `with omniscan.gpu.lock.gpu_lock():` too — this is not yet retrofitted into every existing script.

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

## After G3 (idle gaps around the stages, 2026-09-22)
The same 8-stage command (fresh process, `scripts/measure_run.py`, timeline on) measured **before 44.6 / 44.9 / 45.0 s → after 30.8 / 33.0 / 34.3 s** (min/median 44.9 → 33.0). What the seconds go to now, and the mechanisms behind it (`omniscan.gpu.timeline` marks; never the profiler):

- **MIOpen's first convolution costs ~8.1 s of CPU-saturating work** (library load + kernel find). It holds **no GIL and no driver lock** — a plain Python loop runs at full speed during it — but anything that dispatches many small torch ops starves: the vision loader's ~2 200 dispatches go 1.7 s → 9.4 s, JPEG decode workers 0.04 s → 7.5–9 s. The harm is CPU scheduling, not GPU contention.
- **Non-contiguous H2D copies stall behind that load; contiguous ones do not.** `out[:, y:y+h, :].copy_(page)` launches copy **kernels**, each queued behind the MIOpen library load (11 pages ≈ 9 s); one pinned mirror of the whole strip + a single contiguous `copy_` takes the memcpy path and costs 0.00–0.06 s at any time (`turbo.decode_into`, G3).
- **Two concurrent MIOpen find phases stretch each other (17 s + 28 s vs 12 s + 15.5 s solo) but end at nearly the same wall time as serialising them.** A lock therefore only delays the start: holding a `miopen_lock` across the whole warm-up pushed LaMa's find from t≈3 to t≈17 and the wall from ~34 to 38 s. G3 ships **no lock** — LaMa's warm-up forwards (its find, the pipeline's critical path) start as early as possible behind the vision pass.
- **The find chain is the floor.** MIOpen library load ~8 s + the warm-up thread's remaining finds ~4.5 s + LaMa's window find ~15.5 s ≈ 28 s of mostly-CPU work that cannot be removed without changing kernel selection (forbidden: outputs must stay bit-identical). With ~3.5 s of tail stages and ~1.5 s of imports the wall floors at ~33 s — met at 30.8–34.3 s; the 25 s stretch would need a MIOpen kernel-db/pre-seed change (own card, with a bit-identicality check).
- **Evictions:** `VramManager` evicts a resident local Ollama LLM only when a load would leave less than `est + 4 GiB` free (`_EVICT_MARGIN_GIB`); a chapter without translate/judge now pays **0 s** for evictions instead of 2 × 2 s.

The prefetch architecture (G3): later model groups load on **one background worker in request order** (`VramManager.prefetch`), which waits for the **first `acquire`** — the same event the warm-up thread's gate waits on (`first_acquire_event`), so no MIOpen find overlaps the startup decode; the first group stays synchronous (prefetching it is net-negative: its loader competes with the decode); `_prefetch_groups` in `pipeline/runner.py` queues every group except the first and skips `ollama_local`. Failed prefetches fall back to the synchronous load; `cfg.gpu.warmup=false` switches the whole thing off.

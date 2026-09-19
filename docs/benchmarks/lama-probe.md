# LaMa inpainting probe (2026-09-19)

Question: can LaMa (large-mask inpainting, Apache-2.0) run on this machine (Windows, RX 9070 XT, torch 2.13 + ROCm 10), fast enough and good enough to clean text that sits on artwork?
Model: the TorchScript export `big-lama.pt` (205 669 692 bytes, sha256 `344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9`) from
`https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt` (the file IOPaint uses). Contract: `model(image, mask)` with `image` float `[1,3,H,W]` in 0..1, `mask` float `[1,1,H,W]` (1 = inpaint),
H and W multiples of 8; output float `[1,3,H,W]` in 0..1. Test: a 512×512 crop of a real painted page (Pepper&Carrot, CC BY 4.0) with "GET OUT!" (Bangers 64 px, white with a black stroke) drawn on it and a mask of the text grown by 9 px.

| Measurement | Result |
|---|---|
| Loads with `torch.jit.load(..., map_location=cuda)` | yes, 1.0 s; FFT ops (`_fft_r2c` / `_fft_c2r`) run on the GPU |
| fp32 forward, steady state | **30 ms per 512×512 crop** (profiler: 248 ms of summed GPU kernel time across streams, wall clock 30 ms) |
| Warm-up | the first calls **per new input shape** take 4–25 s (TorchScript profiling executor + MIOpen tuning); `MIOPEN_FIND_MODE=FAST` did not change it. Use ONE fixed window shape and warm it once per process |
| fp16 | fails in the TorchScript interpreter → fp32 only (consistent with `docs/DECISIONS.md`) |
| Quality on painted art | the text is removed completely and the brush texture / gradients are reconstructed almost identically to the ground truth; mean absolute difference to the original art inside the mask 13.7 (of 255); **outside the mask the output is bit-identical (0.0)** |

Decisions that follow: LaMa in fp32 with a fixed 512×512 window per region (windows larger than that are skipped in v1), warmed once per process; weights fetched on first use with the sha256 above checked
(model-management rule). Not measured: speech bubbles with gradients, very large SFX, 1000+ px windows (each new shape costs a 10–25 s warm-up).

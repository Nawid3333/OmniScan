# JPEG codec benchmarks — decode_into wall time, megapixels/s, GPU peak, CPU util

> **Correction (2026-09-18):** the first two runs below report CPU % inflated by a factor of `repeats` (3): the
> script divided CPU seconds accumulated over all repeats by the wall time of a single repeat. Their true values
> are ≈ 3249% / 3 = **1083%** and 3291% / 3 = **1097%**. Fixed in `scripts/bench_codec.py`; the 22:23 run is
> computed correctly. Machine: Ryzen 5 7600X, 12 threads — so ~1070–1100% means the CPU decode pool keeps roughly
> 10–11 of 12 threads busy for the whole run.

Run 2026-09-18 21:23 — synthetic 50 x (800x1400)

| backend | images | wall (s/decode) | Mp/s | peak GPU (MiB) | CPU % |
|---|---|---|---|---|---|
| turbo | 50 | 0.388 | 144.2 | 3 | 3249% |

Run 2026-09-18 21:24 — synthetic 50 x (800x1400)

| backend | images | wall (s/decode) | Mp/s | peak GPU (MiB) | CPU % |
|---|---|---|---|---|---|
| turbo | 50 | 0.379 | 147.9 | 3 | 3291% |

Run 2026-09-18 22:23 — synthetic 50 x (800x1400)

| backend | images | wall (s/decode) | Mp/s | peak GPU (MiB) | CPU % |
|---|---|---|---|---|---|
| turbo | 50 | 0.398 | 140.8 | 3 | 1072% |

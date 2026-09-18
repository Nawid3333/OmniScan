"""M0 GPU sanity check: device, fp16 matmul throughput, VRAM, pinned vs pageable H2D/D2H bandwidth."""

import sys
import time

import torch


def bench(fn, iters: int) -> float:
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def matmul_tflops(dev: torch.device, n: int) -> float:
    a = torch.randn(n, n, device=dev, dtype=torch.float16)
    b = torch.randn(n, n, device=dev, dtype=torch.float16)
    return 2 * n**3 / bench(lambda: a @ b, 20) / 1e12


def main() -> int:
    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  hip {torch.version.hip}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False")
        return 1
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    arch = getattr(props, "gcnArchName", "?")
    print(f"device: {props.name}  arch: {arch}  CUs: {props.multi_processor_count}")
    free, total = torch.cuda.mem_get_info(dev)
    print(f"VRAM free/total: {free / 2**30:.2f} / {total / 2**30:.2f} GiB")

    n = 8192
    print(f"fp16 matmul {n}x{n}: {matmul_tflops(dev, n):.1f} TFLOPS")

    size = 512 * 2**20  # 512 MiB, about one stitched chapter strip
    pageable = torch.empty(size, dtype=torch.uint8)
    pinned = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty(size, dtype=torch.uint8, device=dev)
    for name, host in (("pageable", pageable), ("pinned", pinned)):
        h2d = bench(lambda h=host: gpu.copy_(h, non_blocking=True), 5)
        d2h = bench(lambda h=host: h.copy_(gpu, non_blocking=True), 5)
        print(f"{name:8s} H2D {size / h2d / 1e9:6.1f} GB/s   D2H {size / d2h / 1e9:6.1f} GB/s")

    big = int(free * 0.85)
    blk = torch.empty(big, dtype=torch.uint8, device=dev)
    print(f"allocated {big / 2**30:.2f} GiB in one block: OK")
    del blk
    torch.cuda.empty_cache()
    print("GPU CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""G3 diagnosis probe: what does the first MIOpen conv3x3 call block in other threads?

Thread T runs the warm-up's first step (conv3x3 -> MIOpen library load + find, ~9-14 s). The main
thread probes: a pure-Python busy loop (GIL held?), a blocking pinned H2D copy (driver lock?) and a
pageable H2D copy, each timed during the conv call.
"""

import threading
import time

import torch
import torch.nn.functional as F  # noqa: N812 — torch's standard alias

from omniscan.gpu.device import resolve_device


def main() -> int:
    device = resolve_device("auto")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    started = threading.Event()

    def warm() -> None:
        image = torch.empty((8, 32, 320, 320), device=device, dtype=torch.float32)
        weight = torch.empty((64, 32, 3, 3), device=device, dtype=torch.float32)
        started.set()
        begin = time.perf_counter()
        F.conv2d(image, weight, padding=1)
        print(f"probe conv3x3: {time.perf_counter() - begin:.2f} s", flush=True)

    thread = threading.Thread(target=warm, daemon=True)
    thread.start()
    started.wait()
    time.sleep(0.2)  # let the conv reach its library load

    begin = time.perf_counter()
    total = 0
    for i in range(3_000_000):
        total += i
    print(f"probe python loop: {time.perf_counter() - begin:.2f} s (GIL held if >> 0.3 s)", flush=True)

    staging = torch.empty((3, 2000, 2000), dtype=torch.uint8, pin_memory=True)
    out = torch.empty((3, 2000, 2000), dtype=torch.uint8, device=device)
    begin = time.perf_counter()
    out.copy_(staging)
    print(f"probe pinned H2D: {time.perf_counter() - begin:.2f} s", flush=True)

    pageable = torch.empty((3, 2000, 2000), dtype=torch.uint8)
    begin = time.perf_counter()
    out.copy_(pageable)
    print(f"probe pageable H2D: {time.perf_counter() - begin:.2f} s", flush=True)

    begin = time.perf_counter()
    pinned2 = torch.empty((3, 15000, 2000), dtype=torch.uint8, pin_memory=True)
    print(f"probe 180 MiB pinned alloc: {time.perf_counter() - begin:.2f} s", flush=True)
    del pinned2
    thread.join(timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

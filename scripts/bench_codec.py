"""Benchmark JPEG codec backends (decode_into) against a chapter of images.

Usage: uv run python scripts/bench_codec.py [--chapter DIR] [--n 200] [--backend turbo]
Prints a markdown table to stdout and appends it to docs/benchmarks/codec.md.
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image

from omniscan.gpu.device import resolve_device

_REPO = Path(__file__).resolve().parent.parent
_BENCH_MD = _REPO / "docs" / "benchmarks" / "codec.md"
_MD_HEADER = "# JPEG codec benchmarks — decode_into wall time, megapixels/s, GPU peak, CPU util\n"


def _synthetic_chapter(n: int, width: int, height: int, dest: Path) -> None:
    """Write `n` synthetic JPEGs (flat blocks with a subtle grid, cheap to generate)."""
    for i in range(n):
        r = 40 + (i * 37) % 200
        g = 30 + (i * 53) % 200
        b = 20 + (i * 71) % 200
        img = Image.new("RGB", (width, height), (r, g, b))
        for y in range(0, height, 100):
            for x in range(0, width, 150):
                img.paste((r // 3, g // 3, b // 3), (x, y, min(x + 40, width), y + 20))
        img.save(dest / f"{i:04d}.jpg", format="JPEG", quality=92)


def _load_chapter(chapter: Path) -> list[bytes]:
    """Read all JPEGs of a chapter directory into memory."""
    return [p.read_bytes() for p in sorted(chapter.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg")]


def _build_codec(name: str, device: str | torch.device):
    """Build the named codec on `device`, or None if the backend isn't implemented yet."""
    if name == "turbo":
        from omniscan.gpu.codec.turbo import TurboCodec

        return TurboCodec(device)
    return None


def _strip_info(datas: list[bytes]) -> tuple[list[int], int]:
    """Per-image heights and the common width (header reads only)."""
    heights: list[int] = []
    width = 0
    for data in datas:
        with Image.open(io.BytesIO(data)) as img:
            h, w = img.size[1], img.size[0]
        heights.append(h)
        width = width or w
    return heights, width


def _cpu_time_used() -> float:
    """Process CPU seconds so far (user + system, summed over all threads; works on every OS)."""
    return time.process_time()


def _bench_one(datas: list[bytes], warmup: int, repeats: int, codec) -> dict[str, float]:
    """Time `repeats` decode_into runs over the whole batch; return aggregated metrics."""
    heights, width = _strip_info(datas)
    out = torch.empty((3, sum(heights), width), dtype=torch.uint8, device=codec.device)
    offsets: list[int] = []
    y = 0
    for h in heights:
        offsets.append(y)
        y += h
    total_px = sum(h * width for h in heights)
    is_cuda = codec.device.type == "cuda"
    base_mem = 0.0
    for _ in range(warmup):
        codec.decode_into(datas, out, offsets)
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
    cpu_before = _cpu_time_used()
    t0 = time.perf_counter()
    for _ in range(repeats):
        codec.decode_into(datas, out, offsets)
        if is_cuda:
            torch.cuda.synchronize()
    total_wall = time.perf_counter() - t0
    wall = total_wall / repeats
    # CPU seconds accumulated over ALL repeats must be divided by the wall time of ALL repeats (dividing by the
    # per-repeat wall inflated this by a factor of `repeats`, e.g. ~3300% on a 12-thread machine).
    cpu_pct = ((_cpu_time_used() - cpu_before) / total_wall * 100.0) if total_wall > 0 else 0.0
    peak_mib = ((torch.cuda.max_memory_allocated() - base_mem) / (1024 * 1024)) if is_cuda else 0.0
    return {"wall": wall, "mps": total_px / wall / 1e6, "peak_mib": peak_mib, "cpu_pct": cpu_pct}


def _table(rows: list[tuple[str, dict[str, float]]], n_images: int, label: str) -> str:
    """Render the markdown table for one run."""
    lines = [
        f"Run {time.strftime('%Y-%m-%d %H:%M')} — {label}",
        "",
        "| backend | images | wall (s/decode) | Mp/s | peak GPU (MiB) | CPU % |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in rows:
        lines.append(
            f"| {name} | {n_images} | {m['wall']:.3f} | {m['mps']:.1f} | {m['peak_mib']:.0f} | {m['cpu_pct']:.0f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the benchmark and print/append the markdown table."""
    parser = argparse.ArgumentParser(description="Benchmark JPEG codec backends (decode_into).")
    parser.add_argument("--chapter", type=Path, default=None, help="folder of JPEGs to decode")
    parser.add_argument("--n", type=int, default=200, help="synthetic image count when no --chapter")
    parser.add_argument("--width", type=int, default=800, help="synthetic image width")
    parser.add_argument("--height", type=int, default=1400, help="synthetic image height")
    parser.add_argument("--backend", action="append", default=None, help="backend name (repeatable)")
    parser.add_argument("--warmup", type=int, default=1, help="warmup decodes before timing")
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats, averaged")
    args = parser.parse_args()

    tmp: tempfile.TemporaryDirectory | None = None
    try:
        if args.chapter is None:
            tmp = tempfile.TemporaryDirectory()
            chapter = Path(tmp.name)
            _synthetic_chapter(args.n, args.width, args.height, chapter)
            label = f"synthetic {args.n} x ({args.width}x{args.height})"
        else:
            chapter = args.chapter
            label = f"chapter {chapter}"
        datas = _load_chapter(chapter)
        if not datas:
            raise SystemExit(f"no JPEGs found in {chapter}")

        rows: list[tuple[str, dict[str, float]]] = []
        for name in args.backend or ["turbo"]:
            codec = _build_codec(name, resolve_device())
            if codec is None:
                print(f"notice: backend {name!r} not implemented yet, skipping", file=sys.stderr)
                continue
            print(f"benchmarking {name} on {codec.device} over {len(datas)} images ({label}) ...")
            rows.append((name, _bench_one(datas, args.warmup, args.repeats, codec)))
            codec.close()
        if not rows:
            raise SystemExit("no runnable backends")

        table = _table(rows, len(datas), label)
        print(table)
        _BENCH_MD.parent.mkdir(parents=True, exist_ok=True)
        if not _BENCH_MD.exists():
            _BENCH_MD.write_text(_MD_HEADER)
        with _BENCH_MD.open("a") as f:
            f.write("\n" + table)
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()

"""Generate a synthetic demo chapter (noisy "art" panels + white gutters) so the pipeline and web viewer can be
tried without real raws.

    uv run python scripts/make_demo_chapter.py                 # into the configured library_root
    uv run python scripts/make_demo_chapter.py --library-root /tmp/lib --series Demo --chapter "Chapter 1"

Pages: three 800px-wide pages (one saved as PNG to exercise the conversion cache) and one 720px-wide page (to
exercise off-width resizing). Every panel row is random noise, so the slicer can only cut inside the gutters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from omniscan.core.config import get_config


def make_page(rng: np.random.Generator, width: int, height: int = 1400) -> Image.Image:
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    y = 90  # white gutter on top
    for panel_h, gutter in ((500, 120), (400, 100), (150, 0)):
        arr[y : y + panel_h] = rng.integers(0, 256, (panel_h, width, 3), dtype=np.uint8)
        y += panel_h + gutter
    return Image.fromarray(arr, "RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--library-root", type=Path, default=None, help="default: paths.library_root from config"
    )
    parser.add_argument("--series", default="DemoSeries")
    parser.add_argument("--chapter", default="Chapter 1")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    library_root = args.library_root or get_config().paths.library_root
    chapter_dir = library_root / args.series / args.chapter
    chapter_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    make_page(rng, 800).save(chapter_dir / "001.jpg", quality=92)
    make_page(rng, 800).save(chapter_dir / "002.jpg", quality=92)
    make_page(rng, 800).save(chapter_dir / "003.png")
    make_page(rng, 720).save(chapter_dir / "004.jpg", quality=92)
    print(f"wrote 4 pages to {chapter_dir}")
    print(f"next: omniscan slice {args.series!r} && omniscan serve   (then `npm run dev` in webui/)")


if __name__ == "__main__":
    main()

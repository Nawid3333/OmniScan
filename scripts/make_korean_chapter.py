"""Generate a synthetic Korean demo chapter with exact ground truth (card X1).

    uv run python scripts/make_korean_chapter.py                 # into the configured library_root
    uv run python scripts/make_korean_chapter.py --library-root /tmp/lib --truth-out truth.json

Pages: 800x1400 webtoon pages with Korean speech bubbles, free-standing text and SFX, deterministic
per seed. `--truth-out` additionally writes one `RegionsArtifact` JSON object per page into a list.
`--scan-artifacts` degrades every page like a scan (blur, noise, halftone, skew, JPEG).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # direct script run: make omniscan/ and tests/ importable
    sys.path.insert(0, str(_REPO_ROOT))

from tests.fixtures.korean_pages import make_korean_page, to_regions_artifact  # noqa: E402
from tests.fixtures.scan_artifacts import degrade  # noqa: E402

from omniscan.core.config import get_config  # noqa: E402


def write_page(image: Image.Image, path: Path) -> None:
    """Save one page as quality-92 JPEG."""
    image.save(path, format="JPEG", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--library-root", type=Path, default=None, help="default: paths.library_root from config"
    )
    parser.add_argument("--series", default="KoreanDemo")
    parser.add_argument("--chapter", default="Chapter 1")
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--truth-out",
        type=Path,
        default=None,
        help="write one RegionsArtifact JSON object per page (a list) to this file",
    )
    parser.add_argument(
        "--scan-artifacts",
        action="store_true",
        help="degrade every page like a scan (blur, noise, halftone, skew, JPEG) before writing it;"
        " --truth-out is unchanged and still describes the clean page (rotation makes it approximate)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=60,
        help="JPEG quality of the scan degradation (with --scan-artifacts)",
    )
    parser.add_argument(
        "--noise", type=float, default=4.0, help="Gaussian noise sigma of the scan degradation"
    )
    parser.add_argument(
        "--blur", type=float, default=0.6, help="Gaussian blur radius in px of the scan degradation"
    )
    parser.add_argument("--halftone", action="store_true", help="add a halftone dot screen")
    parser.add_argument(
        "--rotate", type=float, default=0.0, help="page skew in degrees of the scan degradation"
    )
    args = parser.parse_args()

    library_root = args.library_root or get_config().paths.library_root
    chapter_dir = library_root / args.series / args.chapter
    chapter_dir.mkdir(parents=True, exist_ok=True)

    truths = []
    for i in range(1, args.pages + 1):
        page = make_korean_page(args.seed + i, n_bubbles=4, n_free=1, n_sfx=1 if i % 2 == 1 else 0)
        image: Image.Image = page.image
        if args.scan_artifacts:
            image = degrade(
                image,
                seed=args.seed + i,
                blur_px=args.blur,
                noise_sigma=args.noise,
                halftone=args.halftone,
                rotate_deg=args.rotate,
                jpeg_quality=args.jpeg_quality,
            )
        write_page(image, chapter_dir / f"{i:03d}.jpg")
        truths.append(to_regions_artifact(page))
        print(f"wrote page {i:03d}.jpg with {len(page.regions)} regions to {chapter_dir}")

    if args.truth_out is not None:
        args.truth_out.parent.mkdir(parents=True, exist_ok=True)
        args.truth_out.write_text(
            json.dumps([json.loads(t.model_dump_json()) for t in truths], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote ground truth for {len(truths)} pages to {args.truth_out}")

    print(f"next: omniscan slice {args.series!r}  (pass --truth-out to also get the OCR ground truth)")


if __name__ == "__main__":
    main()

"""Generate a synthetic Korean demo chapter with exact ground truth (card X1).

    uv run python scripts/make_korean_chapter.py                 # into the configured library_root
    uv run python scripts/make_korean_chapter.py --library-root /tmp/lib --truth-out truth.json

Pages: 800x1400 webtoon pages with Korean speech bubbles, free-standing text and SFX, deterministic
per seed. `--truth-out` additionally writes one `RegionsArtifact` JSON object per page into a list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # direct script run: make omniscan/ and tests/ importable
    sys.path.insert(0, str(_REPO_ROOT))

from tests.fixtures.korean_pages import KoreanPage, make_korean_page, to_regions_artifact  # noqa: E402

from omniscan.core.config import get_config  # noqa: E402


def write_page(page: KoreanPage, path: Path) -> None:
    """Save one page as quality-92 JPEG."""
    page.image.save(path, format="JPEG", quality=92)


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
    args = parser.parse_args()

    library_root = args.library_root or get_config().paths.library_root
    chapter_dir = library_root / args.series / args.chapter
    chapter_dir.mkdir(parents=True, exist_ok=True)

    truths = []
    for i in range(1, args.pages + 1):
        page = make_korean_page(args.seed + i, n_bubbles=4, n_free=1, n_sfx=1 if i % 2 == 1 else 0)
        write_page(page, chapter_dir / f"{i:03d}.jpg")
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

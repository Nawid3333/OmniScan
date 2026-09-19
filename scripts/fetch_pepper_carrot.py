"""Fetch Pepper&Carrot test data: real lettered pages plus exact ground truth (legal: CC BY 4.0, David Revoy, peppercarrot.com).

For a language code (`kr`, `cn`, `ja`, ...) and a range of episodes it downloads

    raws/<Series>/Episode NN/PP.jpg                                the lettered pages in that language (Series = PepperCarrot<LANG>)
    translated-check/<Series>/Episode NN/english/PP.jpg            the English original of the same pages
    translated-check/<Series>/Episode NN/text-free/PP.jpg         the same art without any text (not pixel-aligned with the lettered pages)
    translated-check/<Series>/Episode NN/truth/<lang>/EnnPpp.svg   the text layer (boxes + lines) in that language, plus info.json (translator credits)
    translated-check/<Series>/Episode NN/truth/en/EnnPpp.svg       the same in English

Usage (from the repo root):
    uv run python scripts/fetch_pepper_carrot.py --lang kr --episodes 1-32
    uv run python scripts/fetch_pepper_carrot.py --lang cn --episodes 6,10

Existing non-empty files are skipped, so it is safe to re-run. Roots come from the machine config (`paths.library_root`, and
`translated-check` next to it). Please keep the request rate polite (the script sleeps between requests).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from omniscan.core.config import get_config  # noqa: E402

BASE = "https://www.peppercarrot.com/0_sources"
USER_AGENT = "OmniScan-test-data-fetch/1.0 (personal research; CC BY 4.0 sources)"
PAUSE_S = 0.15


def fetch(url: str) -> bytes | None:
    """GET a URL; None on 404 (a missing optional file), other errors propagate."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    finally:
        time.sleep(PAUSE_S)


def save(url: str, path: Path) -> bool:
    """Download `url` to `path` unless a non-empty file is already there; True when the file exists afterwards."""
    if path.is_file() and path.stat().st_size > 0:
        return True
    data = fetch(url)
    if data is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def parse_episodes(spec: str) -> list[int]:
    """'1-5,8' -> [1, 2, 3, 4, 5, 8]."""
    numbers: list[int] = []
    for part in spec.split(","):
        first, _, last = part.partition("-")
        numbers.extend(range(int(first), int(last or first) + 1))
    return sorted(set(numbers))


def episode_folders() -> dict[int, str]:
    """Episode number -> source folder name (e.g. 6 -> 'ep06_The-Potion-Contest')."""
    listing = (fetch(f"{BASE}/") or b"").decode("utf-8", "replace")
    return {int(m.group(1)): m.group(0) for m in re.finditer(r'ep(\d+)_[^/"]+', listing)}


def fetch_episode(
    folder: str, number: int, lang: str, series: str, raws: Path, check: Path
) -> tuple[int, int]:
    """Download one episode; returns (pages in `lang`, pages that also got the English + text-free versions)."""
    low = f"{BASE}/{folder}/low-res"
    listing = (fetch(f"{low}/") or b"").decode("utf-8", "replace")
    tag = f"E{number:02d}"
    pages = sorted(set(re.findall(rf"{lang}_Pepper-and-Carrot_by-David-Revoy_{tag}P(\d\d)\.jpg", listing)))
    if not pages:
        return 0, 0
    chapter = f"Episode {number:02d}"
    complete = 0
    for page in pages:
        name = f"Pepper-and-Carrot_by-David-Revoy_{tag}P{page}"
        got = save(f"{low}/{lang}_{name}.jpg", raws / series / chapter / f"{page}.jpg")
        english = save(f"{low}/en_{name}.jpg", check / series / chapter / "english" / f"{page}.jpg")
        textfree = save(
            f"{low}/gfx-only/gfx_{name}.jpg", check / series / chapter / "text-free" / f"{page}.jpg"
        )
        for code in (lang, "en"):
            save(
                f"{BASE}/{folder}/lang/{code}/{tag}P{page}.svg",
                check / series / chapter / "truth" / code / f"{tag}P{page}.svg",
            )
        complete += int(got and english and textfree)
    for code in (lang, "en"):
        save(
            f"{BASE}/{folder}/lang/{code}/info.json", check / series / chapter / "truth" / code / "info.json"
        )
    return len(pages), complete


def main() -> None:
    """Parse arguments and download the requested episodes."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lang", default="kr", help="language code of the lettered pages (kr, cn, ja, ...)")
    parser.add_argument(
        "--episodes", default="1-32", help="e.g. 1-32 or 6,10,25 (episodes without that language are skipped)"
    )
    args = parser.parse_args()

    raws = get_config().paths.library_root
    check = raws.parent / "translated-check"
    series = f"PepperCarrot{args.lang.upper()}"
    folders = episode_folders()
    total = 0
    for number in parse_episodes(args.episodes):
        if number not in folders:
            print(f"episode {number:02d}: not found on the server")
            continue
        pages, complete = fetch_episode(folders[number], number, args.lang, series, raws, check)
        total += pages
        print(
            f"episode {number:02d}: {pages} {args.lang} pages ({complete} with English + text-free)",
            flush=True,
        )
    print(f"done: {total} pages in {raws / series}")


if __name__ == "__main__":
    main()

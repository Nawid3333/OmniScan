"""Print a ready `[[model]]` TOML block for one Hugging Face repo (card O1a).

Queries the Hugging Face API for the repo metadata (`?blobs=true`: licence, languages, head
revision) and the file tree at the pinned revision (`?recursive=true`: sizes, `lfs.oid` sha256 for
LFS files), downloads and hashes every small non-LFS model file, and prints a `[[model]]` block
with `format = "hf"` to stdout. The block is validated against the catalog's ModelEntry before it
is printed; the script never writes files.

Usage:
    uv run python scripts/hf_catalog.py REPO [--id ID --role ROLE --family F --size-class S --revision SHA]

Only top-level files with model-relevant suffixes are listed (.safetensors/.bin weights, .json
configs, .txt/.model tokenizers, .jinja chat templates) — README, .gitattributes and images are
excluded, and so is every subfolder file (PaddleOCR-VL's PP-DocLayoutV2/ for example). `size_mb`
is the sum of the listed files rounded up (decimal MB, same convention as the catalog).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tomllib
from collections.abc import Callable

import httpx

API_BASE = "https://huggingface.co/api/models"
RESOLVE_BASE = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
USER_AGENT = "omniscan-hf-catalog/0.1 (OmniScan model catalog helper)"

# Suffixes of the files a loader needs: weights (.safetensors/.bin, .gguf for llama.cpp builds),
# configs, tokenizers. Everything else (README, .gitattributes, images) is skipped, as is any file
# inside a subfolder.
INCLUDE_SUFFIXES = (".bin", ".gguf", ".jinja", ".json", ".model", ".safetensors", ".tiktoken", ".txt")

Fetch = Callable[[str], bytes]  # URL -> response body


class FetchError(RuntimeError):
    """A Hugging Face request failed (HTTP status, network error or unusable payload)."""


def fetch_http(url: str) -> bytes:
    """GET `url` with a User-Agent and return the body; raise FetchError on any failure."""
    try:
        response = httpx.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60.0)
    except httpx.HTTPError as exc:
        raise FetchError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code != 200:
        raise FetchError(f"HTTP {response.status_code} for {url}")
    return response.content


def metadata_url(repo: str) -> str:
    """The repo metadata endpoint (licence, languages, head sha, sibling sizes)."""
    return f"{API_BASE}/{repo}?blobs=true"


def tree_url(repo: str, revision: str) -> str:
    """The per-file tree endpoint at `revision` (sizes, LFS oids)."""
    return f"{API_BASE}/{repo}/tree/{revision}?recursive=true"


def _json(fetch: Fetch, url: str) -> object:
    try:
        return json.loads(fetch(url))
    except ValueError as exc:
        raise FetchError(f"invalid JSON from {url}: {exc}") from exc


def collect_files(repo: str, revision: str, *, fetch: Fetch) -> dict[str, tuple[str, int]]:
    """`path -> (sha256, size)` for every top-level model file at `revision`.

    LFS files take their sha256 from `lfs.oid` (no download); smaller files are downloaded and
    hashed. Raises FetchError when nothing model-relevant remains after filtering.
    """
    tree = _json(fetch, tree_url(repo, revision))
    if not isinstance(tree, list):
        raise FetchError(f"{tree_url(repo, revision)}: expected a JSON list of files")
    files: dict[str, tuple[str, int]] = {}
    for item in tree:
        if not isinstance(item, dict) or item.get("type") != "file":
            continue
        path = str(item.get("path") or "")
        if "/" in path or not path.endswith(INCLUDE_SUFFIXES):
            continue  # subfolder files, README, .gitattributes and images are not model files
        lfs = item.get("lfs") or {}
        oid = str(lfs.get("oid") or "")
        if oid:
            sha256, size = oid.lower(), int(item.get("size") or lfs.get("size") or 0)
        else:
            data = fetch(RESOLVE_BASE.format(repo=repo, revision=revision, path=path))
            sha256, size = hashlib.sha256(data).hexdigest(), len(data)
        files[path] = (sha256, size)
    if not files:
        raise FetchError(f"{repo}@{revision}: no model files found (top-level weights/configs)")
    return files


def _slug(repo: str) -> str:
    """`org/PP-OCRv6_small_rec_safetensors` -> `pp-ocrv6-small-rec-safetensors` (default catalog id)."""
    name = repo.rsplit("/", 1)[-1].removesuffix("_safetensors")
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"


def _humanize(repo: str) -> str:
    """`org/PP-OCRv6_small_rec_safetensors` -> `PP-OCRv6 small rec` (default display name)."""
    name = repo.rsplit("/", 1)[-1].removesuffix("_safetensors")
    return name.replace("_", " ").strip() or repo


def build_block(
    repo: str,
    *,
    model_id: str,
    role: str | None,
    family: str | None,
    size_class: str | None,
    revision: str | None,
    fetch: Fetch,
) -> str:
    """The full `[[model]]` TOML block for `repo` pinned at `revision` (default: the head sha)."""
    meta = _json(fetch, metadata_url(repo))
    if not isinstance(meta, dict):
        raise FetchError(f"{metadata_url(repo)}: expected a JSON object")
    revision = revision or str(meta.get("sha") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise FetchError(f"{repo}: no 40-hex revision (head sha {meta.get('sha')!r})")
    card = meta.get("cardData") or {}
    if not isinstance(card, dict):
        card = {}
    license_name = str(card.get("license") or "")
    language = card.get("language") or []
    langs = [language] if isinstance(language, str) else [str(code) for code in language]
    files = collect_files(repo, revision, fetch=fetch)
    size_mb = math.ceil(sum(size for _, size in files.values()) / 1_000_000)
    lines = [
        "[[model]]",
        f"id = {json.dumps(model_id)}",
        f"name = {json.dumps(_humanize(repo))}",
        f"kind = {json.dumps('vision' if role == 'detector' else 'ocr')}",
        "required = false",
        'format = "hf"',
        f"size_mb = {size_mb}",
        f"license = {json.dumps(license_name)}",
        f"description = {json.dumps(f'{repo} (generated by scripts/hf_catalog.py; edit this description)')}",
        "used_by = []",
    ]
    if role is not None:
        lines.append(f"role = {json.dumps(role)}")
    if family is not None:
        lines.append(f"family = {json.dumps(family)}")
    if size_class is not None:
        lines.append(f"size_class = {json.dumps(size_class)}")
    lines.append("langs = [" + ", ".join(json.dumps(code) for code in langs) + "]")
    lines.append(f"upstream_repo = {json.dumps(repo)}")
    lines.append(f"upstream_revision = {json.dumps(revision)}")
    lines += ["", "[model.files]"]
    lines += [f"{json.dumps(path)} = {json.dumps(sha256)}" for path, (sha256, _) in sorted(files.items())]
    return "\n".join(lines) + "\n"


def validate_block(block: str) -> None:
    """Parse the block and check the catalog would accept it — the script never prints junk."""
    from omniscan.models.catalog import ModelEntry  # lazy: only needed when a block is built

    table = tomllib.loads(block)["model"][0]
    ModelEntry(**table).validate_for_format()


def main(argv: list[str] | None = None, *, fetch: Fetch | None = None) -> int:
    """CLI entry point; returns the process exit code (0 printed, 1 on any error)."""
    parser = argparse.ArgumentParser(
        description="Print a ready [[model]] TOML block for a Hugging Face repo."
    )
    parser.add_argument("repo", help="Hugging Face repo id, e.g. PaddlePaddle/PP-OCRv6_small_rec_safetensors")
    parser.add_argument(
        "--id", dest="model_id", help=f"catalog id (default: {_slug('org/REPO_NAME')!r} style)"
    )
    parser.add_argument("--role", help="catalog role (detector|text_line_detector|recognizer|vlm_ocr)")
    parser.add_argument("--family", help="catalog family (ppocrv6|ppocrv5|paddleocr-vl|manga-ocr)")
    parser.add_argument(
        "--size-class", dest="size_class", help="catalog size_class (tiny|small|medium|server|mobile|base)"
    )
    parser.add_argument("--revision", help="commit sha to pin (default: the repo's current head sha)")
    args = parser.parse_args(argv)
    try:
        block = build_block(
            args.repo,
            model_id=args.model_id or _slug(args.repo),
            role=args.role,
            family=args.family,
            size_class=args.size_class,
            revision=args.revision,
            fetch=fetch or fetch_http,
        )
        validate_block(block)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(block, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

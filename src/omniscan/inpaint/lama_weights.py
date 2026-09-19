"""LaMa weights on disk: fetch `big-lama.pt` once, verify its sha256, replace bad files atomically."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from omniscan.core.config import InpaintConfig

_CHUNK = 1 << 20  # 1 MiB: streamed download and hashing


def lama_path(models_dir: Path, cfg: InpaintConfig) -> Path:
    """Path of the LaMa TorchScript weights: `<models_dir>/lama/<cfg.lama_file>`."""
    return models_dir / "lama" / cfg.lama_file


def sha256_file(path: Path) -> str:
    """Lowercase hex sha256 of a file, streamed (the weights are 205 MB)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_lama_weights(models_dir: Path, cfg: InpaintConfig, *, client: httpx.Client | None = None) -> Path:
    """The verified weights file under `models_dir`, downloaded from `cfg.lama_url` on first use.

    An existing file with the expected sha256 is returned untouched; a missing file or one with a stale
    hash is downloaded into `<file>.part` and atomically renamed over the target. `client` is injectable
    for tests; a private one is created (and closed) when None.
    """
    path = lama_path(models_dir, cfg)
    if path.is_file() and sha256_file(path) == cfg.lama_sha256:
        return path
    http = client if client is not None else httpx.Client(follow_redirects=True, timeout=None)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.parent / (path.name + ".part")
        digest = hashlib.sha256()
        with http.stream("GET", cfg.lama_url) as response:
            if response.status_code != 200:
                raise RuntimeError(f"LaMa download failed: HTTP {response.status_code}")
            with part.open("wb") as fh:
                for chunk in response.iter_bytes(_CHUNK):
                    digest.update(chunk)
                    fh.write(chunk)
        actual = digest.hexdigest()
        if actual != cfg.lama_sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"LaMa weights checksum mismatch: expected {cfg.lama_sha256} got {actual}")
        part.replace(path)
        return path
    finally:
        if client is None:
            http.close()

"""Scanner-style degradation of a page image (card X2).

One deterministic function that turns a clean page into what a real raw looks like: a little blur,
per-pixel Gaussian noise, an optional halftone dot screen, a slight skew and a JPEG round trip.
Steps always run in this order (blur, noise, halftone, rotation, JPEG) so the result looks like one
scan rather than five stacked filters. Pure function of its arguments: the same call always yields
identical pixels, and the input image is never modified. CPU only.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

_DOT_PERIOD = 6.0  # halftone dot grid period in pixels
_DOT_DEPTH = 0.08  # halftone darkening depth (multiplier dips to 1 - _DOT_DEPTH)


def degrade(
    image: Image.Image,
    seed: int,
    *,
    blur_px: float = 0.6,
    noise_sigma: float = 4.0,
    halftone: bool = False,
    rotate_deg: float = 0.0,
    jpeg_quality: int = 60,
) -> Image.Image:
    """Return an RGB scan-degraded copy of `image` (same size), deterministic per `seed`."""
    if not 1 <= jpeg_quality <= 100:
        raise ValueError(f"jpeg_quality must be in 1..100, got {jpeg_quality}")
    if blur_px < 0:
        raise ValueError(f"blur_px must be >= 0, got {blur_px}")
    if noise_sigma < 0:
        raise ValueError(f"noise_sigma must be >= 0, got {noise_sigma}")

    page: Image.Image = image.convert("RGB")
    if blur_px > 0:
        page = page.filter(ImageFilter.GaussianBlur(radius=blur_px))
    arr = np.asarray(page, dtype=np.float64)
    if noise_sigma > 0:
        rng = np.random.Generator(np.random.PCG64(seed))
        arr = arr + rng.normal(0.0, noise_sigma, arr.shape)
    if halftone:
        x = np.arange(arr.shape[1], dtype=np.float64)
        y = np.arange(arr.shape[0], dtype=np.float64)
        d = (
            0.5
            + 0.5
            * np.sin(2.0 * np.pi * x / _DOT_PERIOD)[None, :]
            * np.sin(2.0 * np.pi * y / _DOT_PERIOD)[:, None]
        )
        arr = arr * (1.0 - _DOT_DEPTH * d[:, :, None])
    if noise_sigma > 0 or halftone:
        arr = np.clip(np.round(arr), 0.0, 255.0)
    page = Image.fromarray(arr.astype(np.uint8), "RGB")
    if rotate_deg != 0.0:
        page = page.rotate(
            rotate_deg, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255, 255, 255)
        )
    buffer = io.BytesIO()
    page.save(buffer, format="JPEG", quality=jpeg_quality)
    return Image.open(buffer).convert("RGB")

"""The pixels of a hand cleanup patch: pure numpy/OpenCV over strip-space crops (no torch, no I/O).

These run once per brush stroke the user applies, on a crop the size of the stroke — interactive editing,
not a pipeline hot path; the pipeline's own cleaning stays on the GPU (inpaint/, LaMa).
"""

from __future__ import annotations

import cv2
import numpy as np

from omniscan.core.schemas import RGB

RING_PX = 3  # width of the band around a mask whose median colour an automatic fill takes
INPAINT_RADIUS = 5  # OpenCV inpainting neighbourhood (px)


def ring(mask: np.ndarray, width: int = RING_PX) -> np.ndarray:
    """Bool mask of the band `width` px wide just outside `mask` (same shape)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    grown = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    return grown & ~mask


def ring_color(crop: np.ndarray, mask: np.ndarray) -> RGB:
    """Median colour of the band around `mask` in `crop` (uint8 [h, w, 3]); white when there is no band."""
    band = ring(mask)
    if not band.any():
        return (255, 255, 255)
    r, g, b = (int(v) for v in np.median(crop[band], axis=0))
    return (r, g, b)


def fill_pixels(shape: tuple[int, int], color: RGB) -> np.ndarray:
    """uint8 [h, w, 3] of one colour."""
    pixels = np.empty((shape[0], shape[1], 3), dtype=np.uint8)
    pixels[:] = color
    return pixels


def inpaint_pixels(context: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """`context` (uint8 [h, w, 3], RGB) with the `mask` pixels rebuilt from their surroundings (OpenCV Telea)."""
    bgr = cv2.cvtColor(context, cv2.COLOR_RGB2BGR)
    rebuilt = cv2.inpaint(bgr, mask.astype(np.uint8) * 255, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return cv2.cvtColor(rebuilt, cv2.COLOR_BGR2RGB)

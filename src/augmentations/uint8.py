"""uint8 conventions shared by all augmentations.

Every augmentation takes a grayscale uint8 HxW frame and returns one. There is no
float or 3-channel path anywhere in the pipeline.

* Point operations (y = f(x), e.g. histogram shift, brightness) go through a 256-entry
  lookup table (`cv2.LUT`), so the image itself is never converted.
* Filters (blur, sharpen) run in OpenCV directly on uint8 (OpenCV rounds and saturates).
* Operations that need fractions per pixel (noise, blending with a weight map) compute
  once in float and come back through `to_uint8`, which rounds instead of truncating
  (truncation would darken every frame by ~0.5 grey values on average).
"""
from __future__ import annotations

import numpy as np

RAMP = np.arange(256, dtype=np.float32)  # all uint8 grey values: input of a lookup table


def to_uint8(x: np.ndarray) -> np.ndarray:
    """Round to the nearest grey value and saturate to [0, 255]."""
    y = np.rint(x)
    np.clip(y, 0, 255, out=y)
    return y.astype(np.uint8)


def check_gray_uint8(img: np.ndarray) -> None:
    if img.dtype != np.uint8 or img.ndim != 2:
        raise TypeError(f"expected a grayscale uint8 HxW image, got {img.dtype} with shape {img.shape}")

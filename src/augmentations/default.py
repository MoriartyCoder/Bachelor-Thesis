"""Section "default" of the augmentation config: generic augmentations.

`TRANSFORMS` is the registry of the names allowed in the `default:` section of a run config.
For now it holds the generic Albumentations entries the template already had, plus the grey-value
histogram shift / compression (functions in augmentations/ir.py). The final list (with a
literature reference per entry) is still to be decided.

All of them work on grayscale uint8 HxW frames; the spatial ones also move the boxes.
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import albumentations as A
import numpy as np

from augmentations.ir import thermal_histogram_compression, thermal_histogram_shift
from augmentations.uint8 import check_kwargs

# Grey-value point operations (uint8 HxW in, uint8 HxW out). Like the IR effects they draw from the
# global NumPy RNG (seeded per DataLoader worker by Ultralytics).
GREY_VALUE_OPS: dict[str, Callable[..., np.ndarray]] = {
    "histogram_shift": thermal_histogram_shift,              # level: Y + delta, delta ~ U(+-delta_limit)
    "histogram_compression": thermal_histogram_compression,  # gain around the median, gamma ~ U(gamma_min, gamma_max)
}


class GreyValueTransform(A.ImageOnlyTransform):
    """Wraps one grey-value operation as a pixel-only Albumentations transform (boxes are unchanged)."""

    def __init__(self, op: str, p: float = 0.5, **params):
        super().__init__(p=p)
        if op not in GREY_VALUE_OPS:
            raise KeyError(f"Unknown grey-value operation '{op}'. Available: {sorted(GREY_VALUE_OPS)}")
        self.op = op
        self.op_kwargs = params  # not `self.params`: Albumentations overwrites that with the sampled params on every call
        check_kwargs(GREY_VALUE_OPS[op], params, f"grey-value operation '{op}'")

    def apply(self, img: np.ndarray, **_) -> np.ndarray:
        return GREY_VALUE_OPS[self.op](img, **self.op_kwargs)

    def get_transform_init_args_names(self):
        return ("op", "op_kwargs")


# Name in the `default:` section -> factory(p=..., **params) -> transform
TRANSFORMS: dict[str, Callable[..., A.BasicTransform]] = {
    "HorizontalFlip": A.HorizontalFlip,
    "VerticalFlip": A.VerticalFlip,
    "Affine": A.Affine,
    "RandomBrightnessContrast": A.RandomBrightnessContrast,
    "GaussNoise": A.GaussNoise,
    "GaussianBlur": A.GaussianBlur,
    "CLAHE": A.CLAHE,
    **{name: partial(GreyValueTransform, name) for name in GREY_VALUE_OPS},
}

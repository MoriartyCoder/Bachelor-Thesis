"""Section "default" of the augmentation config: generic augmentations.

`TRANSFORMS` is the registry of the names allowed in the `default:` section of a run config.
For now it holds the generic Albumentations entries the template already had. The final
list (with a literature reference per entry) is still to be decided.

All of them work on grayscale uint8 HxW frames; the spatial ones also move the boxes.
"""
from __future__ import annotations

from typing import Callable

import albumentations as A

# Name in the `default:` section -> factory(p=..., **params) -> transform
TRANSFORMS: dict[str, Callable[..., A.BasicTransform]] = {
    "HorizontalFlip": A.HorizontalFlip,
    "VerticalFlip": A.VerticalFlip,
    "Affine": A.Affine,
    "RandomBrightnessContrast": A.RandomBrightnessContrast,
    "GaussNoise": A.GaussNoise,
    "GaussianBlur": A.GaussianBlur,
    "CLAHE": A.CLAHE,
}

"""Section "edge_cases" of the augmentation config: augmentations for difficult detection cases
(tiny targets, occlusion, low contrast, ...).

`TRANSFORMS` is the registry of the names allowed in the `edge_cases:` section of a run config.
It is empty until the list (with a literature reference per entry) is decided.
"""
from __future__ import annotations

from typing import Callable

import albumentations as A

# Name in the `edge_cases:` section -> factory(p=..., **params) -> transform
TRANSFORMS: dict[str, Callable[..., A.BasicTransform]] = {}

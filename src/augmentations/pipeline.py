"""Build the Albumentations pipeline from the `augmentation:` config section.

Entry names:
  * ``ir.<name>``  -> IR sensor effect from ir_camara_effects.py (see IR_EFFECTS)
  * ``ACB``        -> copy-paste of training targets (acb.py, Tong et al. 2024); needs the train records
  * anything else  -> ``albumentations.<name>`` (e.g. HorizontalFlip, GaussNoise, ...)

Each entry: ``{name, enabled, p, params}``. Disabled entries are skipped.
The pipeline works on grayscale uint8 HxW images (see augmentations/uint8.py) and on
YOLO-normalized boxes.
"""
from __future__ import annotations

import warnings
from typing import Callable

import albumentations as A
import numpy as np

from augmentations import ir_camara_effects as ir
from augmentations.acb import ACB
from log_setup import get_logger

logger = get_logger(__name__)


# Name used in the config (after "ir.") -> implementation (uint8 HxW in, uint8 HxW out).
IR_EFFECTS: dict[str, Callable[..., np.ndarray]] = {
    "netd": ir.netd_noise,
    "histogram_shift": ir.thermal_histogram_shift,
    "histogram_compression": ir.thermal_histogram_compression,
    "fpn": ir.fpn,
    "salt_and_pepper": ir.salt_and_peper,
    "psf_blur": ir.blurring_with_psf,
}


class IRTransform(A.ImageOnlyTransform):
    """Wraps one IR effect as a pixel-only Albumentations transform."""

    def __init__(self, effect: str, fn_kwargs: dict | None = None, p: float = 0.5):
        super().__init__(p=p)
        if effect not in IR_EFFECTS:
            raise KeyError(f"Unknown IR effect 'ir.{effect}'. Available: {sorted('ir.' + k for k in IR_EFFECTS)}")
        self.effect = effect
        self.fn_kwargs = dict(fn_kwargs or {})

    def apply(self, img: np.ndarray, **_) -> np.ndarray:
        return IR_EFFECTS[self.effect](img, **self.fn_kwargs)

    def get_transform_init_args_names(self):
        return ("effect", "fn_kwargs")


def build_transform(entry: dict, train_records: list | None = None) -> A.BasicTransform:
    name, p, params = entry["name"], float(entry["p"]), dict(entry.get("params") or {})
    logger.debug(f"augmentation {name}: p={p}, params={params}")
    if name.startswith("ir."):
        return IRTransform(name[3:], params, p=p)
    if name == "ACB":
        if train_records is None:
            raise ValueError("ACB needs the training records as copy sources: build_albumentations(..., train_records=...)")
        return ACB(train_records, p=p, **params)
    cls = getattr(A, name, None)
    if cls is None or not isinstance(cls, type) or not issubclass(cls, A.BasicTransform):
        raise KeyError(f"Unknown augmentation '{name}'. Use an albumentations transform name or 'ir.<effect>'.")
    return cls(p=p, **params)


def build_albumentations(aug_cfg: dict, train_records: list | None = None) -> A.Compose | None:
    """Compose of all enabled entries, or None if nothing is enabled.

    `train_records` (FrameRecords of the training split) are the copy sources of ACB.
    """
    transforms = [build_transform(e, train_records) for e in aug_cfg.get("pipeline", []) if e.get("enabled", True)]
    if not transforms:
        logger.info("No augmentation enabled -> training on original frames.")
        return None
    bbox_cfg = {"min_visibility": 0.0, "clip": True, **aug_cfg.get("bbox", {})}
    with warnings.catch_warnings():  # pixel-only pipelines: "Got processor for bboxes, but no transform..."
        warnings.filterwarnings("ignore", message="Got processor for bboxes")
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], **bbox_cfg),
            save_applied_params=True,
        )


def describe_pipeline(aug_cfg: dict) -> str:
    lines = ["Albumentations pipeline (Ultralytics augmentations are always off):"]
    for e in aug_cfg.get("pipeline", []):
        state = "ON " if e.get("enabled", True) else "off"
        lines.append(f"  [{state}] {e['name']:<28} p={e.get('p', 1.0):<5} params={e.get('params') or {}}")
    if len(lines) == 1:
        lines.append("  (empty - no augmentation)")
    lines.append(f"  bbox params: {aug_cfg.get('bbox', {})}")
    return "\n".join(lines)

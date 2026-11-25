"""Build the Albumentations pipeline from the `augmentation:` config section.

The config has four sections, each with its own registry of allowed names:

  * ``sota``       -> augmentations/sota.py       (ACB; needs the training records)
  * ``default``    -> augmentations/default.py
  * ``edge_cases`` -> augmentations/edge_cases.py
  * ``ir``         -> augmentations/ir.py         (IR camera effects)

They run in the order of ``augmentation.order`` (default: config.AUG_SECTION_ORDER).
A section's ``pipeline`` holds entries ``{name, enabled, p, params}`` and groups
``{one_of: {enabled, p, pipeline: [entries]}}``. A group becomes ``A.OneOf``: with probability
``p`` exactly one enabled member runs, chosen with the members' ``p`` as relative weights.
``enabled: false`` switches off a section, a group or an entry.

The pipeline works on grayscale uint8 HxW images (see augmentations/uint8.py) and on
YOLO-normalized boxes.
"""
from __future__ import annotations

import warnings
from typing import Callable

import albumentations as A
import cv2
import numpy as np

from augmentations import default, edge_cases, ir, sota
from config import AUG_SECTION_ORDER, ConfigError
from log_setup import get_logger

logger = get_logger(__name__)

# Section name in the config -> registry (name -> factory(p=..., **params))
SECTION_REGISTRIES: dict[str, dict[str, Callable[..., A.BasicTransform]]] = {
    "sota": sota.TRANSFORMS,
    "default": default.TRANSFORMS,
    "edge_cases": edge_cases.TRANSFORMS,
    "ir": ir.TRANSFORMS,
}
assert set(SECTION_REGISTRIES) == set(AUG_SECTION_ORDER)


def _construct(section: str, name: str, p: float, params: dict, train_records: list | None) -> A.BasicTransform:
    registry = SECTION_REGISTRIES[section]
    if section == "sota" and name in sota.NEEDS_TRAIN_RECORDS:
        if train_records is None:
            raise ValueError(f"{name} needs the training records: build_albumentations(..., train_records=...)")
        return registry[name](train_records, p=p, **params)
    if section == "ir":  # effect parameters as one dict: an effect may have its own `p` (salt_and_pepper)
        return registry[name](params, p=p)
    return registry[name](p=p, **params)


def build_transform(section: str, entry: dict, train_records: list | None = None) -> A.BasicTransform:
    """One transform. Every problem with the entry (unknown name, unknown or invalid parameter) is a ConfigError.

    Albumentations itself only WARNS about unknown arguments and then ignores them (the transform would
    silently run with its defaults); that warning is turned into an error here.
    """
    name, p, params = entry["name"], float(entry["p"]), dict(entry.get("params") or {})
    where = f"augmentation.{section}.{name}"
    registry = SECTION_REGISTRIES[section]
    if name not in registry:
        raise ConfigError(f"Unknown augmentation '{name}' in section '{section}'. "
                          f"Available: {sorted(registry) or '(none yet)'}")
    logger.debug(f"augmentation {section}.{name}: p={p}, params={params}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            transform = _construct(section, name, p, params, train_records)
        except (TypeError, ValueError, KeyError) as e:
            raise ConfigError(f"{where}: {e}") from e
    for w in caught:
        msg = str(w.message)
        if "are not valid for transform" in msg:
            raise ConfigError(f"{where}: {msg} (albumentations {A.__version__}). Check `params`.")
        logger.warning(f"{where}: {msg}")
    return transform


def build_section(section: str, sec_cfg: dict, train_records: list | None = None
                  ) -> tuple[list[A.BasicTransform | A.OneOf], list[tuple[str, A.BasicTransform]]]:
    """Transforms of one section in config order (disabled entries and groups are skipped), and the
    single transforms with their config path (for `check_transforms`)."""
    out, leaves = [], []
    for item in sec_cfg.get("pipeline", []):
        if "one_of" not in item:
            if item.get("enabled", True):
                t = build_transform(section, item, train_records)
                out.append(t)
                leaves.append((f"{section}.{item['name']}", t))
            continue
        group = item["one_of"]
        if not group.get("enabled", True):
            continue
        enabled = [m for m in group["pipeline"] if m.get("enabled", True)]
        members = [build_transform(section, m, train_records) for m in enabled]
        if not members:
            logger.debug(f"{section}: one_of group without enabled members -> dropped")
            continue
        if len(members) == 1:
            logger.debug(f"{section}: one_of group with only {enabled[0]['name']} enabled -> acts as a plain entry "
                         f"with the group's p={group['p']}")
        out.append(A.OneOf(members, p=float(group["p"])))
        leaves += [(f"{section}.one_of.{m['name']}", t) for m, t in zip(enabled, members)]
    return out, leaves


def _check_sample(train_records: list | None) -> tuple[np.ndarray, list[list[float]]]:
    """A real training frame with its box (YOLO format), or a synthetic one if no records are given."""
    for r in train_records or []:
        if r.bbox is None:
            continue
        img = cv2.imread(r.img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        x, y, w, h = r.bbox
        H, W = img.shape
        x0, y0, x1, y1 = max(x, 0.0), max(y, 0.0), min(x + w, W), min(y + h, H)
        if x1 > x0 and y1 > y0:
            return img, [[(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]]
    img = np.full((512, 640), 80, np.uint8)
    img[250:260, 315:325] = 200
    return img, [[320 / 640, 255 / 512, 10 / 640, 10 / 512]]


def check_transforms(leaves: list[tuple[str, A.BasicTransform]], bbox_cfg: dict, train_records: list | None) -> None:
    """Run every enabled transform once (forced, p = 1) on a real frame, so that runtime errors (bad parameter
    values, ...) show up before the run and not inside a DataLoader worker. Raises ConfigError."""
    img, bboxes = _check_sample(train_records)
    for label, t in leaves:
        p_orig = t.p
        t.p = 1.0
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Got processor for bboxes")
                comp = A.Compose([t], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], **bbox_cfg))
                out = comp(image=img, bboxes=bboxes, class_labels=[0] * len(bboxes))
            res = out["image"]
            if res.dtype != np.uint8 or res.shape != img.shape:
                raise TypeError(f"returned {res.dtype} {res.shape}, expected uint8 {img.shape}")
        except ConfigError:
            raise
        except Exception as e:
            raise ConfigError(f"augmentation.{label} fails on a test frame: {type(e).__name__}: {e}") from e
        finally:
            t.p = p_orig
    logger.debug(f"Checked {len(leaves)} transforms on a test frame")


def build_albumentations(aug_cfg: dict, train_records: list | None = None) -> A.Compose | None:
    """Compose of all enabled sections/groups/entries in `aug_cfg["order"]`, or None if nothing is enabled.

    `train_records` (FrameRecords of the training split) are the copy sources of ACB/PGDAA and give the
    test frame of `check_transforms`. Every config problem raises ConfigError before training starts.
    """
    transforms, leaves = [], []
    for section in aug_cfg.get("order", AUG_SECTION_ORDER):
        sec_cfg = aug_cfg.get(section) or {}
        if sec_cfg.get("enabled", True):
            items, sec_leaves = build_section(section, sec_cfg, train_records)
            transforms += items
            leaves += sec_leaves
    if not transforms:
        logger.info("No augmentation enabled -> training on original frames.")
        return None
    bbox_cfg = {"min_visibility": 0.0, "clip": True, **aug_cfg.get("bbox", {})}
    # Before the final Compose: the check wraps each transform in its own Compose, and the final
    # Compose attaches its bbox processors to every transform again.
    check_transforms(leaves, bbox_cfg, train_records)
    with warnings.catch_warnings():  # pixel-only pipelines: "Got processor for bboxes, but no transform..."
        warnings.filterwarnings("ignore", message="Got processor for bboxes")
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], **bbox_cfg),
            save_applied_params=True,
        )


def _entry_line(e: dict, indent: str, on: bool) -> str:
    state = "ON " if on and e.get("enabled", True) else "off"
    return f"{indent}[{state}] {e['name']:<26} p={e.get('p', 1.0):<5} params={e.get('params') or {}}"


def describe_pipeline(aug_cfg: dict) -> str:
    order = list(aug_cfg.get("order", AUG_SECTION_ORDER))
    lines = ["Albumentations pipeline (Ultralytics augmentations are always off):",
             f"  order: {' -> '.join(order)}"]
    for section in order:
        sec_cfg = aug_cfg.get(section) or {}
        sec_on = sec_cfg.get("enabled", True)
        lines.append(f"  [{'ON ' if sec_on else 'off'}] section {section}")
        items = sec_cfg.get("pipeline", [])
        if not items:
            lines.append("      (empty)")
        for item in items:
            if "one_of" in item:
                g = item["one_of"]
                g_on = sec_on and g.get("enabled", True)
                lines.append(f"      [{'ON ' if g_on else 'off'}] one_of p={g.get('p', 1.0)} (member p = weight)")
                lines.extend(_entry_line(m, "          ", g_on) for m in g.get("pipeline", []))
            else:
                lines.append(_entry_line(item, "      ", sec_on))
    lines.append(f"  bbox params: {aug_cfg.get('bbox', {})}")
    return "\n".join(lines)

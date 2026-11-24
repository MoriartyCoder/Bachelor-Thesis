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

from augmentations import default, edge_cases, ir, sota
from config import AUG_SECTION_ORDER
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


def build_transform(section: str, entry: dict, train_records: list | None = None) -> A.BasicTransform:
    name, p, params = entry["name"], float(entry["p"]), dict(entry.get("params") or {})
    registry = SECTION_REGISTRIES[section]
    if name not in registry:
        raise KeyError(f"Unknown augmentation '{name}' in section '{section}'. "
                       f"Available: {sorted(registry) or '(none yet)'}")
    logger.debug(f"augmentation {section}.{name}: p={p}, params={params}")
    if section == "sota" and name in sota.NEEDS_TRAIN_RECORDS:
        if train_records is None:
            raise ValueError(f"{name} needs the training records as copy sources: "
                             "build_albumentations(..., train_records=...)")
        return registry[name](train_records, p=p, **params)
    if section == "ir":  # effect parameters as one dict: an effect may have its own `p` (salt_and_pepper)
        return registry[name](params, p=p)
    return registry[name](p=p, **params)


def build_section(section: str, sec_cfg: dict, train_records: list | None = None) -> list[A.BasicTransform | A.OneOf]:
    """Transforms of one section, in config order (disabled entries and groups are skipped)."""
    out = []
    for item in sec_cfg.get("pipeline", []):
        if "one_of" not in item:
            if item.get("enabled", True):
                out.append(build_transform(section, item, train_records))
            continue
        group = item["one_of"]
        if not group.get("enabled", True):
            continue
        members = [build_transform(section, m, train_records) for m in group["pipeline"] if m.get("enabled", True)]
        names = [m["name"] for m in group["pipeline"] if m.get("enabled", True)]
        if not members:
            logger.debug(f"{section}: one_of group without enabled members -> dropped")
            continue
        if len(members) == 1:
            logger.debug(f"{section}: one_of group with only {names[0]} enabled -> acts as a plain entry "
                         f"with the group's p={group['p']}")
        out.append(A.OneOf(members, p=float(group["p"])))
    return out


def build_albumentations(aug_cfg: dict, train_records: list | None = None) -> A.Compose | None:
    """Compose of all enabled sections/groups/entries in `aug_cfg["order"]`, or None if nothing is enabled.

    `train_records` (FrameRecords of the training split) are the copy sources of ACB.
    """
    transforms = []
    for section in aug_cfg.get("order", AUG_SECTION_ORDER):
        sec_cfg = aug_cfg.get(section) or {}
        if sec_cfg.get("enabled", True):
            transforms.extend(build_section(section, sec_cfg, train_records))
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

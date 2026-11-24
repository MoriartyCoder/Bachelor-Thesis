"""Dataset format registry.

Every on-disk dataset layout gets one loader. A loader reads ONE split folder
(e.g. ``data/CST-AntiUAV/train``) and returns a flat list of ``FrameRecord``.
Everything downstream (sampling, YOLO dataset, manifests) only sees FrameRecords,
so a new layout only needs a new loader registered here.

Select a loader per dataset in the run config:  ``format: default``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from config import DatasetName
from log_setup import get_logger

logger = get_logger(__name__)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# One record per frame, i.e. hundreds of thousands of them. To keep that cheap:
#   slots=True      -> no per-instance __dict__
#   img_path: str   -> a Path object per frame costs several times more than its string
#   bbox: tuple     -> smaller than a list
#   dataset / seq / img_w / img_h are shared objects (one per dataset/sequence), not copies.
@dataclass(slots=True)
class FrameRecord:
    img_path: str
    bbox: tuple[float, float, float, float] | None  # [x, y, w, h] in pixels, top-left origin; None = no target
    img_w: int
    img_h: int
    dataset: DatasetName
    seq: str
    frame_idx: int  # 0-based position inside its sequence (used for stride sampling)


FormatLoader = Callable[[Path, DatasetName], list[FrameRecord]]
FORMAT_LOADERS: dict[str, FormatLoader] = {}


def register_format(name: str):
    """Decorator: ``@register_format("company")`` makes the loader selectable via ``format: company``."""
    def deco(fn: FormatLoader) -> FormatLoader:
        if name in FORMAT_LOADERS:
            raise KeyError(f"Format '{name}' is already registered.")
        FORMAT_LOADERS[name] = fn
        return fn
    return deco


def _valid_box(box) -> bool:
    return box is not None and len(box) == 4 and box[2] > 0 and box[3] > 0


@register_format("default")
def load_default(split_dir: Path, dataset: DatasetName) -> list[FrameRecord]:
    """Anti-UAV style: ``<split>/<seq>/000001.jpg ...`` + ``<seq>/IR_label.json``.

    JSON: ``{"exist": [0/1, ...], "gt_rect" | "gt": [[x, y, w, h] | [], ...]}``.
    Anti-UAV410 uses ``gt_rect``, CST-AntiUAV uses ``gt``. Annotation i belongs to
    the i-th image in sorted filename order.
    """
    tag = f"[{dataset}/{split_dir.name}]"
    records: list[FrameRecord] = []
    for seq_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        seq = seq_dir.name  # one string shared by all records of the sequence
        label_file = seq_dir / "IR_label.json"
        if not label_file.exists():
            logger.warning(f"{tag} {seq}: no IR_label.json, sequence skipped.")
            continue
        with open(label_file, encoding="utf-8") as f:
            ann = json.load(f)

        boxes = ann["gt_rect"] if "gt_rect" in ann else ann.get("gt")
        exist = ann.get("exist")
        if boxes is None:
            logger.warning(f"{tag} {seq}: no 'gt_rect'/'gt' key, sequence skipped.")
            continue
        if exist is None:
            logger.debug(f"{tag} {seq}: no 'exist' key, derived from the boxes.")
            exist = [1 if _valid_box(b) else 0 for b in boxes]

        img_paths = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
        n = min(len(img_paths), len(boxes), len(exist))
        if not (len(img_paths) == len(boxes) == len(exist)):
            logger.warning(
                f"{tag} {seq}: {len(img_paths)} images, {len(boxes)} boxes, "
                f"{len(exist)} exist flags -> using first {n}."
            )
        if n == 0:
            logger.debug(f"{tag} {seq}: empty, skipped.")
            continue

        first = cv2.imread(str(img_paths[0]), cv2.IMREAD_UNCHANGED)
        if first is None:
            logger.warning(f"{tag} {seq}: cannot read {img_paths[0]}, sequence skipped.")
            continue
        h, w = first.shape[:2]

        n_box = 0
        for i in range(n):
            box = boxes[i]
            has_target = bool(exist[i]) and _valid_box(box)
            n_box += has_target
            records.append(FrameRecord(
                img_path=str(img_paths[i]),
                bbox=tuple(float(v) for v in box) if has_target else None,
                img_w=w,
                img_h=h,
                dataset=dataset,
                seq=seq,
                frame_idx=i,
            ))
        logger.debug(f"{tag} {seq}: {n} frames ({n_box} with target), {w}x{h}, label key "
                     f"'{'gt_rect' if 'gt_rect' in ann else 'gt'}'")
    return records


# ---------------------------------------------------------------------------
# Template for the upcoming second layout. Uncomment, implement and select it
# in the run config with ``format: company``. It only has to return FrameRecords.
# ---------------------------------------------------------------------------
# @register_format("company")
# def load_company(split_dir: Path, dataset: DatasetName) -> list[FrameRecord]:
#     records = []
#     ...
#     return records

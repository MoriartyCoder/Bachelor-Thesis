"""Bridge between FrameRecords and Ultralytics.

`RecordYOLODataset` builds Ultralytics labels directly from FrameRecords (no label
.txt files, no .cache files) and replaces the Ultralytics augmentation stack with
our Albumentations pipeline. Ultralytics' mosaic/affine/HSV/flip and its built-in
Albumentations step are never constructed.
"""
from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np
import torch
from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.utils import LOGGER, colorstr

from data_handler.formats import FrameRecord


class AlbuStep:
    """Ultralytics transform step that runs an Albumentations Compose on image + boxes."""

    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if getattr(self.transform, "seed", None) != torch.initial_seed():
            # own RNG per DataLoader worker (same approach as ultralytics' Albumentations wrapper)
            self.transform.set_random_seed(torch.initial_seed())

        img = labels["img"][..., 0]  # Ultralytics loads grayscale as HxWx1; the pipeline works on HxW (view)
        h, w = img.shape
        instances = labels["instances"]
        instances.convert_bbox("xywh")
        instances.normalize(w, h)
        bboxes = np.clip(instances.bboxes, 0.0, 1.0)
        cls = labels["cls"].reshape(-1)

        new = self.transform(image=img, bboxes=bboxes, class_labels=cls)

        labels["img"] = np.ascontiguousarray(new["image"])[..., None]
        labels["cls"] = np.asarray(new["class_labels"], dtype=np.float32).reshape(-1, 1)
        instances.update(np.asarray(new["bboxes"], dtype=np.float32).reshape(-1, 4))
        labels["instances"] = instances
        return labels


class RecordYOLODataset(YOLODataset):
    def __init__(self, *args, records: list[FrameRecord], albu: A.Compose | None = None, **kwargs):
        # must exist before super().__init__, which calls get_img_files/get_labels/build_transforms
        self.records = records
        self.albu = albu
        super().__init__(*args, **kwargs)

    def get_img_files(self, img_path) -> list[str]:
        return [r.img_path for r in self.records]

    def get_labels(self) -> list[dict]:
        labels = []
        n_boxes = 0
        for r in self.records:
            if r.bbox is not None:
                x, y, bw, bh = r.bbox
                x0, y0 = max(x, 0.0), max(y, 0.0)
                x1, y1 = min(x + bw, r.img_w), min(y + bh, r.img_h)
                if x1 > x0 and y1 > y0:
                    box = np.array([[(x0 + x1) / 2 / r.img_w, (y0 + y1) / 2 / r.img_h,
                                     (x1 - x0) / r.img_w, (y1 - y0) / r.img_h]], dtype=np.float32)
                    cls = np.zeros((1, 1), dtype=np.float32)
                    n_boxes += 1
                else:
                    box, cls = np.zeros((0, 4), np.float32), np.zeros((0, 1), np.float32)
            else:
                box, cls = np.zeros((0, 4), np.float32), np.zeros((0, 1), np.float32)
            labels.append({
                "im_file": r.img_path,
                "shape": (r.img_h, r.img_w),
                "cls": cls,
                "bboxes": box,
                "segments": [],
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",
            })
        LOGGER.info(f"{self.prefix}{len(labels)} images, {n_boxes} boxes, {len(labels) - n_boxes} backgrounds")
        if not labels:
            raise RuntimeError(f"{self.prefix}no records")
        return labels

    def build_transforms(self, hyp) -> Compose:
        if self.augment:
            steps = [AlbuStep(self.albu)] if self.albu is not None else []
            steps.append(LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False))
            transforms = Compose(steps)
        else:
            transforms = Compose([LetterBox(new_shape=self.imgsz, scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                return_obb=False,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=0.0,
            )
        )
        return transforms


def build_record_dataset(cfg, records, albu, batch, data, mode="train", stride=32) -> RecordYOLODataset:
    """Same arguments as ultralytics.data.build_yolo_dataset, but backed by FrameRecords."""
    train = mode == "train"
    return RecordYOLODataset(
        records=records,
        albu=albu if train else None,
        img_path=f"<records:{mode}>",
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=train,
        hyp=cfg,
        rect=cfg.rect or not train,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if train else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=1.0,
        tile=cfg.tile,
    )

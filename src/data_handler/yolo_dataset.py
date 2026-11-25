"""Bridge between FrameRecords and Ultralytics.

`RecordYOLODataset` builds Ultralytics labels directly from FrameRecords (no label
.txt files, no .cache files) and replaces the Ultralytics augmentation stack with
our Albumentations pipeline. Ultralytics' mosaic/affine/HSV/flip and its built-in
Albumentations step are never constructed.

`EvalAlbuStep` applies a fixed corruption to the evaluation frames (robustness tests, training/evaluate.py),
seeded per frame so every model sees exactly the same corrupted images.
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.utils import LOGGER, colorstr

from data_handler.formats import FrameRecord


def frame_id(im_file: str) -> str:
    """<dataset>/<split>/<seq>/<frame>: names a frame independently of the machine."""
    return "/".join(Path(im_file).parts[-4:])


class AlbuStep:
    """Ultralytics transform step that runs an Albumentations Compose on image + boxes.

    Every call also passes `frame_key` = "<run seed>|<dataset>/<split>/<seq>/<frame>" to the Compose.
    Transforms ignore it, except ACB with `fixed: true`, which seeds its own RNG from it so that a frame
    gets the same pastes in every epoch (the paper's "initially enriched" dataset).
    """

    def __init__(self, transform: A.Compose, seed: int = 0):
        self.transform = transform
        self.seed = seed  # run seed (Ultralytics `seed`), part of the frame key

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if getattr(self.transform, "seed", None) != torch.initial_seed():
            # own RNG per DataLoader worker (same approach as ultralytics' Albumentations wrapper)
            self.transform.set_random_seed(torch.initial_seed())
        return self.apply(labels)

    def apply(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Run the Compose on image + boxes of one Ultralytics label dict."""
        img = labels["img"][..., 0]  # Ultralytics loads grayscale as HxWx1; the pipeline works on HxW (view)
        h, w = img.shape
        instances = labels["instances"]
        instances.convert_bbox("xywh")
        instances.normalize(w, h)
        bboxes = np.clip(instances.bboxes, 0.0, 1.0)
        cls = labels["cls"].reshape(-1)

        new = self.transform(image=img, bboxes=bboxes, class_labels=cls,
                             frame_key=f"{self.seed}|{frame_id(labels['im_file'])}")

        labels["img"] = np.ascontiguousarray(new["image"])[..., None]
        labels["cls"] = np.asarray(new["class_labels"], dtype=np.float32).reshape(-1, 1)
        instances.update(np.asarray(new["bboxes"], dtype=np.float32).reshape(-1, 4))
        labels["instances"] = instances
        return labels


class EvalAlbuStep(AlbuStep):
    """AlbuStep for evaluation: the random state depends only on (test name, <dataset>/<split>/<seq>/<frame>).

    Seeds the global NumPy RNG (used by the IR / grey-value functions) and the Compose's own RNG
    before every frame, so the corrupted frame is the same for every model, worker count and batch size.
    """

    def __init__(self, transform: A.Compose, name: str):
        super().__init__(transform)
        self.name = name

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        seed = zlib.crc32(f"{self.name}|{frame_id(labels['im_file'])}".encode())
        np.random.seed(seed)
        self.transform.set_random_seed(seed)
        return self.apply(labels)


class RecordYOLODataset(YOLODataset):
    def __init__(self, *args, records: list[FrameRecord], albu: A.Compose | None = None,
                 eval_albu: EvalAlbuStep | None = None, **kwargs):
        # must exist before super().__init__, which calls get_img_files/get_labels/build_transforms
        self.records = records
        self.albu = albu
        self.eval_albu = eval_albu  # evaluation only (augment=False): fixed per-frame corruption
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
            steps = [AlbuStep(self.albu, seed=int(hyp.seed))] if self.albu is not None else []
            steps.append(LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False))
            transforms = Compose(steps)
        else:
            steps = [self.eval_albu] if self.eval_albu is not None else []
            steps.append(LetterBox(new_shape=self.imgsz, scaleup=False))
            transforms = Compose(steps)
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


def build_record_dataset(cfg, records, albu, batch, data, mode="train", stride=32,
                         eval_albu: EvalAlbuStep | None = None) -> RecordYOLODataset:
    """Same arguments as ultralytics.data.build_yolo_dataset, but backed by FrameRecords.

    `albu` is used only in train mode, `eval_albu` (robustness tests) only in val mode.
    """
    train = mode == "train"
    return RecordYOLODataset(
        records=records,
        albu=albu if train else None,
        eval_albu=None if train else eval_albu,
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

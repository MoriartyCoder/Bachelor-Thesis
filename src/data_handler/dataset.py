from collections.abc import Iterable
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from augmentations.uint8 import RAMP, to_uint8
from config import DatasetName
from data_handler.sources import load_dataset_split

class AugmentationError(Exception):
    """Something went wrong while augmentation."""


class AntiUAVDataset(Dataset):
    """Frames of one split of one or several datasets (for the notebooks, not for training).

    Uses the same loader as training (FrameRecords), so labels are read identically.
        AntiUAVDataset(DatasetName.ANTI_UAV_410)                  # one dataset
        AntiUAVDataset([DatasetName.ANTI_UAV_410,
                        DatasetName.CST_ANTI_UAV], split="val")   # both, concatenated in this order
    Returns (image tensor [1,H,W] in [0,1], target dict, original grayscale image).
    """

    def __init__(self, datasets: DatasetName | Iterable[DatasetName], split: str = "train",
                 transform=None, debug=False):
        self.datasets = [datasets] if isinstance(datasets, DatasetName) else list(datasets)
        self.split = split
        self.debug = debug
        self.samples = [r for ds in self.datasets for r in load_dataset_split(ds, split)]
        self.transform = transform
        if debug:
            self.reverse_path = {r.img_path: i for i, r in enumerate(self.samples)}

    def __len__(self) -> int:
        return len(self.samples)

    def get_index_by_path(self, target_path: Path | str) -> int | None:
        """Index of an image by its absolute path (only with debug=True)."""
        return self.reverse_path.get(str(target_path), None)

    def __getitem__(self, idx):
        sample       = self.samples[idx]
        boxes        = [list(sample.bbox)] if sample.bbox is not None else []
        class_labels = [0] if boxes else []

        # Bild laden (Infrarot als Grayscale)
        img = cv2.imread(sample.img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Konnte Bild nicht laden: {sample.img_path}")

        augmented = {}
        img_augmented = img
        if self.transform:
            try:
                augmented = self.transform(image=img, bboxes=boxes, class_labels=class_labels)
            except Exception as e:
                # Falls z.B. eine Box durch Augmentierung aus dem Bild fliegt
                raise AugmentationError(f"{sample.img_path} couldn't be augmentated!") from e
            img_augmented = augmented["image"]
            boxes = augmented["bboxes"]

        img_tensor = torch.tensor(img_augmented, dtype=torch.float32).unsqueeze(0) / 255.0

        target = {
            "boxes": torch.tensor(np.asarray(boxes, dtype=np.float32).reshape(-1, 4)),
            "labels": torch.tensor(class_labels, dtype=torch.int64),
            "img_path": Path(sample.img_path),
            "dataset": sample.dataset,
            "seq": sample.seq,
            "frame_idx": sample.frame_idx,
        }

        if "applied_transforms" in augmented:
            target["applied_transforms"] = augmented["applied_transforms"]

        if self.debug:
            debug_img = cv2.cvtColor(img_augmented, cv2.COLOR_GRAY2RGB)
            for bbox in boxes:
                x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                debug_img = cv2.rectangle(debug_img, (x0, y0), (x1, y1), color=(255, 0, 0))
            target["debug"] = debug_img

        return img_tensor, target, img

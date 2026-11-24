from collections.abc import Iterable
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import DatasetName
from data_handler.sources import load_dataset_split

def sinusoidal_intensity_mapping(image):
    """
    Wendet eine sinusförmige Kurve auf die Pixelwerte an,
    um periodische Kontrast-Kompressionen zu erzeugen.
    """
    # Auf [0, 1] skalieren
    img_norm = image.astype(np.float32) / 255.0
    
    # Sinusförmiges Mapping (z.B. Verstärkung mittlerer Kontraste)
    # sin(x * pi) mappt 0->0, 0.5->1.0, 1.0->0 (je nach Formel anpassbar)
    mapped = np.sin(img_norm * np.pi) 
    
    # Zurück auf [0, 255] uint8 skalieren
    return (np.clip(mapped, 0, 1) * 255).astype(np.uint8)

basic_augmentation = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(-30, 30), p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        
    ]
)

# Definiere die IR-Augmentierungs-Pipeline
ir_augmentation_pipeline = A.Compose([
    #*basic_augmentation.transforms,
    # 1. Simuliert thermisches Rauschen des Sensors
    #A.GaussNoise(var_limit=(5.0, 25.0), p=0.5),
    
    # 2. Simuliert Thermal Blooming (leichtes Verschwimmen der Hitze)
    #A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    
    # 3. Simuliert Thermal Crossover / wechselnde Temperaturkontraste
    #A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
    
    # 4. Verbessert/verschlechtert lokale Kontraste (wie echte Wärmebilder)
    #A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    
    # 5. Simuliert Hitzeflimmern in Bodennähe
    #A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.2),

    #A.Lambda(image=sinusoidal_intensity_mapping, p=0.3),
    A.Normalize()
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']),
   save_applied_params=True)


ir_augmentation_pipeline_bkg = A.Compose([
    # 1. Simuliert thermisches Rauschen des Sensors
    # The Power of Augmentations in IR Object Detection
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    
    # 2. Simuliert Thermal Blooming (leichtes Verschwimmen der Hitze)
    # Augmentation of Atmospheric Turbulence Effects on Thermal-Adapted Object Detection Models
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    
    # 3. Simuliert Thermal Crossover / wechselnde Temperaturkontraste
    # Open Dataset and Enhancement Method for Long-wave Thermal Diurnal Material
    # LSOTB-TIR: A Large-Scale High-Diversity Thermal Infrared Object Tracking Benchmark
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
    
    # 4. Verbessert/verschlechtert lokale Kontraste (wie echte Wärmebilder)
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    
    # 5. Simuliert Hitzeflimmern in Bodennähe
    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.2),
    A.GridDistortion(num_steps=5, distort_range=(-0.1, 0.1), p=0.2),
    
    # 6. Geometrische Standard-Augmentierungen (für Robustheit)
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.3),
    A.Normalize()
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']),
   save_applied_params=True)



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
        self.transform = transform if transform is not None else ir_augmentation_pipeline
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

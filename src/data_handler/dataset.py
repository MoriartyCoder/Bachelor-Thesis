import json

import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
import numpy as np

def sinusoidal_intensity_mapping(image, **kwargs):
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
    def __init__(self, dataset_root_dir, transform=None):
        self.dataset_root_dir = Path(dataset_root_dir)
        self.samples = []  # Flache Liste für alle einzelnen Frames

        # 1. Ordnerstruktur durchlaufen und alle Bilder + Labels einsammeln
        # Wir suchen nach allen Sequenz-Ordnern im Split (z.B. train/val/test)
        for seq_dir in self.dataset_root_dir.iterdir():
            if not seq_dir.is_dir():
                continue
            
            with open(seq_dir / "IR_label.json", 'r') as f:
                annotations_data = json.load(f)

            # Alle Bilder in dieser Sequenz finden
            img_paths = sorted(list(seq_dir.glob("*.jpg")))
            
            for i, img_path in enumerate(img_paths):
                
                self.samples.append({
                    'img_path': img_path,
                    'bbox': annotations_data["gt_rect"][i],
                    'exist': annotations_data["exist"][i]
                })

        # 2. Transform zuweisen
        self.transform = transform if transform is not None else ir_augmentation_pipeline


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        sample       = self.samples[idx]
        boxes        = [sample['bbox']] if sample['exist'] == 1 else []
        class_labels = [0] if boxes else []
        
        # 3. Bild laden (Infrarot als Grayscale)
        img = cv2.imread(str(sample['img_path']), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Konnte Bild nicht laden: {sample['img_path']}")

        if self.transform:
            try:
                augmented = self.transform(
                    image=img,
                    bboxes=boxes,
                    class_labels=class_labels
                )
                img_augmented = augmented['image']
                boxes = augmented['bboxes']
                #class_labels = augmented['class_labels']
            except Exception as e:
                # Falls z.B. eine Box durch Augmentierung aus dem Bild fliegt
                raise AugmentationError(f"{str(sample['img_path'])} couldn't be augmentated!")

        #img_tensor = torch.tensor(img_augmented, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_tensor = torch.tensor(img_augmented, dtype=torch.float32).unsqueeze(0) / 255.0

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(class_labels, dtype=torch.int64),
            "img_path": sample['img_path']
        }

        if "applied_transforms" in augmented:
            target["applied_transforms"] = augmented["applied_transforms"]

        return img_tensor, target, img

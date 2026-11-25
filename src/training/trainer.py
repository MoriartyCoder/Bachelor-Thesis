"""Ultralytics trainer/validator that read FrameRecords instead of YOLO label folders."""
from __future__ import annotations

import shutil
from pathlib import Path

import albumentations as A
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils.torch_utils import unwrap_model

from data_handler.formats import FrameRecord
from log_setup import get_logger
from data_handler.yolo_dataset import EvalAlbuStep, build_record_dataset

logger = get_logger(__name__)


def _split_of(img_path) -> str:
    """data.yaml points train/val/test to manifests/<split>.txt -> split name = file stem."""
    return Path(str(img_path)).stem


class RecordDetectionTrainer(DetectionTrainer):
    def __init__(self, split_records: dict[str, list[FrameRecord]], albu: A.Compose | None,
                 snapshot_every: int = -1, **kwargs):
        self.split_records = split_records
        self.albu = albu
        self.snapshot_every = snapshot_every
        super().__init__(**kwargs)

    def save_model(self):
        """last.pt/best.pt as usual, plus weights/epoch<N>.pt after every `snapshot_every` completed epochs.

        Replaces ultralytics' save_period, which counts from 0 (epoch0.pt after the first epoch).
        """
        super().save_model()
        n = self.snapshot_every
        if n > 0 and (self.epoch + 1) % n == 0 and self.last.exists():
            shutil.copy2(self.last, self.wdir / f"epoch{self.epoch + 1}.pt")
            logger.info(f"Checkpoint saved: {self.wdir / f'epoch{self.epoch + 1}.pt'}")

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32) if self.model else 32
        records = self.split_records[_split_of(img_path)]
        logger.debug(f"build_dataset({_split_of(img_path)}, mode={mode}): {len(records)} records, stride {gs}, "
                     f"albumentations {'on' if mode == 'train' and self.albu else 'off'}")
        return build_record_dataset(self.args, records, self.albu, batch, self.data, mode=mode, stride=gs)


class RecordDetectionValidator(DetectionValidator):
    """Standalone evaluation (e.g. test split) on FrameRecords; never augments (`eval_albu`: only the fixed
    corruption of a robustness test, see training/evaluate.py).

    Also keeps the matching statistics of every image (`image_stats`: img_path -> tp, conf, pred_cls,
    target_cls, target_img), so that subsets of the frames can be evaluated afterwards without running
    the model again (training/evaluate.py).
    """

    def __init__(self, split_records: dict[str, list[FrameRecord]], eval_albu: EvalAlbuStep | None = None, **kwargs):
        self.split_records = split_records
        self.eval_albu = eval_albu  # robustness test: fixed per-frame corruption before the letterbox
        self.image_stats: dict[str, dict] = {}
        super().__init__(**kwargs)

    def update_metrics(self, preds, batch) -> None:
        n0 = len(self.metrics.stats["tp"])
        super().update_metrics(preds, batch)
        for si in range(len(preds)):
            self.image_stats[str(batch["im_file"][si])] = {k: v[n0 + si] for k, v in self.metrics.stats.items()}

    def build_dataset(self, img_path, mode="val", batch=None):
        records = self.split_records[_split_of(img_path)]
        stride = self.stride.max() if hasattr(self.stride, "max") else self.stride
        return build_record_dataset(self.args, records, None, batch, self.data, mode="val", stride=max(int(stride), 32),
                                    eval_albu=self.eval_albu)

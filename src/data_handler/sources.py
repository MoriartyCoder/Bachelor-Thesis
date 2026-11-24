"""Collect the frames of one split from all configured datasets.

The same split of every dataset is combined (train = all train/ folders, ...).
Datasets that are disabled or not extracted are skipped with a warning; this
module never unpacks archives or calls the prepare_* functions.
"""
from __future__ import annotations

import csv
import time
from collections import Counter
from pathlib import Path

from config import PREPARE_IDS, SPLITS, DatasetName
from data_handler.formats import FORMAT_LOADERS, FrameRecord
from data_handler.sampling import apply_sampler
from log_setup import get_logger

logger = get_logger(__name__)

_PREPARE_ID_OF = {ds: pid for pid, ds in PREPARE_IDS.items()}


class NoDataError(RuntimeError):
    """No configured dataset provided frames for a split."""


def _missing_hint(dataset: DatasetName) -> str:
    if dataset.zip_path.exists():
        pid = _PREPARE_ID_OF.get(dataset)
        cmd = f"python src/main.py -p {pid}" if pid else "the matching prepare step"
        return f" Archive {dataset.zip_path.name} exists but is not extracted (run `{cmd}` manually)."
    return ""


def load_dataset_split(dataset: DatasetName, split: str, root: Path | None = None,
                       fmt: str = "default") -> list[FrameRecord]:
    """All frames of one split of one dataset (no sampling). Empty list if the split folder is missing."""
    split_dir = (root or dataset.root) / split
    if not split_dir.is_dir():
        logger.warning(f"[{split}] {dataset}: {split_dir} not found, skipped.{_missing_hint(dataset)}")
        return []
    if fmt not in FORMAT_LOADERS:
        raise KeyError(f"Dataset {dataset}: unknown format '{fmt}'. Available: {sorted(FORMAT_LOADERS)}")
    start = time.perf_counter()
    recs = FORMAT_LOADERS[fmt](split_dir, dataset)
    logger.info(f"[{split}] {dataset}: {len(recs)} frames in {len({r.seq for r in recs})} sequences "
                f"({time.perf_counter() - start:.1f} s)")
    return recs


def load_split(cfg: dict, split: str) -> list[FrameRecord]:
    """All frames of `split` from every enabled + available dataset in the run config (no sampling)."""
    records: list[FrameRecord] = []
    for ds in cfg["datasets"]:
        if not ds["enabled"]:
            logger.info(f"[{split}] {ds['name']}: disabled in config, skipped.")
            continue
        records.extend(load_dataset_split(ds["name"], split, root=Path(ds["root"]), fmt=ds["format"]))
    return records


def collect_split(cfg: dict, split: str) -> list[FrameRecord]:
    """Load + sample one split. Raises NoDataError if nothing is left."""
    records = load_split(cfg, split)
    if not records:
        raise NoDataError(f"No frames for split '{split}': all datasets are disabled, missing or empty.")
    seed = cfg["run"]["seed"] + SPLITS.index(split)  # independent but reproducible per split
    sampled = apply_sampler(records, cfg["sampling"][split], seed)
    if not sampled:
        raise NoDataError(f"Sampling left no frames for split '{split}'.")

    per_ds = Counter(str(r.dataset) for r in sampled)
    with_box = sum(r.bbox is not None for r in sampled)
    logger.info(
        f"[{split}] sampled {len(sampled)}/{len(records)} frames ({with_box} with target) - "
        + ", ".join(f"{k}: {v}" for k, v in sorted(per_ds.items()))
    )
    return sampled


def write_manifest(records: list[FrameRecord], csv_path: Path) -> Path:
    """Write the used frames as CSV and as a plain image list (.txt, used as Ultralytics data path)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["img_path", "dataset", "seq", "frame_idx", "img_w", "img_h", "x", "y", "w", "h"])
        for r in records:
            box = r.bbox if r.bbox is not None else ("", "", "", "")
            w.writerow([r.img_path, r.dataset.value, r.seq, r.frame_idx, r.img_w, r.img_h, *box])
    txt_path = csv_path.with_suffix(".txt")
    txt_path.write_text("\n".join(r.img_path for r in records) + "\n", encoding="utf-8")
    logger.debug(f"Manifest written: {csv_path} + {txt_path.name} ({len(records)} frames)")
    return txt_path

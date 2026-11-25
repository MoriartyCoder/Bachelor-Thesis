"""Collect the frames of one split from all configured datasets.

The same split of every dataset is combined (train = all train/ folders, ...).
Datasets that are disabled or not extracted are skipped with a warning; this
module never unpacks archives or calls the prepare_* functions.
"""
from __future__ import annotations

import csv
import hashlib
import time
from collections import Counter
from pathlib import Path

from config import PREPARE_IDS, ROOT_DIR, SPLITS, DatasetName
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


# ---------------------------------------------------------------------------
# Frozen test set: the same test frames for every experiment
# ---------------------------------------------------------------------------
TEST_SET_COLUMNS = ["dataset", "seq", "frame_idx", "img_path"]


def _rel(path: str) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path


def freeze_test_set(cfg: dict, path: Path) -> Path:
    """Sample the test split once (datasets + `sampling.test` of `cfg`) and write it to `path`.

    The file defines the test set of every later run. An existing file is never overwritten: delete it
    by hand only if a new test set is really wanted (old results are then no longer comparable).
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists. The test set must stay fixed; delete the file by hand only if you "
                              "really want a new one (results of earlier runs are then no longer comparable).")
    if "test" not in cfg["sampling"]:
        raise NoDataError("`sampling.test` is missing in the config; it defines the frames of the frozen test set.")
    records = collect_split(cfg, "test")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(TEST_SET_COLUMNS)
        for r in records:
            w.writerow([r.dataset.value, r.seq, r.frame_idx, _rel(r.img_path)])
    logger.info(f"Frozen test set: {len(records)} frames -> {path} (sha256 {file_sha256(path)[:12]})")
    return path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_test_set(cfg: dict, path: Path) -> tuple[list[FrameRecord], str]:
    """The frames of the frozen test set (in file order) and the file's sha256.

    Frames are matched by (dataset, seq, frame_idx), so the file works on any machine. Root and format of
    a dataset come from the run config if it lists the dataset (also when it is disabled for training),
    otherwise the defaults. Raises NoDataError if the file is missing or a frame cannot be found.
    """
    path = Path(path)
    if not path.exists():
        raise NoDataError(f"Frozen test set {path} not found. Create it once with "
                          "`python src/main.py -c configs/template.yaml --freeze-test`.")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    wanted = [(DatasetName(r["dataset"]), r["seq"], int(r["frame_idx"])) for r in rows]
    cfg_ds = {d["name"]: d for d in cfg["datasets"]}
    index: dict[tuple, FrameRecord] = {}
    for ds in sorted({w[0] for w in wanted}, key=lambda d: d.value):
        d = cfg_ds.get(ds, {})
        for r in load_dataset_split(ds, "test", root=Path(d["root"]) if "root" in d else None,
                                    fmt=d.get("format", "default")):
            index[(r.dataset, r.seq, r.frame_idx)] = r
    missing = [w for w in wanted if w not in index]
    if missing:
        raise NoDataError(f"{len(missing)} of {len(wanted)} frames of the frozen test set {path} were not found "
                          f"(first: {missing[0][0].value}/{missing[0][1]}/{missing[0][2]}). Are all test splits extracted?")
    records = [index[w] for w in wanted]
    per_ds = Counter(str(r.dataset) for r in records)
    logger.info(f"[test] frozen test set {path.name}: {len(records)} frames "
                f"({sum(r.bbox is not None for r in records)} with target) - "
                + ", ".join(f"{k}: {v}" for k, v in sorted(per_ds.items())))
    return records, file_sha256(path)

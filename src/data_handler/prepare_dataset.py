"""Extract a downloaded dataset zip from data/ into data/<dataset>/.

Called via `python src/main.py -p <id>`. The target folder is DELETED and recreated.
"""
import json
import shutil
import subprocess
import time
from pathlib import Path

from config import DatasetName
from log_setup import get_logger

logger = get_logger(__name__)


def reformat_json_file(file_path: Path) -> None:
    """Rewrite a JSON file indented (the original label files are one long line)."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def prepare_dataset(dataset: DatasetName) -> Path:
    start = time.perf_counter()
    archive, target = dataset.zip_path, dataset.root
    logger.info(f"[{dataset}] Prepare: {archive} -> {target}")

    if not archive.exists():
        raise FileNotFoundError(
            f"'{archive.name}' not found in {archive.parent}. Download the dataset once manually "
            f"in the browser and save it as {archive}."
        )

    if target.exists():
        logger.info(f"[{dataset}] Deleting existing folder {target}")
        shutil.rmtree(target)

    logger.info(f"[{dataset}] Extracting with 'unzip' ...")
    subprocess.run(["unzip", "-q", str(archive), "-d", str(target)], check=True)

    json_files = sorted(target.rglob("*.json"))
    logger.info(f"[{dataset}] Reformatting {len(json_files)} JSON label files")
    for file_path in json_files:
        logger.debug(f"[{dataset}] reformat {file_path.relative_to(target)}")
        reformat_json_file(file_path)

    for split_dir in sorted(p for p in target.iterdir() if p.is_dir()):
        n_seq = sum(1 for p in split_dir.iterdir() if p.is_dir())
        logger.info(f"[{dataset}]   {split_dir.name}/: {n_seq} sequences")
    logger.info(f"[{dataset}] Done in {time.perf_counter() - start:.1f} s")
    return target

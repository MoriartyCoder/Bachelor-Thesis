"""Run one training experiment from a YAML config.

Run folder: <run.output_root>/<YYYYMMDD-HHMMSS>/ (start time), containing
config snapshots, manifests, git/env info, the full log of the run (run.log),
Ultralytics results + TensorBoard logs, checkpoints in weights/ and optional test metrics.
"""
from __future__ import annotations

import json
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import cv2
import yaml

import config
from data_handler.augmentation import build_albumentations, describe_pipeline
from data_handler.sources import NoDataError, collect_split, write_manifest
from log_setup import add_log_file, get_logger, remove_log_file
from models.yolo import resolve_device, resolve_model_weights
from utils.utils import env_info, git, plain, seed_everything

logger = get_logger(__name__)

CLASS_NAMES = {0: "uav"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@contextmanager
def step(name: str):
    """Log start (DEBUG) and duration (INFO) of one step of the run."""
    logger.debug(f"--> {name}")
    start = time.perf_counter()
    yield
    logger.info(f"{name}: done in {time.perf_counter() - start:.1f} s")


def write_snapshots(run_dir: Path, cfg_path: Path, cfg: dict, overrides: dict, device) -> None:
    shutil.copy(cfg_path, run_dir / "config.yaml")
    resolved = {k: v for k, v in cfg.items()}
    resolved["ultralytics_overrides"] = overrides
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(plain(resolved), sort_keys=False), encoding="utf-8")
    (run_dir / "augmentation.txt").write_text(describe_pipeline(cfg["augmentation"]) + "\n", encoding="utf-8")
    status = git("status", "--porcelain")
    git_txt = f"commit: {git('rev-parse', 'HEAD').strip()}\nbranch: {git('rev-parse', '--abbrev-ref', 'HEAD').strip()}\n"
    git_txt += f"dirty:  {bool(status.strip())}\n\n--- status ---\n{status}\n--- diff ---\n{git('diff')}"
    (run_dir / "git.txt").write_text(git_txt, encoding="utf-8")
    (run_dir / "env.txt").write_text(env_info(device) + "\n", encoding="utf-8")
    logger.debug(f"Snapshots written: config.yaml, config.resolved.yaml, augmentation.txt, git.txt, env.txt "
                 f"(git dirty: {bool(status.strip())})")


def write_data_yaml(run_dir: Path, manifest_txts: dict[str, Path]) -> Path:
    data = {"path": str(run_dir), "names": CLASS_NAMES, "channels": 1}  # grayscale IR: 1-channel model input
    data.update({split: str(p) for split, p in manifest_txts.items()})
    path = run_dir / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def save_previews(records, albu, cfg_overrides: dict, out_dir: Path, n: int = 16) -> None:
    """Run frames through the real training dataset (Albumentations + letterbox) and draw the boxes."""
    from ultralytics.cfg import get_cfg

    from data_handler.yolo_dataset import build_record_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    with_box = [r for r in records if r.bbox is not None][: n // 2]
    without = [r for r in records if r.bbox is None][: n - len(with_box)]
    subset = with_box + without
    ucfg = get_cfg(overrides={k: v for k, v in cfg_overrides.items() if k not in ("model", "data")})
    ds = build_record_dataset(ucfg, subset, albu, batch=len(subset), data={"names": CLASS_NAMES, "nc": 1, "channels": 1},
                              mode="train")
    logger.debug(f"Previews: {len(with_box)} frames with target, {len(without)} without")
    for i in range(len(ds)):
        item = ds[i]
        img = cv2.cvtColor(item["img"].numpy()[0], cv2.COLOR_GRAY2BGR)  # 1xHxW -> colour, only to draw the red box
        h, w = img.shape[:2]
        for cx, cy, bw, bh in item["bboxes"].numpy():
            x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (0, 0, 255), 1)
        r = subset[i]
        cv2.imwrite(str(out_dir / f"{i:02d}_{r.dataset}_{r.seq}_{r.frame_idx:06d}.jpg"), img)
    logger.info(f"Saved {len(ds)} augmented previews to {out_dir}")


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def run(cfg_path: str | Path, dry_run: bool = False) -> Path:
    """Train (and test) one experiment. The complete log of the run is also written to <run_dir>/run.log."""
    cfg_path = Path(cfg_path).resolve()
    logger.info(f"Run config: {cfg_path}" + ("  [DRY RUN]" if dry_run else ""))
    cfg = config.load_run_config(cfg_path)
    logger.debug("Resolved run config:\n" + yaml.safe_dump(plain(cfg), sort_keys=False))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = config.OUTPUT_DIR / "dryrun" if dry_run else Path(cfg["run"]["output_root"])
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    log_handler = add_log_file(run_dir / "run.log")
    try:
        logger.info(f"Run directory: {run_dir}")
        _run(cfg, cfg_path, run_dir, output_root, timestamp, dry_run)
    except Exception as e:
        logger.exception(f"Run failed. Partial results in {run_dir}")  # traceback also lands in run.log
        e.already_logged = True  # main.py does not log it a second time
        raise
    finally:
        remove_log_file(log_handler)
    return run_dir


def _run(cfg: dict, cfg_path: Path, run_dir: Path, output_root: Path, timestamp: str, dry_run: bool) -> None:
    from ultralytics import settings as ul_settings

    from training.trainer import RecordDetectionTrainer, RecordDetectionValidator

    run_cfg = cfg["run"]
    seed_everything(run_cfg["seed"], run_cfg["deterministic"])
    logger.debug(f"Seeded python/numpy/torch with {run_cfg['seed']}, deterministic={run_cfg['deterministic']}")

    device = resolve_device(cfg["train"].get("device"))
    logger.info("Environment:\n" + env_info(device))

    # --- data -------------------------------------------------------------
    splits = ["train", "val"] + (["test"] if run_cfg["test_after_training"] else [])
    split_records = {}
    with step("Loading + sampling data"):
        for split in splits:
            try:
                split_records[split] = collect_split(cfg, split)
            except NoDataError:
                if split == "test":
                    logger.warning("No test data available -> test evaluation disabled.")
                    continue
                raise

    # Copy sources and priors of the augmentation (ACB, PGDAA) always come from the full train split,
    # also in a dry run (a 32-frame subset may have no background frame at all).
    aug_records = split_records["train"]
    if dry_run:  # tiny, fast subsets
        split_records = {s: r[:: max(1, len(r) // 32)][:32] for s, r in split_records.items()}
        logger.info("Dry run: reduced to " + ", ".join(f"{s}={len(r)}" for s, r in split_records.items()) + " frames")

    manifest_txts = {s: write_manifest(r, run_dir / "manifests" / f"{s}.csv") for s, r in split_records.items()}
    data_yaml = write_data_yaml(run_dir, manifest_txts)

    # --- augmentation ------------------------------------------------------
    albu = build_albumentations(cfg["augmentation"], train_records=aug_records)
    logger.info(describe_pipeline(cfg["augmentation"]))
    logger.debug(f"Albumentations Compose:\n{albu}")

    # --- ultralytics overrides ---------------------------------------------
    train_args = dict(cfg["train"])
    train_args["device"] = device

    if dry_run:
        train_args.update(epochs=1, batch=4, workers=0, device="cpu", plots=False)

    overrides = {
        **train_args,
        "model": resolve_model_weights(str(cfg["model"]["weights"])),
        "data": str(data_yaml),
        "imgsz": cfg["model"]["imgsz"],
        "project": str(output_root),
        "name": timestamp,
        "exist_ok": True,  # run_dir was created above with exist_ok=False
        "save_period": -1,  # snapshots are written by RecordDetectionTrainer (1-based epoch numbering)
        "seed": run_cfg["seed"],
        "deterministic": run_cfg["deterministic"],
        **config.ULTRALYTICS_AUG_OFF,  # last: can never be overridden
    }
    logger.debug("Ultralytics overrides:\n" + yaml.safe_dump(plain(overrides), sort_keys=False))
    write_snapshots(run_dir, cfg_path, cfg, overrides, device)

    if dry_run:
        with step("Augmented previews"):
            save_previews(split_records["train"], albu, overrides, run_dir / "previews")

    ul_settings.update({"tensorboard": True})

    # --- train -------------------------------------------------------------
    logger.info(f"Training: {overrides.get('epochs')} epochs, batch {overrides.get('batch')}, device {device}, "
                f"model {overrides['model']}")
    with step("Training"):
        trainer = RecordDetectionTrainer(
            split_records=split_records, albu=albu,
            snapshot_every=1 if dry_run else run_cfg["checkpoint_every_n_epochs"], overrides=overrides,
        )
        trainer.train()

    # --- test --------------------------------------------------------------
    best = run_dir / "weights" / "best.pt"
    if "test" in split_records and best.exists():
        logger.info(f"Evaluating {best.name} on the test split")
        val_args = dict(
            model=str(best), data=str(data_yaml), split="test", imgsz=cfg["model"]["imgsz"],
            batch=train_args.get("batch", 16), device=train_args["device"], workers=train_args.get("workers", 8),
            project=str(run_dir), name="test", exist_ok=True, plots=not dry_run,
        )
        logger.debug(f"Test validator args: {val_args}")
        with step("Test evaluation"):
            validator = RecordDetectionValidator(split_records=split_records, args=val_args)
            stats = validator(model=str(best))
        metrics = {k: float(v) for k, v in stats.items()}
        (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("Test metrics:\n" + "\n".join(f"  {k:<28} {v:.4f}" for k, v in metrics.items()))
    elif "test" in split_records:
        logger.warning(f"{best} not found -> no test evaluation.")

    logger.info(f"Done. Results in {run_dir}")

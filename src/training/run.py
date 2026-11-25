"""Run one training experiment from a YAML config, or resume an interrupted one.

Run folder: <run.output_root>/<YYYYMMDD-HHMMSS>_<config name>/ (start time + file name of the run config
without .yaml, so runs can be told apart in TensorBoard and notebooks/08_compare_runs.ipynb), containing
config snapshots, manifests, git/env info, the full log of the run (run.log),
Ultralytics results + TensorBoard logs, checkpoints in weights/ and the test metrics.

The test set is always the frozen file `run.test_set` (default configs/test_set.csv), so every
experiment is evaluated on exactly the same frames. With `run.robustness_suite` the model is also tested on
corrupted versions of these frames (robustness_metrics.json); `evaluate_run` does this for a finished run.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import cv2
import yaml

import config
from data_handler.augmentation import build_albumentations, describe_pipeline
from data_handler.sources import NoDataError, collect_split, load_test_set, write_manifest
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
    dirty = write_git_snapshot(run_dir / "git.txt")
    (run_dir / "env.txt").write_text(env_info(device) + "\n", encoding="utf-8")
    logger.debug(f"Snapshots written: config.yaml, config.resolved.yaml, augmentation.txt, git.txt, env.txt "
                 f"(git dirty: {dirty})")


def write_git_snapshot(path: Path) -> bool:
    """Commit, branch, status and diff of the working tree; returns whether it is dirty."""
    status = git("status", "--porcelain")
    git_txt = f"commit: {git('rev-parse', 'HEAD').strip()}\nbranch: {git('rev-parse', '--abbrev-ref', 'HEAD').strip()}\n"
    git_txt += f"dirty:  {bool(status.strip())}\n\n--- status ---\n{status}\n--- diff ---\n{git('diff')}"
    path.write_text(git_txt, encoding="utf-8")
    return bool(status.strip())


def write_data_yaml(run_dir: Path, manifest_txts: dict[str, Path]) -> Path:
    data = {"path": str(run_dir), "names": CLASS_NAMES, "channels": 1}  # grayscale IR: 1-channel model input
    data.update({split: str(p) for split, p in manifest_txts.items()})
    path = run_dir / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def run_folder_label(cfg_path: Path) -> str:
    """Config file name without .yaml, reduced to characters that are safe in a folder name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", Path(cfg_path).stem).strip("-") or "run"


def test_val_args(cfg: dict, data_yaml: Path, run_dir: Path, device, dry_run: bool) -> dict:
    """Validator arguments for the test evaluation of a run."""
    train = cfg["train"]
    return dict(
        data=str(data_yaml), split="test", imgsz=cfg["model"]["imgsz"],
        batch=4 if dry_run else train.get("batch", 16), device=device, workers=0 if dry_run else train.get("workers", 8),
        project=str(run_dir), name="test", exist_ok=True, plots=not dry_run,
    )


def dry_run_subset(records: list) -> list:
    """The ~32 frames a dry run uses of a split."""
    return records[:: max(1, len(records) // 32)][:32]


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
def run(cfg_path: str | Path | None, dry_run: bool = False, resume_dir: str | Path | None = None) -> Path:
    """Train (and test) one experiment, or resume the run in `resume_dir` from its weights/last.pt.

    The complete log of the run is also written to <run_dir>/run.log (appended when resuming).
    """
    resume_ckpt = None
    if resume_dir is not None:
        run_dir = Path(resume_dir).resolve()
        resume_ckpt = run_dir / "weights" / "last.pt"
        cfg_path = run_dir / "config.yaml"
        for f in (resume_ckpt, cfg_path, run_dir / "data.yaml"):
            if not f.exists():
                raise FileNotFoundError(f"Cannot resume {run_dir}: {f.name} not found.")
        output_root, timestamp = run_dir.parent, run_dir.name
        cfg = config.load_run_config(cfg_path)
    else:
        cfg_path = Path(cfg_path).resolve()
        cfg = config.load_run_config(cfg_path)
        if cfg["run"]["test_after_training"] and not Path(cfg["run"]["test_set"]).exists():
            raise NoDataError(f"Frozen test set {cfg['run']['test_set']} not found. Create it once with "
                              "`python src/main.py -c configs/template.yaml --freeze-test`.")
        timestamp = f"{datetime.now():%Y%m%d-%H%M%S}_{run_folder_label(cfg_path)}"
        output_root = config.OUTPUT_DIR / "dryrun" if dry_run else Path(cfg["run"]["output_root"])
        run_dir = output_root / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)

    log_handler = add_log_file(run_dir / "run.log")
    try:
        if resume_ckpt is not None:
            logger.info("=" * 15 + f" RESUME {datetime.now():%Y-%m-%d %H:%M:%S} " + "=" * 15)
        logger.info(f"Run config: {cfg_path}" + ("  [DRY RUN]" if dry_run else ""))
        logger.debug("Resolved run config:\n" + yaml.safe_dump(plain(cfg), sort_keys=False))
        logger.info(f"Run directory: {run_dir}")
        _run(cfg, cfg_path, run_dir, output_root, timestamp, dry_run, resume_ckpt)
    except Exception as e:
        logger.exception(f"Run failed. Partial results in {run_dir}")  # traceback also lands in run.log
        e.already_logged = True  # main.py does not log it a second time
        raise
    finally:
        remove_log_file(log_handler)
    return run_dir


def _check_same_frames(split_records: dict, run_dir: Path) -> None:
    """Resume: the rebuilt frame lists must equal the manifests of the original run."""
    for split, records in split_records.items():
        txt = run_dir / "manifests" / f"{split}.txt"
        if not txt.exists():
            raise FileNotFoundError(f"Cannot resume: {txt} not found.")
        saved = txt.read_text(encoding="utf-8").split()
        if saved != [r.img_path for r in records]:
            raise ValueError(f"Cannot resume: the {split} frames differ from the original run ({len(saved)} saved, "
                             f"{len(records)} now). Data, config or test set changed since the run started.")
    logger.info("Resume: frame lists identical to the original run.")


def _run(cfg: dict, cfg_path: Path, run_dir: Path, output_root: Path, timestamp: str, dry_run: bool,
         resume_ckpt: Path | None = None) -> None:
    from ultralytics import settings as ul_settings

    from training.evaluate import evaluate_robustness, evaluate_test
    from training.trainer import RecordDetectionTrainer

    run_cfg = cfg["run"]
    seed_everything(run_cfg["seed"], run_cfg["deterministic"])
    logger.debug(f"Seeded python/numpy/torch with {run_cfg['seed']}, deterministic={run_cfg['deterministic']}")

    device = resolve_device(cfg["train"].get("device"))
    logger.info("Environment:\n" + env_info(device))

    # --- data -------------------------------------------------------------
    split_records, test_set = {}, None
    with step("Loading + sampling data"):
        for split in ("train", "val"):
            split_records[split] = collect_split(cfg, split)
        if run_cfg["test_after_training"]:
            split_records["test"], sha = load_test_set(cfg, run_cfg["test_set"])
            test_set = {"file": str(run_cfg["test_set"]), "sha256": sha, "frames": len(split_records["test"])}

    # Copy sources and priors of the augmentation (ACB, PGDAA) always come from the full train split,
    # also in a dry run (a 32-frame subset may have no background frame at all).
    aug_records = split_records["train"]
    if dry_run:  # tiny, fast subsets
        split_records = {s: dry_run_subset(r) for s, r in split_records.items()}
        logger.info("Dry run: reduced to " + ", ".join(f"{s}={len(r)}" for s, r in split_records.items()) + " frames")
        if test_set is not None:
            test_set["frames"] = len(split_records["test"])
            test_set["dry_run_subset"] = True

    if resume_ckpt is not None:
        _check_same_frames(split_records, run_dir)
        data_yaml = run_dir / "data.yaml"
    else:
        manifest_txts = {s: write_manifest(r, run_dir / "manifests" / f"{s}.csv") for s, r in split_records.items()}
        data_yaml = write_data_yaml(run_dir, manifest_txts)

    # --- augmentation ------------------------------------------------------
    albu = build_albumentations(cfg["augmentation"], train_records=aug_records)
    logger.info(describe_pipeline(cfg["augmentation"]))
    logger.debug(f"Albumentations Compose:\n{albu}")
    if run_cfg["robustness_suite"] is not None and run_cfg["test_after_training"]:
        # fail before training, not after it: every test is loaded and built (and checked on a frame) once now
        tests, sha = config.load_robustness_suite(run_cfg["robustness_suite"])
        for _, test_aug in tests:
            build_albumentations(test_aug)
        logger.info(f"Robustness suite {run_cfg['robustness_suite']} (sha256 {sha[:12]}): {len(tests)} tests, "
                    "evaluated after training")

    # --- ultralytics overrides ---------------------------------------------
    train_args = dict(cfg["train"])
    train_args["device"] = device

    if dry_run:
        train_args.update(epochs=1, batch=4, workers=0, device="cpu", plots=False)

    if resume_ckpt is not None:
        # Ultralytics takes every argument from the checkpoint; only these may change on resume.
        overrides = {"resume": str(resume_ckpt), "data": str(data_yaml), "device": device,
                     "workers": train_args.get("workers", 8)}
        dirty = write_git_snapshot(run_dir / f"git.resume-{datetime.now():%Y%m%d-%H%M%S}.txt")
        logger.info(f"Resuming from {resume_ckpt} (code state at resume: git dirty = {dirty})")
    else:
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
    logger.info(f"Training: {train_args.get('epochs')} epochs, batch {train_args.get('batch')}, device {device}, "
                f"model {overrides.get('model', resume_ckpt)}")
    with step("Training"):
        trainer = RecordDetectionTrainer(
            split_records=split_records, albu=albu,
            snapshot_every=1 if dry_run else run_cfg["checkpoint_every_n_epochs"], overrides=overrides,
        )
        trainer.train()

    # --- test --------------------------------------------------------------
    best = run_dir / "weights" / "best.pt"
    if "test" in split_records and best.exists():
        logger.info(f"Evaluating {best.name} on the frozen test set")
        val_args = {"model": str(best), **test_val_args(cfg, data_yaml, run_dir, train_args["device"], dry_run)}
        logger.debug(f"Test validator args: {val_args}")
        with step("Test evaluation"):
            clean = evaluate_test(best, val_args, split_records["test"], test_set, run_dir / "test_metrics.json")
        if run_cfg["robustness_suite"] is not None:
            with step("Robustness evaluation"):
                evaluate_robustness(best, val_args, split_records["test"], test_set, run_cfg["robustness_suite"],
                                    run_dir / "robustness_metrics.json", clean_groups=clean["groups"])
    elif "test" in split_records:
        logger.warning(f"{best} not found -> no test evaluation.")

    logger.info(f"Done. Results in {run_dir}")


def evaluate_run(run_dir: str | Path, suite: str | Path | None = None, dry_run: bool = False) -> Path:
    """Robustness evaluation of a finished run: its weights/best.pt on the frozen test set, clean and per test of
    the suite (`suite`, else the run's `run.robustness_suite`, else config.ROBUSTNESS_SUITE_FILE).

    Uses the run's own config.yaml and data.yaml, appends to its run.log and writes robustness_metrics.json
    (an existing one is replaced) and git.evaluate-<ts>.txt. `dry_run`: the 32 test frames and CPU settings of a
    dry run. Returns the path of robustness_metrics.json.
    """
    from training.evaluate import evaluate_robustness

    run_dir = Path(run_dir).resolve()
    best, cfg_path, data_yaml = run_dir / "weights" / "best.pt", run_dir / "config.yaml", run_dir / "data.yaml"
    for f in (best, cfg_path, data_yaml):
        if not f.exists():
            raise FileNotFoundError(f"Cannot evaluate {run_dir}: {f.relative_to(run_dir)} not found.")
    if "test" not in (yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}):
        raise ValueError(f"Cannot evaluate {run_dir}: data.yaml has no test split (run.test_after_training was off).")
    cfg = config.load_run_config(cfg_path)
    suite_path = config._resolve(suite) if suite else (cfg["run"]["robustness_suite"] or config.ROBUSTNESS_SUITE_FILE)
    out_path = run_dir / "robustness_metrics.json"

    log_handler = add_log_file(run_dir / "run.log")
    try:
        logger.info("=" * 15 + f" EVALUATE {datetime.now():%Y-%m-%d %H:%M:%S} " + "=" * 15)
        logger.info(f"Robustness evaluation of {best} with suite {suite_path}" + ("  [DRY RUN]" if dry_run else ""))
        seed_everything(cfg["run"]["seed"], cfg["run"]["deterministic"])
        device = "cpu" if dry_run else resolve_device(cfg["train"].get("device"))
        dirty = write_git_snapshot(run_dir / f"git.evaluate-{datetime.now():%Y%m%d-%H%M%S}.txt")
        logger.info(f"Code state at evaluation: git dirty = {dirty}")

        with step("Loading frozen test set"):
            records, sha = load_test_set(cfg, cfg["run"]["test_set"])
            test_set = {"file": str(cfg["run"]["test_set"]), "sha256": sha, "frames": len(records)}
            if dry_run:
                records = dry_run_subset(records)
                test_set.update(frames=len(records), dry_run_subset=True)
        if out_path.exists():
            logger.info(f"{out_path.name} exists and is replaced.")
        val_args = {"model": str(best), **test_val_args(cfg, data_yaml, run_dir, device, dry_run)}
        logger.debug(f"Test validator args: {val_args}")
        with step("Robustness evaluation"):
            evaluate_robustness(best, val_args, records, test_set, suite_path, out_path)
    except Exception as e:
        logger.exception(f"Evaluation of {run_dir} failed.")
        e.already_logged = True
        raise
    finally:
        remove_log_file(log_handler)
    return out_path

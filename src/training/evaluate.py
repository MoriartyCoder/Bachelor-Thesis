"""Test evaluation with breakdowns: all frames, per dataset, per target size, frames without target.

The model runs once over the frozen test set (RecordDetectionValidator). The metrics of every subset
are computed afterwards from the per-image statistics with Ultralytics' own DetMetrics, so they are
defined exactly like the combined result (precision/recall at the best F1, mAP50, mAP50-95).

Size classes: sqrt(w * h) of the ground-truth box in px, with the edges of the Anti-UAV410 size
classes (data_handler/target_library.py: <= 10, 10-30, 30-50, > 50 px). Frames without a target form
their own group; there only false alarms are meaningful.

False alarms: predictions with confidence >= FA_CONF on frames without a target.

Robustness (evaluate_robustness): the same evaluation once per test of a fixed suite (configs/robustness.yaml),
each test a corruption pipeline applied to every test frame before the letterbox. The random state of a
corruption depends only on (test name, frame), so every model is tested on exactly the same corrupted frames.
The result holds the clean groups, the groups per test and the difference test - clean. Diagnosis only:
model selection stays on the clean val set.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics.utils.metrics import DetMetrics

from config import load_robustness_suite
from data_handler.augmentation import build_albumentations, describe_pipeline
from data_handler.formats import FrameRecord
from data_handler.target_library import SIZE_CLASSES, size_class
from log_setup import get_logger
from data_handler.yolo_dataset import EvalAlbuStep
from training.trainer import RecordDetectionValidator

logger = get_logger(__name__)

FA_CONF = 0.25  # Ultralytics' default confidence threshold for predictions
METRIC_KEYS = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
}


def frame_groups(records: list[FrameRecord]) -> dict[str, list[FrameRecord]]:
    """all, one group per dataset, one per size class (frames with target), frames without target."""
    groups: dict[str, list[FrameRecord]] = {"all": list(records)}
    per_ds, per_size = defaultdict(list), defaultdict(list)
    for r in records:
        per_ds[f"dataset {r.dataset.value}"].append(r)
        if r.bbox is not None:
            per_size[int(size_class(r.bbox[2], r.bbox[3]))].append(r)
    groups.update(sorted(per_ds.items()))
    groups.update((f"size {SIZE_CLASSES[k].replace(' ', '')}", v) for k, v in sorted(per_size.items()))
    groups["no target"] = [r for r in records if r.bbox is None]
    return {k: v for k, v in groups.items() if v}


def group_metrics(stats: list[dict], names: dict[int, str]) -> dict:
    """Metrics of one group from the per-image statistics."""
    n_targets = int(sum(len(st["target_cls"]) for st in stats))
    background = [st for st in stats if len(st["target_cls"]) == 0]
    false_alarms = int(sum((np.asarray(st["conf"]) >= FA_CONF).sum() for st in background))
    out = {
        "frames": len(stats),
        "targets": n_targets,
        "frames_without_target": len(background),
        "false_alarms": false_alarms,
        "false_alarms_per_frame": false_alarms / len(background) if background else None,
    }
    if n_targets:
        m = DetMetrics(names=names)
        for k in m.stats:
            m.stats[k] = [st[k] for st in stats]
        m.process()
        out.update({new: float(m.results_dict[old]) for old, new in METRIC_KEYS.items()})
    else:
        out.update({new: None for new in METRIC_KEYS.values()})
    return out


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.4f}"


def _evaluate_groups(best: Path, val_args: dict, records: list[FrameRecord],
                     eval_albu: EvalAlbuStep | None = None) -> tuple[dict, dict]:
    """One inference pass of `best` over `records` (optionally corrupted by `eval_albu`).

    Returns the metrics per frame group and Ultralytics' own result on all frames.
    """
    validator = RecordDetectionValidator(split_records={"test": records}, eval_albu=eval_albu, args=val_args)
    ul_stats = validator(model=str(best))
    names = getattr(validator, "names", None) or {0: "uav"}

    missing = [r.img_path for r in records if r.img_path not in validator.image_stats]
    if missing:
        logger.warning(f"{len(missing)} test frames have no statistics (first: {missing[0]}); they are left out.")

    groups = {}
    for name, recs in frame_groups(records).items():
        stats = [validator.image_stats[r.img_path] for r in recs if r.img_path in validator.image_stats]
        if stats:
            groups[name] = group_metrics(stats, names)
    return groups, {k: float(v) for k, v in ul_stats.items()}


def evaluate_test(best: Path, val_args: dict, records: list[FrameRecord], test_set: dict, out_path: Path) -> dict:
    """Evaluate `best` on `records` (the frozen test set), write `out_path` (JSON) and return its content."""
    groups, ul_stats = _evaluate_groups(best, val_args, records)
    result = {
        "test_set": test_set,
        "false_alarm_conf": FA_CONF,
        "groups": groups,
        "ultralytics": ul_stats,  # Ultralytics' own result on all frames
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    header = f"  {'group':<22}{'frames':>8}{'targets':>9}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}{'FA/frame':>10}"
    lines = [header]
    for name, g in groups.items():
        lines.append(f"  {name:<22}{g['frames']:>8}{g['targets']:>9}{_fmt(g['precision']):>8}{_fmt(g['recall']):>8}"
                     f"{_fmt(g['mAP50']):>8}{_fmt(g['mAP50-95']):>10}{_fmt(g['false_alarms_per_frame']):>10}")
    logger.info(f"Test metrics (false alarms: conf >= {FA_CONF} on frames without target) -> {out_path.name}\n"
                + "\n".join(lines))
    return result


DELTA_KEYS = (*METRIC_KEYS.values(), "false_alarms_per_frame")


def _delta(test: dict, clean: dict) -> dict:
    """test - clean per group and metric (None where either side has no value)."""
    out = {}
    for name, g in test.items():
        c = clean.get(name, {})
        out[name] = {k: (g[k] - c[k] if g.get(k) is not None and c.get(k) is not None else None) for k in DELTA_KEYS}
    return out


def evaluate_robustness(best: Path, val_args: dict, records: list[FrameRecord], test_set: dict, suite_path: Path,
                        out_path: Path, clean_groups: dict | None = None) -> dict:
    """Evaluate `best` on `records` once per test of the suite, write `out_path` (JSON) and return its content.

    `clean_groups`: the clean result of the same model on the same frames (`evaluate_test(...)["groups"]`);
    computed here with one extra pass if not given.
    """
    tests, sha = load_robustness_suite(suite_path)
    logger.info(f"Robustness suite {Path(suite_path).name} (sha256 {sha[:12]}): {len(tests)} tests on {len(records)} frames")
    # every test is built (and checked on a test frame) before the first, slow inference pass
    pipelines = [(name, aug, build_albumentations(aug)) for name, aug in tests]
    val_args = {**val_args, "name": "robustness", "plots": False}
    if clean_groups is None:
        logger.info("Robustness: clean pass")
        clean_groups, _ = _evaluate_groups(best, val_args, records)

    results = {}
    for name, aug, compose in pipelines:
        logger.info(f"Robustness test '{name}'")
        step = EvalAlbuStep(compose, name) if compose is not None else None  # nothing enabled -> clean frames
        groups, ul_stats = _evaluate_groups(best, val_args, records, eval_albu=step)
        results[name] = {"pipeline": describe_pipeline(aug), "groups": groups,
                         "delta": _delta(groups, clean_groups), "ultralytics": ul_stats}

    result = {
        "suite": {"file": str(suite_path), "sha256": sha},
        "test_set": test_set,
        "false_alarm_conf": FA_CONF,
        "clean": clean_groups,
        "tests": results,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    header = (f"  {'test':<22}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}{'FA/frame':>10}"
              f"{'dmAP50':>9}{'dmAP50-95':>11}")
    lines = [header]
    rows = [("clean", clean_groups.get("all", {}), {})]
    rows += [(n, r["groups"].get("all", {}), r["delta"].get("all", {})) for n, r in results.items()]
    for name, g, d in rows:
        lines.append(f"  {name:<22}{_fmt(g.get('precision')):>8}{_fmt(g.get('recall')):>8}{_fmt(g.get('mAP50')):>8}"
                     f"{_fmt(g.get('mAP50-95')):>10}{_fmt(g.get('false_alarms_per_frame')):>10}"
                     f"{_fmt(d.get('mAP50')):>9}{_fmt(d.get('mAP50-95')):>11}")
    logger.info(f"Robustness metrics (all frames; per group in {out_path.name})\n" + "\n".join(lines))
    return result

"""Model setup for training: device and model weights.

From scratch: `model.weights: yolo26n.yaml` (architecture only, no weights are loaded; Ultralytics
initializes the layers, the input channels come from data.yaml `channels: 1`).
Pretrained: `model.weights: yolo26n.pt` (downloaded to weights/pretrained/).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import config
from log_setup import get_logger
from utils.utils import auto_device

logger = get_logger(__name__)


def resolve_device(device):
    """'auto' -> 0 (first NVIDIA GPU) if CUDA is available, else 'cpu'. Other values pass through."""
    if device in (None, "auto"):
        resolved = 0 if auto_device() == "cuda" else "cpu"
        logger.debug(f"device 'auto' -> {resolved!r}")
        return resolved
    return device


def resolve_model_weights(weights: str) -> str:
    """Bare model names (e.g. yolo26n.pt) are kept in weights/pretrained/ instead of the cwd."""
    p = Path(weights)
    local = config._resolve(p)
    if local.exists():
        logger.debug(f"model weights: local file {local}")
        return str(local)
    if p.suffix != ".pt" or len(p.parts) > 1:  # e.g. yolo26n.yaml (built-in architecture) -> let ultralytics resolve
        logger.debug(f"model weights: '{weights}' resolved by ultralytics")
        return weights
    from ultralytics.utils.downloads import attempt_download_asset

    target = config.WEIGHTS_DIR / "pretrained" / p.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        logger.info(f"Downloading pretrained weights {p.name} -> {target.parent}")
        downloaded = Path(attempt_download_asset(p.name))
        if downloaded.resolve() != target.resolve():
            shutil.move(str(downloaded), target)
    logger.debug(f"model weights: {target}")
    return str(target)

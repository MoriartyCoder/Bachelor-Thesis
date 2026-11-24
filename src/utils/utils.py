"""General helpers (device, seeding, environment/git info, config values for YAML/logs)."""
from __future__ import annotations

import platform
import random
import subprocess
import sys
from enum import Enum
from pathlib import Path

import numpy as np
import torch


def auto_device() -> str:
    """Pick ``cuda`` when available, else ``cpu``. Single source of truth."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def plain(o):
    """Config values -> YAML/log friendly (Path/Enum -> str)."""
    if isinstance(o, dict):
        return {k: plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, Enum):
        return o.value
    return str(o) if isinstance(o, Path) else o


def git(*args, cwd: Path | None = None) -> str:
    """Output of `git <args>`, or a short error text if git is missing / not a repo."""
    if cwd is None:
        from config import ROOT_DIR
        cwd = ROOT_DIR
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout
    except Exception as e:  # no git / not a repo
        return f"<git {' '.join(args)} failed: {e}>"


def env_info(device) -> str:
    import albumentations
    import cv2
    import ultralytics

    lines = [
        f"python        {sys.version.split()[0]} ({platform.platform()})",
        f"torch         {torch.__version__} (cuda build {torch.version.cuda})",
        f"ultralytics   {ultralytics.__version__}",
        f"albumentations {albumentations.__version__}",
        f"opencv        {cv2.__version__}",
        f"device        {device}",
    ]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            lines.append(f"gpu{i}          {prop.name}, {prop.total_memory / 2**30:.1f} GiB")
    else:
        lines.append("gpu           none (CUDA not available)")
    return "\n".join(lines)

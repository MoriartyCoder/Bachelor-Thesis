import torch

def auto_device() -> str:
    """Pick ``cuda`` when available, else ``cpu``. Single source of truth."""
    return "cuda" if torch.cuda.is_available() else "cpu"
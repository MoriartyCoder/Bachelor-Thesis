from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
LOGS_DIR = ROOT_DIR / "logs"
WEIGHTS_DIR = ROOT_DIR / "weights"
CONFIGS_DIR = ROOT_DIR / "configs"


ENSURE_DIRS_TO_EXIST = [DATA_DIR, OUTPUT_DIR]
for dir in ENSURE_DIRS_TO_EXIST:
    dir.mkdir(exist_ok=True)

TRAIN_MEAN = 82.34 # 82.34215282752244
TRAIN_STD  = 50.87 # 50.86908839649263


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class DatasetName(str, Enum):
    """All known datasets. Value = folder name in data/ (and name of its .zip).

    A member is a singleton, so every FrameRecord only holds a reference to it.
    New dataset: add a member here (+ an entry in PREPARE_IDS if it has a zip to extract).
    """
    ANTI_UAV_410 = "Anti-UAV410"
    CST_ANTI_UAV = "CST-AntiUAV"

    def __str__(self) -> str:  # "Anti-UAV410" instead of "DatasetName.ANTI_UAV_410"
        return self.value

    @property
    def root(self) -> Path:
        return DATA_DIR / self.value

    @property
    def zip_path(self) -> Path:
        return DATA_DIR / f"{self.value}.zip"


# Short ids for `python src/main.py -p <id>`
PREPARE_IDS: dict[int, DatasetName] = {
    100: DatasetName.CST_ANTI_UAV,
    410: DatasetName.ANTI_UAV_410,
}


OPTIMIZER_REGISTRY = []
SCHEDULER_REGISTRY = []


# ---------------------------------------------------------------------------
# Run configuration (YAML, see configs/template.yaml)
# ---------------------------------------------------------------------------

# Every Ultralytics-internal augmentation, forced off. Merged LAST into the
# trainer overrides so no config can switch one on. Albumentations (see
# augmentations/pipeline.py) is the only augmentation source.
ULTRALYTICS_AUG_OFF: dict[str, Any] = {
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
    "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0, "perspective": 0.0,
    "flipud": 0.0, "fliplr": 0.0, "bgr": 0.0,
    "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
    "close_mosaic": 0, "multi_scale": 0.0, "erasing": 0.0,
}

# Keys the run code sets itself; they must not appear in the `train:` section.
_MANAGED_TRAIN_KEYS = {
    "model", "data", "imgsz", "project", "name", "exist_ok", "save_period", "seed", "deterministic",
}

_TOP_LEVEL_KEYS = {"run", "datasets", "sampling", "model", "train", "augmentation"}
SPLITS = ("train", "val", "test")


class ConfigError(ValueError):
    """Invalid run configuration."""


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT_DIR / p).resolve()


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a run config YAML. Relative paths are resolved against ROOT_DIR."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    unknown = set(cfg) - _TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"Unknown top-level keys {sorted(unknown)}. Allowed: {sorted(_TOP_LEVEL_KEYS)}")
    missing = {"datasets", "sampling", "model", "train"} - set(cfg)
    if missing:
        raise ConfigError(f"Missing sections: {sorted(missing)}")

    run = cfg.setdefault("run", {})
    run.setdefault("seed", 42)
    run.setdefault("deterministic", True)
    run.setdefault("checkpoint_every_n_epochs", -1)
    run.setdefault("test_after_training", True)
    run["output_root"] = _resolve(run.get("output_root", WEIGHTS_DIR))

    if not cfg["datasets"]:
        raise ConfigError("`datasets` is empty.")
    for ds in cfg["datasets"]:
        if "name" not in ds:
            raise ConfigError(f"Dataset entry {ds} is missing `name`.")
        try:
            ds["name"] = DatasetName(ds["name"])
        except ValueError:
            raise ConfigError(
                f"Unknown dataset '{ds['name']}'. Known: {[d.value for d in DatasetName]} "
                "(add new ones to config.DatasetName)."
            ) from None
        ds.setdefault("format", "default")
        ds.setdefault("enabled", True)
        ds["root"] = _resolve(ds["root"]) if "root" in ds else ds["name"].root

    for split in SPLITS:
        if split not in cfg["sampling"]:
            raise ConfigError(f"`sampling.{split}` is missing.")
        if "method" not in cfg["sampling"][split]:
            raise ConfigError(f"`sampling.{split}.method` is missing.")

    if "weights" not in cfg["model"]:
        raise ConfigError("`model.weights` is missing (e.g. yolo26n.pt or a model .yaml).")
    cfg["model"].setdefault("imgsz", 640)

    train = cfg["train"] or {}
    cfg["train"] = train
    forbidden = set(train) & set(ULTRALYTICS_AUG_OFF)
    if forbidden:
        raise ConfigError(
            f"Ultralytics augmentation keys {sorted(forbidden)} are not allowed in `train:`. "
            "They are always disabled; use the `augmentation:` section instead."
        )
    managed = set(train) & _MANAGED_TRAIN_KEYS
    if managed:
        raise ConfigError(
            f"Keys {sorted(managed)} are set from other sections (run/model) and must not appear in `train:`."
        )
    train.setdefault("device", "auto")

    aug = cfg.setdefault("augmentation", {})
    aug.setdefault("bbox", {})
    aug.setdefault("pipeline", [])
    for entry in aug["pipeline"]:
        if "name" not in entry:
            raise ConfigError(f"Augmentation entry {entry} is missing `name`.")
        entry.setdefault("enabled", True)
        entry.setdefault("p", 1.0)
        entry.setdefault("params", {})

    return cfg


TEMPLATE_CONFIG = {
    "lr": 1e-3,
    "batch_size": 10,
    "optimizer": "Adam",
    "dataset": 410,
    "model": "Yolo",
    "save_point_interval": 10,
    "save_last_point": True, # Saves the last epoch and overwrites it after the current one is finished
    "loss": 0
}

'''
@dataclass
class TrainConfig:


    # Reproducibility / safety
    seed: int = 42
    deterministic: bool = True
    strict_finite_loss: bool = True


    # Model
    model_name: str = "srcnn3d"
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    output_patch_shape: tuple[int, int, int] = DEFAULT_OUTPUT_SHAPE



    # Training loop
    batch_size: int = 5
    num_epochs: int = 20
    num_workers: int = 0
    log_interval: int = 1

    # Loss + optimizer + scheduler (modular: swap names, kwargs stay JSON)
    loss_name: str = "masked_mse"
    loss_kwargs: dict[str, Any] = field(default_factory=dict)
    optimizer_name: str = "adam"
    learning_rate: float = 1e-3
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    # Gradient-norm clipping (training stability).
    # Purpose: cap the global L2 norm of the gradients each step so a single
    #   bad batch cannot blow up the Adam update and wreck the weights. The
    #   srcnn3d runs show exactly that failure -- a mid-training loss spike
    #   (train_loss ~0.005 -> ~0.035) that costs many epochs to recover from.
    # Effect: when set, ``torch.nn.utils.clip_grad_norm_`` runs after
    #   ``loss.backward()`` and before ``optimizer.step()`` in the train loop.
    # Influences: only the optimizer step; metrics/checkpoints are unchanged.
    # Change guidance: ``None`` (default) disables clipping so existing runs
    #   and resumes reproduce bit-for-bit. Typical stabilising value is 1.0;
    #   lower it (e.g. 0.5) if spikes persist, raise it if training stalls.
    grad_clip_norm: float | None = None
    scheduler_name: str = "plateau"
    scheduler_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"factor": 0.5, "patience": 3}
    )

    # Logging
    tensorboard: bool = True


def to_json(config: TrainConfig, path: Path) -> None:
    """Write ``config`` to ``path`` as pretty-printed JSON.

    Atomic via tmp + replace so an interrupt cannot leave a half-written
    file that breaks a later resume.
    """
    payload = _config_to_dict(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def from_json(path: Path) -> TrainConfig:
    """Load an ``TrainConfig`` from a JSON file written by ``to_json``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return _config_from_dict(raw)


def _config_to_dict(config: TrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["manifest_path"] = str(config.manifest_path)
    payload["run_root"] = str(config.run_root)
    payload["output_patch_shape"] = list(config.output_patch_shape)
    payload["patch_hr_shape"] = list(config.patch_hr_shape)
    return payload


def _config_from_dict(raw: dict[str, Any]) -> TrainConfig:
    kwargs: dict[str, Any] = dict(raw)
    if "manifest_path" in kwargs:
        kwargs["manifest_path"] = Path(kwargs["manifest_path"])
    if "run_root" in kwargs:
        kwargs["run_root"] = resolve_run_root(kwargs["run_root"])
    if "output_patch_shape" in kwargs:
        kwargs["output_patch_shape"] = tuple(kwargs["output_patch_shape"])
    if "patch_hr_shape" in kwargs:
        kwargs["patch_hr_shape"] = tuple(kwargs["patch_hr_shape"])
    if kwargs.get("loss_kwargs") is None:
        kwargs["loss_kwargs"] = {}
    return TrainConfig(**kwargs)
'''


'''
def validate(config: TrainConfig) -> None:
    """Fail fast on inconsistent or impossible configurations.

    Performs registry lookups via late imports to avoid a circular import
    chain (``models``/``losses``/``components`` all read ``TrainConfig`` types).
    """
    from sr.components import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
    from sr.losses import loss_names_for_validation
    from sr.models import MODEL_REGISTRY

    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.num_epochs < 1:
        raise ValueError("num_epochs must be >= 1")
    if config.num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if config.log_interval < 1:
        raise ValueError("log_interval must be >= 1")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if config.grad_clip_norm is not None and config.grad_clip_norm <= 0:
        raise ValueError(
            "grad_clip_norm must be > 0 when set, or None to disable clipping."
        )
    if not 0.0 < config.train_split <= 1.0:
        raise ValueError("train_split must be in (0, 1]")
    if config.source_voxel_mm <= 0 or config.target_voxel_mm <= 0:
        raise ValueError("source_voxel_mm and target_voxel_mm must be > 0")
    if len(config.output_patch_shape) != 3:
        raise ValueError("output_patch_shape must have 3 entries (D, H, W)")
    if len(config.patch_hr_shape) != 3:
        raise ValueError("patch_hr_shape must have 3 entries (D, H, W)")
    if config.patches_per_volume < 1:
        raise ValueError("patches_per_volume must be >= 1")

    if config.model_name == "srcnn3d_patch":
        for axis, edge in enumerate(config.patch_hr_shape):
            if edge < MIN_PATCH_HR_EDGE:
                raise ValueError(
                    f"patch_hr_shape[{axis}]={edge} must be >= {MIN_PATCH_HR_EDGE} "
                    f"so valid conv output has at least one voxel per axis"
                )
            if edge > config.output_patch_shape[axis]:
                raise ValueError(
                    f"patch_hr_shape[{axis}]={edge} exceeds "
                    f"output_patch_shape[{axis}]={config.output_patch_shape[axis]}"
                )

    if config.model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name '{config.model_name}'. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    if config.loss_name not in loss_names_for_validation():
        raise ValueError(
            f"Unknown loss_name '{config.loss_name}'. "
            f"Available: {sorted(loss_names_for_validation())}"
        )
    _validate_loss_kwargs(config)
    if config.optimizer_name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer_name '{config.optimizer_name}'. "
            f"Available: {sorted(OPTIMIZER_REGISTRY)}"
        )
    if config.scheduler_name not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler_name '{config.scheduler_name}'. "
            f"Available: {sorted(SCHEDULER_REGISTRY)}"
        )

    manifest = Path(config.manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest_path does not exist: {manifest}")
'''
'''
def seed_everything(seed: int, deterministic: bool) -> None:
    """Seed Python/NumPy/torch and toggle deterministic backends.

    Calling this once in the main process is enough; DataLoader workers
    are seeded separately via ``worker_init_fn`` in ``sr.data``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
'''

'''
def summary(config: TrainConfig) -> str:
    """Human-readable, one-screen block of the active configuration.

    Used right after ``validate`` so users see exactly what the run will do
    before any data is read. The order mirrors the dataclass for greppability.
    """
    lines = [
        "Configuration",
        f"  seed              = {config.seed}",
        f"  deterministic     = {config.deterministic}",
        f"  strict_finite     = {config.strict_finite_loss}",
        f"  model_name        = {config.model_name}",
        f"  model_kwargs      = {config.model_kwargs}",
        f"  output_patch      = {tuple(config.output_patch_shape)}",
        f"  train_split       = {config.train_split}",
        f"  batch_size        = {config.batch_size}",
        f"  num_epochs        = {config.num_epochs}",
        f"  num_workers       = {config.num_workers}",
        f"  log_interval      = {config.log_interval}",
        f"  loss_name         = {config.loss_name}",
        f"  loss_kwargs       = {config.loss_kwargs}",
        f"  optimizer_name    = {config.optimizer_name}",
        f"  learning_rate     = {config.learning_rate}",
        f"  optimizer_kwargs  = {config.optimizer_kwargs}",
        f"  grad_clip_norm    = {config.grad_clip_norm}",
        f"  scheduler_name    = {config.scheduler_name}",
        f"  scheduler_kwargs  = {config.scheduler_kwargs}",
        f"  tensorboard       = {config.tensorboard}",
    ]
    return "\n".join(lines)
'''

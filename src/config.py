import hashlib
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
TEST_SET_FILE = CONFIGS_DIR / "test_set.csv"  # frozen test frames (python src/main.py -c <cfg> --freeze-test)
ROBUSTNESS_SUITE_FILE = CONFIGS_DIR / "robustness.yaml"  # fixed corruption tests (training/evaluate.py)
LIBRARY_DIR = DATA_DIR / "library"  # PGDA target library (data_handler/target_library.py), not created on import


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
# data_handler/augmentation.py) is the only augmentation source.
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

# Sections of `augmentation:` (registries in src/augmentations/<section>.py). Default order of
# `augmentation.order`: copy-paste builds the scene first, the IR camera effects come last.
AUG_SECTION_ORDER = ("sota", "default", "edge_cases", "ir")
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
    run["test_set"] = _resolve(run.get("test_set", TEST_SET_FILE))
    run["output_root"] = _resolve(run.get("output_root", WEIGHTS_DIR))
    suite = run.get("robustness_suite")  # null = no robustness evaluation after training
    run["robustness_suite"] = _resolve(suite) if suite else None

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

    # sampling.test is only used by --freeze-test; training always evaluates the frozen test set (run.test_set)
    for split in SPLITS:
        if split not in cfg["sampling"]:
            if split == "test":
                continue
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

    cfg["augmentation"] = cfg.get("augmentation") or {}
    _validate_augmentation(cfg["augmentation"])
    return cfg


def _check_p(p: Any, where: str) -> None:
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= float(p) <= 1.0:
        raise ConfigError(f"{where}: `p` must be a number in [0, 1], got {p!r}.")


def _validate_entry(entry: Any, where: str) -> None:
    if not isinstance(entry, dict) or "name" not in entry:
        raise ConfigError(f"Augmentation entry {entry} in {where} is missing `name`.")
    unknown = set(entry) - {"name", "enabled", "p", "params"}
    if unknown:
        raise ConfigError(f"{where}.{entry['name']}: unknown keys {sorted(unknown)} (allowed: name, enabled, p, params).")
    entry.setdefault("enabled", True)
    entry.setdefault("p", 1.0)
    entry["params"] = entry.get("params") or {}
    _check_p(entry["p"], f"{where}.{entry['name']}")
    if not isinstance(entry["params"], dict):
        raise ConfigError(f"{where}.{entry['name']}: `params` must be a mapping, got {entry['params']!r}.")


def _validate_augmentation(aug: dict) -> None:
    """Sections, order, one_of groups and entries of the `augmentation:` section (in place, with defaults)."""
    if "pipeline" in aug:
        raise ConfigError(
            "`augmentation.pipeline` is the old flat layout. Put the entries into the sections "
            f"{list(AUG_SECTION_ORDER)} (`augmentation.<section>.pipeline`), see configs/template.yaml."
        )
    unknown = set(aug) - {"bbox", "order", *AUG_SECTION_ORDER}
    if unknown:
        raise ConfigError(f"Unknown keys {sorted(unknown)} in `augmentation`. "
                          f"Allowed: bbox, order, {', '.join(AUG_SECTION_ORDER)}")
    aug.setdefault("bbox", {})
    order = aug.setdefault("order", list(AUG_SECTION_ORDER))
    if sorted(order) != sorted(AUG_SECTION_ORDER):
        raise ConfigError(f"`augmentation.order` must list each section exactly once: {list(AUG_SECTION_ORDER)} "
                          f"in any order, got {order}")

    for section in AUG_SECTION_ORDER:
        sec = aug.get(section) or {}
        aug[section] = sec
        unknown = set(sec) - {"enabled", "pipeline"}
        if unknown:
            raise ConfigError(f"Unknown keys {sorted(unknown)} in `augmentation.{section}` (allowed: enabled, pipeline).")
        sec.setdefault("enabled", True)
        sec["pipeline"] = sec.get("pipeline") or []
        for item in sec["pipeline"]:
            if isinstance(item, dict) and "one_of" in item:
                group = item["one_of"] or {}
                item["one_of"] = group
                where = f"augmentation.{section}.one_of"
                if set(item) != {"one_of"}:
                    raise ConfigError(f"A one_of item in augmentation.{section} must only contain `one_of:` "
                                      f"(put enabled/p/pipeline inside it), got keys {sorted(item)}.")
                unknown = set(group) - {"enabled", "p", "pipeline"}
                if unknown:
                    raise ConfigError(f"Unknown keys {sorted(unknown)} in {where} (allowed: enabled, p, pipeline).")
                group.setdefault("enabled", True)
                group.setdefault("p", 1.0)
                _check_p(group["p"], where)
                group["pipeline"] = group.get("pipeline") or []
                for member in group["pipeline"]:
                    if isinstance(member, dict) and "one_of" in member:
                        raise ConfigError(f"{where}: one_of groups cannot be nested.")
                    _validate_entry(member, where)
            else:
                _validate_entry(item, f"augmentation.{section}")


# Sections a robustness test may use: corruptions only (no copy-paste, which also needs the train records).
ROBUSTNESS_SECTIONS = tuple(s for s in AUG_SECTION_ORDER if s != "sota")


def load_robustness_suite(path: str | Path) -> tuple[list[tuple[str, dict]], str]:
    """The tests of a robustness suite YAML (configs/robustness.yaml) and the file's sha256.

    Every test becomes a complete, validated `augmentation` dict (same entry/one_of syntax as the run config,
    sections in AUG_SECTION_ORDER), so data_handler.augmentation.build_albumentations can build it unchanged.
    """
    path = _resolve(path)
    if not path.exists():
        raise ConfigError(f"Robustness suite {path} not found.")
    raw = path.read_bytes()
    suite = yaml.safe_load(raw) or {}
    unknown = set(suite) - {"bbox", "tests"}
    if unknown:
        raise ConfigError(f"{path.name}: unknown keys {sorted(unknown)} (allowed: bbox, tests).")
    bbox = suite.get("bbox") or {}
    tests, names = [], set()
    for i, test in enumerate(suite.get("tests") or []):
        if not isinstance(test, dict) or not test.get("name"):
            raise ConfigError(f"{path.name}: test #{i + 1} is missing `name`.")
        name = str(test["name"])
        if name in names:
            raise ConfigError(f"{path.name}: test name '{name}' is used twice.")
        names.add(name)
        sections = set(test) - {"name"}
        bad = sections - set(ROBUSTNESS_SECTIONS)
        if bad:
            raise ConfigError(f"{path.name}, test '{name}': sections {sorted(bad)} are not allowed "
                              f"(allowed: {', '.join(ROBUSTNESS_SECTIONS)}).")
        aug = {"bbox": dict(bbox), "order": list(AUG_SECTION_ORDER)}
        aug.update({s: {"enabled": s in sections, "pipeline": test.get(s) or []} for s in AUG_SECTION_ORDER})
        try:
            _validate_augmentation(aug)
        except ConfigError as e:
            raise ConfigError(f"{path.name}, test '{name}': {e}") from e
        tests.append((name, aug))
    if not tests:
        raise ConfigError(f"{path.name}: `tests` is empty.")
    return tests, hashlib.sha256(raw).hexdigest()


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

# Drone detection in thermal IR video (YOLO + data augmentation)

Bachelor thesis: UAV detection in thermal infrared video with Ultralytics YOLO.
The subject is **data augmentation**: all Ultralytics augmentations are always off,
and only the Albumentations pipeline in the run config changes between experiments.

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Everything goes through `src/main.py`

```bash
python src/main.py --help                               # all options + examples

python src/main.py -p 410 100                           # prepare datasets (see below)
python src/main.py -l                                   # (re)build the PGDA target library
python src/main.py -c configs/template.yaml --freeze-test   # fix the test set ONCE (configs/test_set.csv)
python src/main.py -c configs/exp.yaml -t               # train
python src/main.py -c configs/exp.yaml -t --dry-run     # quick end-to-end check
python src/main.py -c configs/exp.yaml -t -v            # train with detailed DEBUG log
python src/main.py -r weights/<ts>                      # continue an interrupted run (<ts> = run folder name)
python src/main.py -e weights/<ts>                      # robustness test of a finished run
```

| Option | Meaning |
|---|---|
| `-c`, `--config YAML` | run config (needed for `-t`) |
| `-t`, `--train` | train YOLO with the config, then evaluate on the frozen test set |
| `--freeze-test` | with `-c`: sample the test split once (`sampling.test`) and write it to `run.test_set` (default `configs/test_set.csv`); never overwrites |
| `-r`, `--resume RUN_DIR` | continue an interrupted run from `RUN_DIR/weights/last.pt` with its own `config.yaml` (same folder, same frames), then test |
| `-e`, `--evaluate RUN_DIR` | robustness test of a finished run: `RUN_DIR/weights/best.pt` on the frozen test set, clean and per test of the suite → `RUN_DIR/robustness_metrics.json` |
| `--suite YAML` | with `-e`: robustness suite (default: the run's `run.robustness_suite`, else `configs/robustness.yaml`) |
| `--dry-run` | with `-t`: 32 frames per split, 1 CPU epoch, augmented previews, output in `output/dryrun/<ts>/`; with `-e`: only those 32 test frames, on the CPU |
| `-p`, `--prepare ID [ID ...]` | extract dataset zips: `100` = CST-AntiUAV, `410` = Anti-UAV410 |
| `-l`, `--build-library` | rebuild `data/library/` (PGDA target library) with the default `LibraryConfig` |
| `-v`, `--verbose` | DEBUG log instead of INFO |

If you give several actions, they run in the order `-p`, `-l`, `--freeze-test`, `-t`, `-r`, `-e`.

## Data

1. Download the zips manually into `data/`: `data/Anti-UAV410.zip`, `data/CST-AntiUAV.zip`.
2. Extract them with `python src/main.py -p 410 100`. **This deletes and recreates** `data/<dataset>/`.

Training never extracts anything. If a dataset or split is missing, it is skipped with a
warning. The same split of all enabled datasets is combined (train = every `train/`, …).

`data/library/` is the target library of `PGDAA` (build it before training with PGDAA). It holds
masked cutouts (pseudo masks) of the train targets of all datasets: `targets/*.png` (gray + alpha, the
background is black and transparent), `library.csv` and `build_info.json`. Build it with
`python src/main.py -l` (default settings) or in notebook 05 with `build_library(...)` from
`src/data_handler/target_library.py` (settings in `LibraryConfig`; reading the library and the
Gaussian paste are in `src/augmentations/sota.py`). Both **delete and recreate** the
folder, but only if it is a library (contains `build_info.json`).

## Experiments

```bash
cp configs/template.yaml configs/my_exp.yaml   # edit: datasets, sampling, model, train, augmentation
python src/main.py -c configs/my_exp.yaml -t --dry-run   # check, look at output/dryrun/<ts>/previews/
python src/main.py -c configs/my_exp.yaml -t
tensorboard --logdir weights                             # watch / compare training curves
```

Then compare the finished runs in `notebooks/08_compare_runs.ipynb` (see Notebooks).

Every run gets a folder `weights/<YYYYMMDD-HHMMSS>_<config name>/` (start time + file name of the run
config without `.yaml`, e.g. `weights/20261126-093000_my_exp/`; `<ts>` below means this folder name) with config snapshots, git/env info,
manifests of the used frames, `run.log`, the Ultralytics results, `weights/{best,last,epochN}.pt`
and `test_metrics.json` (+ `robustness_metrics.json`, see below).

**Test set.** Every run is evaluated on the same frozen frames: `configs/test_set.csv` (key
`run.test_set`). Create it once before the first experiment with
`python src/main.py -c configs/template.yaml --freeze-test` (uses `sampling.test`; the file is
never overwritten; commit it). Without the file, training stops with this hint.
`test_metrics.json` has the test-set fingerprint (`sha256`) and one entry per group: `all`, each
dataset, each target size (`sqrt(w*h)`: <=10, 10-30, 30-50, >50 px) and `no target`, each with
frames, targets, precision, recall, mAP50, mAP50-95 and false alarms (predictions with
confidence >= 0.25 on frames without target). The same table is in `run.log`.

**Robustness test.** `configs/robustness.yaml` is a fixed list of corruption tests (sensor noise, blur,
grey-value drift, flip, …; same entry syntax as the `default` / `ir` / `edge_cases` sections, no `sota`).
Each test corrupts every frame of the frozen test set; the random state depends only on the test name and
the frame, so every model sees exactly the same corrupted frames. With `run.robustness_suite:
configs/robustness.yaml` it runs after the test evaluation (the suite is checked before training);
`python src/main.py -e weights/<ts>` runs it for a finished run (appends to its `run.log`, replaces
`robustness_metrics.json`). The JSON holds the suite's `sha256`, the clean groups, and per test the
pipeline, the same groups as `test_metrics.json` and `delta` = test − clean; `run.log` shows a table
for all frames. It is a diagnosis only: model selection stays on the clean val set. A test whose effect
is also in the training pipeline measures in-distribution robustness. Keep the suite fixed once runs are
compared (renaming a test changes its corrupted frames).

**Interrupted run.** `python src/main.py -r weights/<ts>` continues from `last.pt` in the same folder.
The frame lists are rebuilt from the run's `config.yaml` and must match its manifests.

**Config errors** (unknown augmentation, misspelled or invalid parameter, `p` outside [0, 1]) stop the
run before training: every enabled transform is built and run once on a real training frame.

The `augmentation:` section of the config has four sections, run in the order of
`augmentation.order` (default `[sota, default, edge_cases, ir]`):

| Section | Registry (allowed names) | Content |
|---|---|---|
| `sota` | `src/augmentations/sota.py` | `ACB` (Tong et al. 2024), `PGDAA` (PGDA adapted, Wang et al. 2022; needs `-l`) |
| `default` | `src/augmentations/default.py` | generic Albumentations transforms, grey-value `histogram_shift` / `histogram_compression` |
| `edge_cases` | `src/augmentations/edge_cases.py` | difficult detection cases (list still empty) |
| `ir` | `src/augmentations/ir.py` | IR camera effects: `optical_blur`, `motion_blur`, `fpn`, `stripe_noise`, `netd`, `salt_and_pepper` |

A section only accepts the names of its registry. `enabled` switches a section, a group or a
single entry on/off. Mutually exclusive augmentations go into a group:

```yaml
- one_of:          # albumentations.OneOf: with p = 0.2 exactly ONE enabled member runs
    p: 0.2
    pipeline:      # members' p = relative weights
      - {name: optical_blur, p: 1.0, params: {sigma_0: 1.2, rho: 1.0}}
      - {name: motion_blur,  p: 1.0, params: {sigma_0: 2.0, rho: 0.15, theta: 0.0}}
```

All augmentations work on grayscale uint8 HxW frames (also when called directly in a notebook,
e.g. on a `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` frame) and return uint8; the model gets 1 channel.
`ACB` pastes augmented copies of training targets with their own boxes; its parameters are
explained in `configs/template.yaml`. `PGDAA` (`src/augmentations/sota.py`) follows PGDA's Algorithm 1:
with probability p the sample is replaced by a synthetic image (pure-background subfigure + library
targets, scaled to the train size prior A and adjusted to the SCR prior S). Every deviation from the
paper and how it is handled: `src/augmentations/PGDAA_deviations.md` (for ACB:
`src/augmentations/ACB_deviations.md`). `<run folder>/augmentation.txt` and the dry-run previews show
the built pipeline (order, sections, groups, on/off) and the boxes.

Model: `model.weights: yolo26n.pt` = pretrained, `yolo26n.yaml` = from scratch (no weights loaded,
Ultralytics' default initialization). The input is only divided by 255 (no mean/std normalization).

## Code layout (`src/`)

- `augmentations/`: definitions only, one file per section (`default.py`, `ir.py`, `edge_cases.py`,
  `sota.py`) + `uint8.py` (uint8 conventions).
- `data_handler/`: dataset preparation, target library build, frame loading/sampling, the
  Albumentations builder (`augmentation.py`) and the Ultralytics training dataset (`yolo_dataset.py`).
- `models/`: device and model weights (`yolo.py`).
- `training/`: run orchestration (`run.py`), the Ultralytics trainer/validator (`trainer.py`) and the
  test / robustness evaluation (`evaluate.py`).
- `utils/`: helpers (seeding, env/git info).
- `main.py`, `cli.py`, `config.py`, `log_setup.py`: entry point, CLI, config/validation, logging.

## Logging

- **INFO** (default): the main steps with frame counts, paths, durations and metrics.
- **DEBUG** (`-v`): everything else as well: every sequence loaded, the sampler,
  every transform, the resolved config and the Ultralytics overrides. Each line shows
  its module and line number.
- Output goes to the console, to `logs/output.log` (all calls, appended) and to
  `<run folder>/run.log` (only that run). The two log files also contain Ultralytics' own messages.
- In code, use `from log_setup import get_logger; logger = get_logger(__name__)`.

## Notebooks

Run them from inside `notebooks/`. Each one starts with `%run 00_utils.ipynb`.
The datasets are selected in `00_utils.ipynb`:

```python
DATASETS = [DatasetName.ANTI_UAV_410, DatasetName.CST_ANTI_UAV]   # both
DATASETS = [DatasetName.CST_ANTI_UAV]                               # only one
```

- `for label, dirs in selections("train"): ...` runs over each dataset on its own and then,
  if there are several, over all of them together. Notebooks 01 and 02 work this way.
- `AntiUAVDataset(DATASETS, split="train")` loads the frames of all selected datasets in
  that order. The loading cell prints the index range of each dataset.
- `03_augmentation.ipynb` shows single Albumentations transforms and IR effects on one training
  frame (`IMAGE_ID`), each wrapped in its own `Compose` (scratch notebook).
- `04_parameter.ipynb` has sliders for the parameter ranges of the single effects
  (NETD noise, FPN / stripe noise, optical / motion blur, ...), an ACB section (pastes into one frame, sliders for
  the source size limit, N, M, ops, context, fusion) and a PGDAA section (a synthetic image with sliders for its parameters and priors;
  needs the library, `python src/main.py -l`).
- `06_measure.ipynb` measures the data before choosing augmentations: sizes, presence, contrast (SCR),
  distractors, clutter, HUD band and split overlap. Each section first defines its functions, then
  applies them. `per_dataset(report, df)` gives one table with each dataset and the combination.
  Settings (`N_PER_SEQ`, `HUD_ROWS`, `SPLITS`) are at the top. The frame table is cached in
  `output/frames_<datasets>.pkl`; delete that file to rebuild it.
- `07_generic_augmentation.ipynb` has sliders for generic augmentations on one training frame
  (`IMAGE_ID`, boxes drawn): box-safe random crop, scale jitter, small affine, Mosaic, MixUp
  (implemented only in the notebook, not in `src/`), polarity inversion and brightness / contrast / gamma
  (with the tone curve). Mosaic and MixUp take their extra frames from the train split, chosen by `seed`.
- `08_compare_runs.ipynb` compares finished runs: it reads `config.yaml`, `results.csv`, `test_metrics.json`
  and `robustness_metrics.json` of every run folder in `RUN_ROOTS` (default `weights/`; add
  `OUTPUT_DIR / "dryrun"` for dry runs), filtered with `INCLUDE` (regex on the config name). Tables: overview
  (epochs trained/planned, best epoch = the epoch of `best.pt`, early stop, enabled augmentations), test metrics
  of one group, one metric for all groups, mean ± std per experiment over seeds (runs with the same config
  name), robustness deltas. `check_comparable` warns if runs differ in more than augmentation and seed (test
  set, model, image size, `train:` parameters, data, suite). Plots: val mAP50-95 per epoch, metric per group,
  robustness delta. Section 6 exports the tables as CSV to `output/compare/`.
- `05_target_library.ipynb` (target library + PGDAA) uses the train split of all datasets and ignores
  `DATASETS`. It covers the Anti-UAV410 frame tags, the filter funnel, the mask check, the build of
  `data/library/`, the library statistics, a Gaussian-paste preview and PGDAA (priors A and S,
  synthetic images, ablation). All library settings are in `LibraryConfig`; `REBUILD = False` only
  loads the existing library.

## Adding a dataset

1. Add a member to `DatasetName` in `src/config.py`. Its value is the folder name in `data/`.
   If it comes as a zip, also add an id to `PREPARE_IDS`.
2. If its layout differs from Anti-UAV (`<split>/<seq>/*.jpg + IR_label.json`), add a loader
   with `@register_format("<name>")` in `src/data_handler/formats.py`.
3. Add it under `datasets:` in the run config (`name`, optionally `root` and `format`).

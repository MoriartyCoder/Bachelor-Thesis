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
python src/main.py -c configs/exp.yaml -t               # train
python src/main.py -c configs/exp.yaml -t --dry-run     # quick end-to-end check
python src/main.py -c configs/exp.yaml -t -v            # train with detailed DEBUG log
```

| Option | Meaning |
|---|---|
| `-c`, `--config YAML` | run config (needed for `-t`) |
| `-t`, `--train` | train YOLO with the config, then evaluate on the test split |
| `--dry-run` | with `-t`: 32 frames per split, 1 CPU epoch, augmented previews, output in `output/dryrun/<ts>/` |
| `-p`, `--prepare ID [ID ...]` | extract dataset zips: `100` = CST-AntiUAV, `410` = Anti-UAV410 |
| `-v`, `--verbose` | DEBUG log instead of INFO |

If you give `-p` and `-t` together, preparation runs first.

## Data

1. Download the zips manually into `data/`: `data/Anti-UAV410.zip`, `data/CST-AntiUAV.zip`.
2. Extract them with `python src/main.py -p 410 100`. **This deletes and recreates** `data/<dataset>/`.

Training never extracts anything. If a dataset or split is missing, it is skipped with a
warning. The same split of all enabled datasets is combined (train = every `train/`, …).

## Experiments

```bash
cp configs/template.yaml configs/my_exp.yaml   # edit: datasets, sampling, model, train, augmentation
python src/main.py -c configs/my_exp.yaml -t --dry-run   # check, look at output/dryrun/<ts>/previews/
python src/main.py -c configs/my_exp.yaml -t
tensorboard --logdir weights                             # compare runs
```

Every run gets a folder `weights/<YYYYMMDD-HHMMSS>/` with config snapshots, git/env info,
manifests of the used frames, `run.log`, the Ultralytics results, `weights/{best,last,epochN}.pt`
and `test_metrics.json`.

Augmentations are listed in `augmentation.pipeline` of the config. Each one is an Albumentations
transform name, an IR effect `ir.<name>` or `ACB`, and each can be switched on or off with `enabled`.
`ACB` (`src/augmentations/acb.py`, Tong et al. 2024) pastes augmented copies of training targets
with their own boxes; its parameters are explained in `configs/template.yaml`. The dry-run
previews show the pasted targets and their boxes.

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
- `notebooks/measure.py`: `build_frame_table([ds.root for ds in DATASETS])` adds a `dataset`
  column, and `per_dataset(size_report, df)` gives one table with each dataset and the combination.

## Adding a dataset

1. Add a member to `DatasetName` in `src/config.py`. Its value is the folder name in `data/`.
   If it comes as a zip, also add an id to `PREPARE_IDS`.
2. If its layout differs from Anti-UAV (`<split>/<seq>/*.jpg + IR_label.json`), add a loader
   with `@register_format("<name>")` in `src/data_handler/formats.py`.
3. Add it under `datasets:` in the run config (`name`, optionally `root` and `format`).

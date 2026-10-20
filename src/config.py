from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
LOGS_DIR = ROOT_DIR / "logs"

DATA_TEST_DIR = DATA_DIR / "Anti-UAV-RGBT" / "train"
DATA_VAL_DIR  = DATA_DIR / "Anti-UAV-RGBT" / "val"
DATA_TEST_DIR = DATA_DIR / "Anti-UAV-RGBT" / "test"

ENSURE_DIRS_TO_EXIST = [DATA_DIR, OUTPUT_DIR]
for dir in ENSURE_DIRS_TO_EXIST:
    dir.mkdir(exist_ok=True)

TRAIN_MEAN = 82.34 # 82.34215282752244
TRAIN_STD  = 50.87 # 50.86908839649263
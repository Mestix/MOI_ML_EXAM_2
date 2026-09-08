from pathlib import Path

DATA_DIR = Path("data")
TRAIN_FILE = DATA_DIR / "heart_big_train.parq"
TEST_FILE = DATA_DIR / "heart_big_test.parq"

TARGET_COLUMN = "target"
NUM_CLASSES = 5
CLASS_LABELS = ("N", "S", "V", "F", "Q")
ECG_FEATURES = 187
MATRIX_SHAPE = (16, 12)
VALIDATION_FRACTION = 0.2
DEFAULT_SEED = 42

RESULTS_FILE = Path("results/results.csv")
TUNING_RESULTS_FILE = Path("results/ray_tuning_results.csv")

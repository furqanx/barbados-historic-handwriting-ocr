"""Project-wide paths and configuration constants.

Keep this module dependency-light so it can be imported from scripts, notebooks,
and training code without side effects.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_COMPETITION_DIR = RAW_DIR / "road-barbados-historic-handwriting-challenge"
IMAGE_DIR = RAW_DIR / "images"

TRAIN_CSV = RAW_COMPETITION_DIR / "Train.csv"
TEST_CSV = RAW_COMPETITION_DIR / "Test.csv"
SAMPLE_SUBMISSION_CSV = RAW_COMPETITION_DIR / "SampleSubmission.csv"

INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
SPLITS_DIR = DATA_DIR / "splits"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
PYLAIA_OUTPUTS_DIR = OUTPUTS_DIR / "pylaia"
KRAKEN_OUTPUTS_DIR = OUTPUTS_DIR / "kraken"

TRAIN_MANIFEST = METADATA_DIR / "train_manifest.csv"
TEST_MANIFEST = METADATA_DIR / "test_manifest.csv"
CHAR_VOCAB_JSON = METADATA_DIR / "char_vocab.json"
PYLAIA_DATA_DIR = DATA_DIR / "pylaia"
KRAKEN_DATA_DIR = DATA_DIR / "kraken"
MODELS_DIR = PROJECT_ROOT / "models"
PYLAIA_MODELS_DIR = MODELS_DIR / "pylaia"
KRAKEN_MODELS_DIR = MODELS_DIR / "kraken"

ID_COL = "ID"
TARGET_COL = "Target"
FOLD_COL = "fold"

RANDOM_SEED = 42
N_FOLDS = 5

IMAGE_EXT = ".jpg"

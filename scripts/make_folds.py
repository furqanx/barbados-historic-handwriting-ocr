"""Create train/test manifests with image metadata and CV folds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import (
    IMAGE_DIR,
    N_FOLDS,
    RANDOM_SEED,
    SAMPLE_SUBMISSION_CSV,
    TEST_CSV,
    TEST_MANIFEST,
    TRAIN_CSV,
    TRAIN_MANIFEST,
)
from src.data.manifest import build_manifests, save_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument("--sample-submission", type=Path, default=SAMPLE_SUBMISSION_CSV)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--train-output", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--test-output", type=Path, default=TEST_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_manifest, test_manifest = build_manifests(
        image_dir=args.image_dir,
        train_path=args.train_csv,
        test_path=args.test_csv,
        sample_submission_path=args.sample_submission,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    paths = save_manifests(
        train_manifest,
        test_manifest,
        train_output=args.train_output,
        test_output=args.test_output,
    )

    print(f"Saved train manifest: {paths.train_manifest}")
    print(f"Saved test manifest: {paths.test_manifest}")
    print(f"Train rows: {len(train_manifest):,}")
    print(f"Test rows: {len(test_manifest):,}")
    print("Fold counts:")
    print(train_manifest["fold"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

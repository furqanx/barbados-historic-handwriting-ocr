"""Create train/test manifests with image metadata and CV folds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import N_FOLDS, RANDOM_SEED, TEST_MANIFEST, TRAIN_MANIFEST
from src.data.manifest import build_manifests, save_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--train-output", default=str(TRAIN_MANIFEST))
    parser.add_argument("--test-output", default=str(TEST_MANIFEST))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_manifest, test_manifest = build_manifests(
        n_folds=args.n_folds,
        seed=args.seed,
    )
    paths = save_manifests(
        train_manifest,
        test_manifest,
        train_output=Path(args.train_output),
        test_output=Path(args.test_output),
    )

    print(f"Saved train manifest: {paths.train_manifest}")
    print(f"Saved test manifest: {paths.test_manifest}")
    print(f"Train rows: {len(train_manifest):,}")
    print(f"Test rows: {len(test_manifest):,}")
    print("Fold counts:")
    print(train_manifest["fold"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

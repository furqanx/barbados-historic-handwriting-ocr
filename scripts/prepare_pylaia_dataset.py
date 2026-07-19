"""Prepare manifest CSVs as a PyLaia dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import PYLAIA_DATA_DIR, TEST_MANIFEST, TRAIN_MANIFEST  # noqa: E402
from src.pylaia.dataset import prepare_pylaia_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--test-manifest", type=Path, default=TEST_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-mode", choices=["L", "RGB"], default="L")
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PYLAIA_DATA_DIR / f"fold{args.fold}_h{args.image_height}"
    paths = prepare_pylaia_dataset(
        pd.read_csv(args.train_manifest),
        pd.read_csv(args.test_manifest),
        output_dir=output_dir,
        fold=args.fold,
        image_height=args.image_height,
        image_mode=args.image_mode,
        image_format=args.image_format,
        force=args.force,
    )
    print(f"Prepared PyLaia dataset: {paths.root}")
    print(f"Images: {paths.images_dir}")
    print(f"Symbols: {paths.syms}")
    print(f"Train table: {paths.train_txt}")
    print(f"Validation table: {paths.val_txt}")
    print(f"Test IDs: {paths.test_ids}")


if __name__ == "__main__":
    main()


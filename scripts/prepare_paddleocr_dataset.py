"""Prepare manifest CSVs as a PaddleOCR text recognition dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import PADDLEOCR_DATA_DIR, TEST_MANIFEST, TRAIN_MANIFEST  # noqa: E402
from src.paddleocr_rec.dataset import prepare_paddleocr_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--test-manifest", type=Path, default=TEST_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-mode", choices=["L", "RGB"], default="RGB")
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--path-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        suffix = f"h{args.image_height}" if args.image_height else "raw"
        output_dir = PADDLEOCR_DATA_DIR / f"fold{args.fold}_{suffix}"
    paths = prepare_paddleocr_dataset(
        pd.read_csv(args.train_manifest),
        pd.read_csv(args.test_manifest),
        output_dir=output_dir,
        fold=args.fold,
        image_height=args.image_height,
        image_mode=args.image_mode,
        image_format=args.image_format,
        path_mode=args.path_mode,
        force=args.force,
    )
    print(f"Prepared PaddleOCR dataset: {paths.root}")
    print(f"Train labels: {paths.train_labels}")
    print(f"Validation labels: {paths.val_labels}")
    print(f"Test images: {paths.test_images}")
    print(f"Character dictionary: {paths.character_dict}")


if __name__ == "__main__":
    main()


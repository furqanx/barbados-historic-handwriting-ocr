"""Audit target characters for PaddleOCR recognition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import TARGET_COL, TRAIN_MANIFEST  # noqa: E402
from src.paddleocr_rec.text import audit_paddleocr_characters  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_manifest)
    audit = audit_paddleocr_characters(train[TARGET_COL].tolist())
    print(f"Character count including space: {audit.character_count}")
    print(f"Has space: {audit.has_space}")
    print("Characters:")
    print(" ".join(repr(char) for char in audit.characters))


if __name__ == "__main__":
    main()


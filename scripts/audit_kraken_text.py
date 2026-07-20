"""Audit target characters under a Kraken Unicode normalization choice."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import TARGET_COL, TRAIN_MANIFEST  # noqa: E402
from src.htr.kraken.text import audit_characters  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument(
        "--unicode-normalization",
        choices=["preserve", "NFC", "NFKC", "NFD", "NFKD"],
        default="preserve",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_manifest)
    audit = audit_characters(
        train[TARGET_COL].tolist(),
        unicode_form=args.unicode_normalization,
    )
    print(f"Unicode normalization: {audit.unicode_form}")
    print(f"Character count: {audit.character_count}")
    print("Characters:")
    print(" ".join(repr(char) for char in audit.characters))


if __name__ == "__main__":
    main()


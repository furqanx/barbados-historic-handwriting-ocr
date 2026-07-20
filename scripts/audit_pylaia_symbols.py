"""Audit competition target characters against a PyLaia symbols file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import TARGET_COL, TRAIN_MANIFEST  # noqa: E402
from src.htr.pylaia.charset import audit_symbols, load_syms  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--syms", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_manifest)
    symbols = load_syms(args.syms)
    audit = audit_symbols(train[TARGET_COL].tolist(), symbols.keys())
    print(f"Symbols: {len(symbols):,}")
    print(f"Covered target characters: {len(audit.covered_characters):,}")
    print(f"Missing target characters: {len(audit.missing_characters):,}")
    if audit.missing_characters:
        print("Missing:")
        print(" ".join(repr(char) for char in audit.missing_characters))


if __name__ == "__main__":
    main()


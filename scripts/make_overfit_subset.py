"""Create a tiny manifest for small-subset overfit diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import METADATA_DIR, TRAIN_MANIFEST  # noqa: E402
from src.diagnostics.subset_overfit import write_overfit_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--valid-fold", type=int, default=0)
    parser.add_argument("--train-fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-chars", default="")
    parser.add_argument("--no-mirror-valid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        suffix = f"{args.n_samples}"
        if args.require_chars:
            suffix += "_rarechars"
        output = METADATA_DIR / f"overfit_manifest_{suffix}.csv"

    manifest = pd.read_csv(args.train_manifest)
    path = write_overfit_manifest(
        manifest,
        output,
        n_samples=args.n_samples,
        valid_fold=args.valid_fold,
        train_fold=args.train_fold,
        seed=args.seed,
        require_chars=args.require_chars,
        mirror_valid=not args.no_mirror_valid,
    )
    result = pd.read_csv(path)
    print(f"Saved overfit manifest: {path}")
    print(f"rows={len(result)} unique_ids={result['ID'].nunique()}")
    print(result.groupby("fold").size())


if __name__ == "__main__":
    main()

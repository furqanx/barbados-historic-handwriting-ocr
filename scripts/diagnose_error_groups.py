"""Group validation OCR errors by interpretable sample categories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import TARGET_COL, TRAIN_MANIFEST  # noqa: E402
from src.diagnostics.error_analysis import run_grouped_error_analysis  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "diagnostics" / "error_groups")
    parser.add_argument("--truth-col", default=TARGET_COL)
    parser.add_argument("--pred-col", default=TARGET_COL)
    parser.add_argument("--rare-chars", default="^*#|\\")
    parser.add_argument(
        "--require-all-truth",
        action="store_true",
        help="Fail if predictions do not cover every manifest row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    predictions = pd.read_csv(args.predictions)
    _, summary = run_grouped_error_analysis(
        manifest,
        predictions,
        output_dir=args.output_dir,
        truth_col=args.truth_col,
        pred_col=args.pred_col,
        rare_chars=args.rare_chars,
        allow_prediction_subset=not args.require_all_truth,
    )
    print(f"Saved: {args.output_dir / 'grouped_row_errors.csv'}")
    print(f"Saved: {args.output_dir / 'grouped_error_summary.csv'}")
    print(summary.sort_values(["group_name", "mean_row_cer"], ascending=[True, False]))


if __name__ == "__main__":
    main()

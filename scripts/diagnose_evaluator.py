"""Evaluate a validation prediction CSV with row-level diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import ID_COL, TARGET_COL, TRAIN_MANIFEST  # noqa: E402
from src.diagnostics.evaluator import evaluate_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "diagnostics" / "evaluation")
    parser.add_argument("--id-col", default=ID_COL)
    parser.add_argument("--truth-col", default=TARGET_COL)
    parser.add_argument("--pred-col", default=TARGET_COL)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--require-all-truth",
        action="store_true",
        help="Fail if predictions do not cover every truth row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = pd.read_csv(args.truth)
    predictions = pd.read_csv(args.predictions)
    summary, row_errors, confusions = evaluate_predictions(
        truth,
        predictions,
        output_dir=args.output_dir,
        id_col=args.id_col,
        truth_col=args.truth_col,
        pred_col=args.pred_col,
        normalize=not args.no_normalize,
        allow_prediction_subset=not args.require_all_truth,
    )
    print(
        f"rows={summary.rows} "
        f"wer={summary.wer:.5f} "
        f"cer={summary.cer:.5f} "
        f"score={summary.score:.5f} "
        f"exact_match_rate={summary.exact_match_rate:.5f}"
    )
    print(f"Saved row errors: {args.output_dir / 'row_errors.csv'}")
    print(f"Saved confusions: {args.output_dir / 'character_confusions.csv'}")
    print(f"Worst rows preview:\n{row_errors.sort_values('row_cer', ascending=False).head(10)}")
    print(f"Top confusions preview:\n{confusions.head(20)}")


if __name__ == "__main__":
    main()

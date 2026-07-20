"""Convert Kraken OCR text files to prediction and submission CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import ID_COL, OUTPUTS_DIR, SAMPLE_SUBMISSION_CSV, TARGET_COL  # noqa: E402
from src.evaluation.metrics import score_transcriptions  # noqa: E402
from src.htr.kraken.prediction_parser import (  # noqa: E402
    align_predictions_to_sample,
    load_kraken_prediction_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--submission-output", type=Path, default=None)
    parser.add_argument("--sample-submission", type=Path, default=SAMPLE_SUBMISSION_CSV)
    parser.add_argument("--reference-manifest", type=Path, default=None)
    parser.add_argument(
        "--output-unicode-normalization",
        choices=["preserve", "NFC", "NFKC", "NFD", "NFKD"],
        default="NFC",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_kraken_prediction_dir(
        args.prediction_dir,
        output_unicode_normalization=args.output_unicode_normalization,
    )
    if args.reference_manifest is not None:
        reference = pd.read_csv(args.reference_manifest)
        predictions = predictions.merge(
            reference[[ID_COL, TARGET_COL]].rename(columns={TARGET_COL: "reference"}),
            on=ID_COL,
            how="left",
        )
        scored = predictions["reference"].notna()
        if scored.any():
            score = score_transcriptions(
                predictions.loc[scored, "reference"].tolist(),
                predictions.loc[scored, TARGET_COL].tolist(),
            )
            print(
                f"wer={score.wer:.5f} cer={score.cer:.5f} score={score.score:.5f}"
            )

    predictions_output = args.predictions_output
    if predictions_output is None:
        predictions_output = (
            args.output_dir / "predictions" / f"{args.run_name}_{args.split}.csv"
        )
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_output, index=False)
    print(f"Saved predictions: {predictions_output}")

    if args.split == "test" and args.sample_submission.exists():
        sample = pd.read_csv(args.sample_submission)
        submission = align_predictions_to_sample(
            predictions[[ID_COL, TARGET_COL]],
            sample,
        )
        submission_output = args.submission_output
        if submission_output is None:
            submission_output = (
                args.output_dir / "submissions" / f"{args.run_name}_submission.csv"
            )
        submission_output.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(submission_output, index=False)
        print(f"Saved submission: {submission_output}")


if __name__ == "__main__":
    main()


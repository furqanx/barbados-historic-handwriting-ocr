"""Ensemble transcription prediction CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import OUTPUTS_DIR, SAMPLE_SUBMISSION_CSV  # noqa: E402
from src.inference.ensemble import (  # noqa: E402
    ensemble_predictions,
    evaluate_ensemble,
    evaluate_prediction_sets,
    load_named_predictions,
    make_submission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Prediction CSV in format name:path.csv. Repeat for each model.",
    )
    parser.add_argument(
        "--priority",
        nargs="*",
        default=None,
        help="Model names used to break voting ties, first name wins.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--run-name", default="ensemble")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sample-submission", type=Path, default=None)
    parser.add_argument("--submission-output", type=Path, default=None)
    parser.add_argument("--scores-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_sets = [load_named_predictions(spec) for spec in args.prediction]
    ensemble = ensemble_predictions(prediction_sets, priority=args.priority)

    output = args.output
    if output is None:
        output = args.output_dir / "predictions" / f"{args.run_name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(output, index=False)
    print(f"Saved ensemble predictions: {output}")

    scores = evaluate_prediction_sets(prediction_sets)
    ensemble_scores = evaluate_ensemble(
        ensemble,
        prediction_sets[0].predictions,
        name=args.run_name,
    )
    if not ensemble_scores.empty:
        scores = pd.concat([scores, ensemble_scores], ignore_index=True)
    if not scores.empty:
        scores_output = args.scores_output
        if scores_output is None:
            scores_output = args.output_dir / "predictions" / f"{args.run_name}_scores.csv"
        scores_output.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(scores_output, index=False)
        print(f"Saved ensemble scores: {scores_output}")
        print(scores.to_string(index=False))

    sample_submission = args.sample_submission
    if sample_submission is None and SAMPLE_SUBMISSION_CSV.exists():
        sample_submission = SAMPLE_SUBMISSION_CSV
    if sample_submission is not None:
        submission_output = args.submission_output
        if submission_output is None:
            submission_output = (
                args.output_dir / "submissions" / f"{args.run_name}_submission.csv"
            )
        sample = pd.read_csv(sample_submission)
        submission = make_submission(ensemble, sample)
        submission_output.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(submission_output, index=False)
        print(f"Saved ensemble submission: {submission_output}")


if __name__ == "__main__":
    main()

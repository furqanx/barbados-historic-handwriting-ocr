"""Run CTC checkpoint inference and create a submission file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import OUTPUTS_DIR, SAMPLE_SUBMISSION_CSV, TEST_MANIFEST  # noqa: E402
from src.ctc.predictor import (  # noqa: E402
    load_crnn_ctc_checkpoint,
    make_submission,
    predict_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=TEST_MANIFEST)
    parser.add_argument("--sample-submission", type=Path, default=SAMPLE_SUBMISSION_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--run-name", default="crnn_ctc")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else None
    model, tokenizer, checkpoint = load_crnn_ctc_checkpoint(
        args.checkpoint,
        device=device,
    )
    training_config = checkpoint.get("training_config", {})

    test_manifest = pd.read_csv(args.test_manifest)
    sample_submission = pd.read_csv(args.sample_submission)
    predictions = predict_manifest(
        model,
        tokenizer,
        test_manifest,
        batch_size=args.batch_size or int(training_config.get("batch_size", 8)),
        num_workers=args.num_workers
        if args.num_workers is not None
        else int(training_config.get("num_workers", 2)),
        target_height=int(training_config.get("target_height", 96)),
        max_width=training_config.get("max_width", 2048),
        autocontrast_cutoff=training_config.get("autocontrast_cutoff"),
        pad_value=float(training_config.get("pad_value", 1.0)),
        width_multiple=int(training_config.get("width_multiple", 4)),
        use_data_parallel=not args.no_data_parallel,
        device=device,
    )
    predictions_output = args.predictions_output
    if predictions_output is None:
        predictions_output = args.output_dir / "predictions" / f"{args.run_name}_test.csv"
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_output, index=False)

    submission = make_submission(predictions, sample_submission)
    output = args.output
    if output is None:
        output = args.output_dir / "submissions" / f"{args.run_name}_submission.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Saved predictions: {predictions_output}")
    print(f"Saved submission: {output}")


if __name__ == "__main__":
    main()

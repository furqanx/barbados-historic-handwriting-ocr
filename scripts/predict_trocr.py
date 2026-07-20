"""Run TrOCR inference and create a submission file."""

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
from src.trocr.predictor import (  # noqa: E402
    get_prediction_config,
    load_trocr_checkpoint,
    make_submission,
    predict_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=TEST_MANIFEST)
    parser.add_argument("--sample-submission", type=Path, default=SAMPLE_SUBMISSION_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--preprocess-mode", choices=["default", "aspect"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--target-height", type=int, default=None)
    parser.add_argument("--canvas-width", type=int, default=None)
    parser.add_argument("--max-label-length", type=int, default=None)
    parser.add_argument("--max-generation-length", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else None
    model, processor, metadata = load_trocr_checkpoint(
        args.checkpoint_dir,
        device=device,
    )
    prediction_config = get_prediction_config(metadata)
    prediction_config = _override_prediction_config(prediction_config, args)

    test_manifest = pd.read_csv(args.test_manifest)
    sample_submission = pd.read_csv(args.sample_submission)
    predictions = predict_manifest(
        model,
        processor,
        test_manifest,
        device=device,
        **prediction_config,
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


def _override_prediction_config(config: dict, args: argparse.Namespace) -> dict:
    overrides = {
        "preprocess_mode": args.preprocess_mode,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "target_height": args.target_height,
        "canvas_width": args.canvas_width,
        "max_label_length": args.max_label_length,
        "max_generation_length": args.max_generation_length,
        "num_beams": args.num_beams,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


if __name__ == "__main__":
    main()

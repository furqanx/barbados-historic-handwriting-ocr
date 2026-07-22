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

from src.constants import FOLD_COL, OUTPUTS_DIR, SAMPLE_SUBMISSION_CSV, TEST_MANIFEST  # noqa: E402
from src.ctc.decoding import CTCDecoderConfig, CharNGramLanguageModel  # noqa: E402
from src.ctc.predictor import (  # noqa: E402
    load_crnn_ctc_checkpoint,
    make_submission,
    predict_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=TEST_MANIFEST)
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Optional fold filter when predicting from train_manifest for validation.",
    )
    parser.add_argument("--sample-submission", type=Path, default=SAMPLE_SUBMISSION_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--run-name", default="crnn_ctc")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument(
        "--decoder",
        default="greedy",
        choices=["greedy", "beam", "beam_lm", "beam_lm_rerank"],
    )
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument(
        "--top-tokens-per-step",
        type=int,
        default=20,
        help="Limit per-step token expansion. Use 0 to expand the full vocabulary.",
    )
    parser.add_argument("--lm-path", type=Path, default=None)
    parser.add_argument("--lm-weight", type=float, default=0.0)
    parser.add_argument("--length-bonus", type=float, default=0.0)
    parser.add_argument("--candidates-top-k", type=int, default=1)
    parser.add_argument("--rerank-short-text-penalty", type=float, default=0.0)
    parser.add_argument("--rerank-min-chars", type=int, default=0)
    parser.add_argument("--rerank-repeated-whitespace-penalty", type=float, default=0.0)
    parser.add_argument("--rerank-repeated-punctuation-penalty", type=float, default=0.0)
    parser.add_argument("--rerank-edge-space-penalty", type=float, default=0.0)
    parser.add_argument(
        "--no-submission",
        action="store_true",
        help="Only save predictions. Useful for validation manifests.",
    )
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
    if args.fold is not None:
        if FOLD_COL not in test_manifest.columns:
            raise ValueError(f"--fold was provided, but manifest has no {FOLD_COL!r} column.")
        test_manifest = test_manifest[test_manifest[FOLD_COL] == args.fold].reset_index(drop=True)
        if test_manifest.empty:
            raise ValueError(f"No rows found for fold {args.fold}.")
    decoder_config = CTCDecoderConfig(
        decoder=args.decoder,
        beam_size=args.beam_size,
        top_tokens_per_step=(
            None if args.top_tokens_per_step == 0 else args.top_tokens_per_step
        ),
        lm_weight=args.lm_weight,
        length_bonus=args.length_bonus,
        candidates_top_k=args.candidates_top_k,
        rerank_short_text_penalty=args.rerank_short_text_penalty,
        rerank_min_chars=args.rerank_min_chars,
        rerank_repeated_whitespace_penalty=args.rerank_repeated_whitespace_penalty,
        rerank_repeated_punctuation_penalty=args.rerank_repeated_punctuation_penalty,
        rerank_edge_space_penalty=args.rerank_edge_space_penalty,
    )
    language_model = CharNGramLanguageModel.load(args.lm_path) if args.lm_path else None
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
        decoder_config=decoder_config,
        language_model=language_model,
        device=device,
    )
    predictions_output = args.predictions_output
    if predictions_output is None:
        predictions_output = args.output_dir / "predictions" / f"{args.run_name}_test.csv"
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_output, index=False)

    if args.no_submission:
        print(f"Saved predictions: {predictions_output}")
        return

    sample_submission = pd.read_csv(args.sample_submission)
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

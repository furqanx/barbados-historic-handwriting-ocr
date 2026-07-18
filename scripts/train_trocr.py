"""Fine-tune TrOCR for handwritten line recognition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import OUTPUTS_DIR, TRAIN_MANIFEST  # noqa: E402
from src.training.trocr_trainer import TrOCRTrainingConfig, train_trocr  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--valid-predictions-output", type=Path, default=None)
    parser.add_argument("--no-save-valid-predictions", action="store_true")
    parser.add_argument(
        "--model-name",
        default="microsoft/trocr-small-handwritten",
        help="Hugging Face model name or local checkpoint directory.",
    )
    parser.add_argument(
        "--preprocess-mode",
        default="default",
        choices=["default", "aspect"],
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--target-height", type=int, default=384)
    parser.add_argument("--canvas-width", type=int, default=1536)
    parser.add_argument("--max-label-length", type=int, default=192)
    parser.add_argument("--max-generation-length", type=int, default=192)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--freeze-decoder", action="store_true")
    parser.add_argument("--freeze-encoder-layers", type=int, default=0)
    parser.add_argument("--freeze-decoder-layers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = args.output_dir / "checkpoints" / args.run_name

    valid_predictions_output = args.valid_predictions_output
    if valid_predictions_output is None and not args.no_save_valid_predictions:
        valid_predictions_output = (
            args.output_dir / "predictions" / f"{args.run_name}_valid_best.csv"
        )

    config = TrOCRTrainingConfig(
        model_name=args.model_name,
        preprocess_mode=args.preprocess_mode,
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        target_height=args.target_height,
        canvas_width=args.canvas_width,
        max_label_length=args.max_label_length,
        max_generation_length=args.max_generation_length,
        num_beams=args.num_beams,
        gradient_clip_norm=args.gradient_clip_norm,
        use_amp=not args.no_amp,
        use_data_parallel=not args.no_data_parallel,
        seed=args.seed,
        freeze_encoder=args.freeze_encoder,
        freeze_decoder=args.freeze_decoder,
        freeze_encoder_layers=args.freeze_encoder_layers,
        freeze_decoder_layers=args.freeze_decoder_layers,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
    )
    train_manifest = pd.read_csv(args.train_manifest)
    device = torch.device(args.device) if args.device else None
    _, best_score = train_trocr(
        train_manifest,
        config=config,
        checkpoint_dir=checkpoint_dir,
        valid_predictions_path=valid_predictions_output,
        device=device,
    )
    if best_score is not None:
        print(
            f"best_score={best_score.score:.5f} "
            f"best_wer={best_score.wer:.5f} "
            f"best_cer={best_score.cer:.5f}"
        )


if __name__ == "__main__":
    main()

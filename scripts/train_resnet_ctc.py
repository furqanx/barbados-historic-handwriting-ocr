"""Train the ResNet-style CNN + BiLSTM + CTC model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CHAR_VOCAB_JSON, OUTPUTS_DIR, TRAIN_MANIFEST  # noqa: E402
from src.data.char_tokenizer import CharacterTokenizer  # noqa: E402
from src.training.ctc_trainer import CTCTrainingConfig, train_resnet_ctc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--vocab", type=Path, default=CHAR_VOCAB_JSON)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--valid-predictions-output", type=Path, default=None)
    parser.add_argument("--no-save-valid-predictions", action="store_true")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--target-height", type=int, default=96)
    parser.add_argument("--max-width", type=int, default=2048)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--rnn-hidden-size", type=int, default=256)
    parser.add_argument("--rnn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"resnet_ctc_fold{args.fold}"
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = args.output_dir / "checkpoints" / f"{run_name}_best.pt"

    valid_predictions_output = args.valid_predictions_output
    if valid_predictions_output is None and not args.no_save_valid_predictions:
        valid_predictions_output = (
            args.output_dir / "predictions" / f"{run_name}_valid_best.csv"
        )

    train_manifest = pd.read_csv(args.train_manifest)
    tokenizer = CharacterTokenizer.load(args.vocab)
    config = CTCTrainingConfig(
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        target_height=args.target_height,
        max_width=args.max_width,
        gradient_clip_norm=args.gradient_clip_norm,
        use_amp=not args.no_amp,
        use_data_parallel=not args.no_data_parallel,
        seed=args.seed,
    )
    device = torch.device(args.device) if args.device else None
    _, best_score = train_resnet_ctc(
        train_manifest,
        tokenizer,
        config=config,
        checkpoint_path=checkpoint,
        valid_predictions_path=valid_predictions_output,
        device=device,
        base_channels=args.base_channels,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_layers=args.rnn_layers,
        dropout=args.dropout,
    )
    if best_score is not None:
        print(
            f"best_score={best_score.score:.5f} "
            f"best_wer={best_score.wer:.5f} "
            f"best_cer={best_score.cer:.5f}"
        )


if __name__ == "__main__":
    main()

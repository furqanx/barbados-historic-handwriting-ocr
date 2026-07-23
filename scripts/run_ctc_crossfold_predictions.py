"""Run CTC checkpoints on shared validation folds for ensemble diagnostics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import OUTPUTS_DIR, TRAIN_MANIFEST  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Checkpoint fold IDs to run.",
    )
    parser.add_argument(
        "--eval-folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Manifest fold IDs to predict.",
    )
    parser.add_argument(
        "--checkpoint-template",
        default="outputs/checkpoints/resnet_ctc_h96_w2048_fold{model_fold}_best.pt",
        help="Checkpoint path template. Available key: {model_fold}.",
    )
    parser.add_argument(
        "--run-template",
        default="resnet_ctc_h96_w2048_trainfold{model_fold}_evalfold{eval_fold}",
        help="Run name template. Available keys: {model_fold}, {eval_fold}.",
    )
    parser.add_argument("--manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--decoder", choices=["greedy", "beam", "beam_lm", "beam_lm_rerank"], default="greedy")
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--top-tokens-per-step", type=int, default=20)
    parser.add_argument("--candidates-top-k", type=int, default=1)
    parser.add_argument("--lm-path", type=Path, default=None)
    parser.add_argument("--lm-weight", type=float, default=0.0)
    parser.add_argument("--length-bonus", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for eval_fold in args.eval_folds:
        for model_fold in args.model_folds:
            checkpoint = _resolve_project_path(
                args.checkpoint_template.format(model_fold=model_fold)
            )
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint for model fold {model_fold}: {checkpoint}"
                )
            run_name = args.run_template.format(
                model_fold=model_fold,
                eval_fold=eval_fold,
            )
            command = [
                args.python,
                str(PROJECT_ROOT / "scripts" / "predict_ctc.py"),
                "--run-name",
                run_name,
                "--checkpoint",
                str(checkpoint),
                "--test-manifest",
                str(args.manifest),
                "--fold",
                str(eval_fold),
                "--decoder",
                args.decoder,
                "--beam-size",
                str(args.beam_size),
                "--top-tokens-per-step",
                str(args.top_tokens_per_step),
                "--candidates-top-k",
                str(args.candidates_top_k),
                "--length-bonus",
                str(args.length_bonus),
                "--predictions-output",
                str(args.output_dir / "predictions" / f"{run_name}.csv"),
                "--no-submission",
                "--device",
                args.device,
            ]
            if args.lm_path is not None:
                command.extend(["--lm-path", str(args.lm_path), "--lm-weight", str(args.lm_weight)])
            if args.batch_size is not None:
                command.extend(["--batch-size", str(args.batch_size)])
            if args.num_workers is not None:
                command.extend(["--num-workers", str(args.num_workers)])
            _run(command, dry_run=args.dry_run)


def _run(command: list[str], *, dry_run: bool) -> None:
    print("\n" + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


if __name__ == "__main__":
    main()


"""Run CTC decoding experiment grids for existing checkpoints.

This script does not train the visual CTC model. It reuses saved checkpoints and
reruns inference with beam search, optional character LM scoring, and optional
reranking. Use validation mode first to select a decoding configuration, then
run the best configuration on the test manifest.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import (  # noqa: E402
    CHAR_VOCAB_JSON,
    OUTPUTS_DIR,
    SAMPLE_SUBMISSION_CSV,
    TEST_MANIFEST,
    TRAIN_CSV,
    TRAIN_MANIFEST,
)


@dataclass(frozen=True)
class DecodingExperiment:
    """One CTC decoding configuration."""

    name: str
    decoder: str
    beam_size: int = 10
    lm_weight: float = 0.0
    rerank: bool = False


EXPERIMENTS: dict[str, DecodingExperiment] = {
    "beam10": DecodingExperiment(
        name="beam10",
        decoder="beam",
        beam_size=10,
    ),
    "beam25": DecodingExperiment(
        name="beam25",
        decoder="beam",
        beam_size=25,
    ),
    "beam25_lm002": DecodingExperiment(
        name="beam25_lm002",
        decoder="beam_lm",
        beam_size=25,
        lm_weight=0.02,
    ),
    "beam25_lm005": DecodingExperiment(
        name="beam25_lm005",
        decoder="beam_lm",
        beam_size=25,
        lm_weight=0.05,
    ),
    "beam25_lm010": DecodingExperiment(
        name="beam25_lm010",
        decoder="beam_lm",
        beam_size=25,
        lm_weight=0.10,
    ),
    "beam25_lm005_rerank": DecodingExperiment(
        name="beam25_lm005_rerank",
        decoder="beam_lm_rerank",
        beam_size=25,
        lm_weight=0.05,
        rerank=True,
    ),
}

DEFAULT_EXPERIMENT_NAMES = [
    "beam10",
    "beam25",
    "beam25_lm002",
    "beam25_lm005",
    "beam25_lm010",
    "beam25_lm005_rerank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0],
        help="Fold IDs to run. Use validation first before all-fold test inference.",
    )
    parser.add_argument(
        "--phase",
        choices=["valid", "test", "both"],
        default="valid",
        help="Run validation predictions, test predictions, or both.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        choices=sorted(EXPERIMENTS),
        help="Experiment key to run. Repeat to select multiple. Default: all.",
    )
    parser.add_argument(
        "--checkpoint-template",
        default="outputs/checkpoints/resnet_ctc_h96_w2048_fold{fold}_best.pt",
        help="Checkpoint path template. Available format key: {fold}.",
    )
    parser.add_argument(
        "--run-template",
        default="resnet_ctc_h96_w2048_fold{fold}",
        help="Base run-name template. Available format key: {fold}.",
    )
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--test-manifest", type=Path, default=TEST_MANIFEST)
    parser.add_argument("--sample-submission", type=Path, default=SAMPLE_SUBMISSION_CSV)
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--char-vocab", type=Path, default=CHAR_VOCAB_JSON)
    parser.add_argument(
        "--lm-output",
        type=Path,
        default=OUTPUTS_DIR / "language_models" / "char_ngram_order4.json",
    )
    parser.add_argument("--lm-order", type=int, default=4)
    parser.add_argument("--lm-add-k", type=float, default=0.5)
    parser.add_argument(
        "--force-lm",
        action="store_true",
        help="Retrain the character LM even when --lm-output already exists.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--top-tokens-per-step", type=int, default=20)
    parser.add_argument("--candidates-top-k", type=int, default=5)
    parser.add_argument("--length-bonus", type=float, default=0.0)
    parser.add_argument("--rerank-short-text-penalty", type=float, default=0.0)
    parser.add_argument("--rerank-min-chars", type=int, default=0)
    parser.add_argument("--rerank-repeated-whitespace-penalty", type=float, default=0.2)
    parser.add_argument("--rerank-repeated-punctuation-penalty", type=float, default=0.2)
    parser.add_argument("--rerank-edge-space-penalty", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to call project scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [EXPERIMENTS[name] for name in (args.experiment or DEFAULT_EXPERIMENT_NAMES)]

    if any(experiment.decoder.startswith("beam_lm") for experiment in selected):
        _ensure_language_model(args)

    phases = ["valid", "test"] if args.phase == "both" else [args.phase]
    for fold in args.folds:
        checkpoint = _resolve_project_path(args.checkpoint_template.format(fold=fold))
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {checkpoint}")
        base_run_name = args.run_template.format(fold=fold)
        for experiment in selected:
            run_name = f"{base_run_name}_{experiment.name}"
            for phase in phases:
                command = _predict_command(args, experiment, fold, checkpoint, run_name, phase)
                _run(command, dry_run=args.dry_run)
                if phase == "valid":
                    _run(_diagnose_command(args, run_name), dry_run=args.dry_run)


def _ensure_language_model(args: argparse.Namespace) -> None:
    if args.lm_output.exists() and not args.force_lm:
        print(f"Using existing character LM: {args.lm_output}")
        return
    command = [
        args.python,
        str(PROJECT_ROOT / "scripts" / "train_char_lm.py"),
        "--train-csv",
        str(args.train_csv),
        "--char-vocab",
        str(args.char_vocab),
        "--order",
        str(args.lm_order),
        "--add-k",
        str(args.lm_add_k),
        "--output",
        str(args.lm_output),
    ]
    _run(command, dry_run=args.dry_run)


def _predict_command(
    args: argparse.Namespace,
    experiment: DecodingExperiment,
    fold: int,
    checkpoint: Path,
    run_name: str,
    phase: str,
) -> list[str]:
    command = [
        args.python,
        str(PROJECT_ROOT / "scripts" / "predict_ctc.py"),
        "--run-name",
        run_name,
        "--checkpoint",
        str(checkpoint),
        "--decoder",
        experiment.decoder,
        "--beam-size",
        str(experiment.beam_size),
        "--top-tokens-per-step",
        str(args.top_tokens_per_step),
        "--candidates-top-k",
        str(args.candidates_top_k),
        "--length-bonus",
        str(args.length_bonus),
        "--device",
        args.device,
    ]
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.num_workers is not None:
        command.extend(["--num-workers", str(args.num_workers)])

    if experiment.decoder.startswith("beam_lm"):
        command.extend(
            [
                "--lm-path",
                str(args.lm_output),
                "--lm-weight",
                str(experiment.lm_weight),
            ]
        )
    if experiment.rerank:
        command.extend(
            [
                "--rerank-short-text-penalty",
                str(args.rerank_short_text_penalty),
                "--rerank-min-chars",
                str(args.rerank_min_chars),
                "--rerank-repeated-whitespace-penalty",
                str(args.rerank_repeated_whitespace_penalty),
                "--rerank-repeated-punctuation-penalty",
                str(args.rerank_repeated_punctuation_penalty),
                "--rerank-edge-space-penalty",
                str(args.rerank_edge_space_penalty),
            ]
        )

    if phase == "valid":
        command.extend(
            [
                "--test-manifest",
                str(args.train_manifest),
                "--fold",
                str(fold),
                "--predictions-output",
                str(args.output_dir / "predictions" / f"{run_name}_valid.csv"),
                "--no-submission",
            ]
        )
    elif phase == "test":
        command.extend(
            [
                "--test-manifest",
                str(args.test_manifest),
                "--sample-submission",
                str(args.sample_submission),
                "--predictions-output",
                str(args.output_dir / "predictions" / f"{run_name}_test.csv"),
                "--output",
                str(args.output_dir / "submissions" / f"{run_name}_submission.csv"),
            ]
        )
    else:
        raise ValueError(f"Unsupported phase: {phase}")
    return command


def _diagnose_command(args: argparse.Namespace, run_name: str) -> list[str]:
    return [
        args.python,
        str(PROJECT_ROOT / "scripts" / "diagnose_evaluator.py"),
        "--predictions",
        str(args.output_dir / "predictions" / f"{run_name}_valid.csv"),
        "--output-dir",
        str(args.output_dir.parent / "reports" / "diagnostics" / "evaluation" / run_name),
    ]


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

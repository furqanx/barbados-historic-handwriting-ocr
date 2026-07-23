"""Fine-tune a downloaded PyLaia HTR model."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import PYLAIA_OUTPUTS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--augment-training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw PyLaia CLI argument. Repeat for each token.",
    )
    return parser.parse_args()


def build_train_command(args: argparse.Namespace) -> list[str]:
    experiment_dir = args.experiment_dir or PYLAIA_OUTPUTS_DIR / args.run_name
    model_file = args.base_model_dir / "model"
    checkpoint = args.base_model_dir / "weights.ckpt"
    syms = args.dataset_dir / "syms.txt"
    images = args.dataset_dir / "images"
    train_txt = args.dataset_dir / "train.txt"
    val_txt = args.dataset_dir / "val.txt"
    _require_files([model_file, checkpoint, syms, train_txt, val_txt])

    command = [
        "pylaia-htr-train-ctc",
        str(syms),
        _as_pylaia_list_arg(images),
        str(train_txt),
        str(val_txt),
        "--common.experiment_dirname",
        str(experiment_dir),
        "--common.model_filename",
        str(model_file),
        "--common.checkpoint",
        str(checkpoint),
        "--train.pretrain",
        "true",
        "--train.early_stopping_patience",
        str(args.early_stopping_patience),
        "--train.augment_training",
        str(args.augment_training).lower(),
        "--data.batch_size",
        str(args.batch_size),
        "--data.num_workers",
        str(args.num_workers),
        "--optimizers.learning_rate",
        str(args.lr),
        "--trainer.max_epochs",
        str(args.epochs),
        "--trainer.gpus",
        str(args.gpus),
    ]
    command.extend(args.extra_arg)
    return command


def _as_pylaia_list_arg(path: Path) -> str:
    return f'["{path}"]'


def main() -> None:
    args = parse_args()
    command = build_train_command(args)
    print(" ".join(shlex.quote(part) for part in command))
    if not args.dry_run:
        subprocess.run(command, check=True)


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


if __name__ == "__main__":
    main()

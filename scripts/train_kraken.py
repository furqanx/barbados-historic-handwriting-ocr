"""Fine-tune a Kraken recognition model with `ketos train`."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import KRAKEN_OUTPUTS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--lag", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--resize", choices=["union", "new", "fail"], default="new")
    parser.add_argument(
        "--unicode-normalization",
        choices=["NFC", "NFKC", "NFD", "NFKD"],
        default="NFD",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--quit", choices=["early", "fixed"], default="early")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw ketos CLI argument. Repeat for each token.",
    )
    return parser.parse_args()


def build_train_command(args: argparse.Namespace) -> list[str]:
    train_files = args.dataset_dir / "train_files.txt"
    val_files = args.dataset_dir / "val_files.txt"
    _require_files([args.base_model, train_files, val_files])

    output_prefix = (
        args.output_prefix
        or KRAKEN_OUTPUTS_DIR / args.run_name / args.run_name
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ketos",
        "--workers",
        str(args.workers),
        "-d",
        args.device,
        "--precision",
        args.precision,
        "train",
        "-f",
        "path",
        "-i",
        str(args.base_model),
        "--resize",
        args.resize,
        "-o",
        str(output_prefix),
        "-N",
        str(args.epochs),
        "--min-epochs",
        str(args.min_epochs),
        "--lag",
        str(args.lag),
        "-B",
        str(args.batch_size),
        "-r",
        str(args.lr),
        "-w",
        str(args.weight_decay),
        "--optimizer",
        args.optimizer,
        "-u",
        args.unicode_normalization,
        "-q",
        args.quit,
        "-t",
        str(train_files),
        "-e",
        str(val_files),
    ]
    command.append("--augment" if args.augment else "--no-augment")
    command.extend(args.extra_arg)
    return command


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


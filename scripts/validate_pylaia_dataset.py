"""Validate a prepared PyLaia dataset with the PyLaia CLI."""

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
    parser.add_argument("--run-name", default="pylaia_dataset_validation")
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--statistics-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw PyLaia CLI argument. Repeat for each token.",
    )
    return parser.parse_args()


def build_validate_command(args: argparse.Namespace) -> list[str]:
    experiment_dir = args.experiment_dir or PYLAIA_OUTPUTS_DIR / args.run_name
    statistics_output = (
        args.statistics_output
        or PYLAIA_OUTPUTS_DIR / args.run_name / "dataset_statistics.md"
    )
    syms = args.dataset_dir / "syms.txt"
    images = args.dataset_dir / "images"
    train_txt = args.dataset_dir / "train.txt"
    val_txt = args.dataset_dir / "val.txt"
    test_txt = args.dataset_dir / "test.txt"
    _require_files([syms, train_txt, val_txt, test_txt])

    command = [
        "pylaia-htr-dataset-validate",
        str(syms),
        str(images),
        str(train_txt),
        str(val_txt),
        str(test_txt),
        "--fixed_input_height",
        str(args.image_height),
        "--statistics_output",
        str(statistics_output),
        "--common.experiment_dirname",
        str(experiment_dir),
    ]
    command.extend(args.extra_arg)
    statistics_output.parent.mkdir(parents=True, exist_ok=True)
    return command


def main() -> None:
    args = parse_args()
    command = build_validate_command(args)
    print(" ".join(shlex.quote(part) for part in command))
    if not args.dry_run:
        subprocess.run(command, check=True)


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


if __name__ == "__main__":
    main()


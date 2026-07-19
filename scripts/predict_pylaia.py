"""Run PyLaia decoding and save raw predictions."""

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
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-dir-arg", default="--img_dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw PyLaia CLI argument. Repeat for each token.",
    )
    return parser.parse_args()


def build_predict_command(args: argparse.Namespace) -> list[str]:
    experiment_dir = args.experiment_dir or PYLAIA_OUTPUTS_DIR / args.run_name
    output = args.output or PYLAIA_OUTPUTS_DIR / args.run_name / f"{args.split}_raw.txt"
    model_file = args.base_model_dir / "model"
    syms = args.dataset_dir / "syms.txt"
    images = args.dataset_dir / "images"
    img_list = args.dataset_dir / f"{args.split}_ids.txt"
    _require_files([model_file, syms, img_list])

    command = [
        "pylaia-htr-decode-ctc",
        "--common.experiment_dirname",
        str(experiment_dir),
        "--common.model_filename",
        str(model_file),
        args.img_dir_arg,
        str(images),
        "--data.batch_size",
        str(args.batch_size),
        "--data.num_workers",
        str(args.num_workers),
    ]
    if args.checkpoint is not None:
        command.extend(["--common.checkpoint", str(args.checkpoint)])
    command.extend(args.extra_arg)
    command.extend([str(syms), str(img_list)])
    output.parent.mkdir(parents=True, exist_ok=True)
    return command


def main() -> None:
    args = parse_args()
    output = args.output or PYLAIA_OUTPUTS_DIR / args.run_name / f"{args.split}_raw.txt"
    command = build_predict_command(args)
    print(" ".join(shlex.quote(part) for part in command))
    print(f"Raw output: {output}")
    if args.dry_run:
        return
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.stdout, encoding="utf-8")
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


if __name__ == "__main__":
    main()


"""Train a PaddleOCR text recognition model via PaddleOCR's `tools/train.py`."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paddleocr-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional PaddleOCR -o override, e.g. Global.epoch_num=10.",
    )
    return parser.parse_args()


def build_train_command(args: argparse.Namespace) -> list[str]:
    train_py = args.paddleocr_dir / "tools" / "train.py"
    _require_files([train_py, args.config])
    if args.gpus:
        command = [
            sys.executable,
            "-m",
            "paddle.distributed.launch",
            "--gpus",
            args.gpus,
            str(train_py),
            "-c",
            str(args.config),
        ]
    else:
        command = [sys.executable, str(train_py), "-c", str(args.config)]
    if args.override:
        command.append("-o")
        command.extend(args.override)
    return command


def main() -> None:
    args = parse_args()
    command = build_train_command(args)
    print(" ".join(shlex.quote(part) for part in command))
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=args.paddleocr_dir)


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


if __name__ == "__main__":
    main()


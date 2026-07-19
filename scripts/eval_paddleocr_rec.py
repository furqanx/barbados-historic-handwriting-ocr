"""Evaluate a PaddleOCR recognition checkpoint via PaddleOCR's `tools/eval.py`."""

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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional PaddleOCR -o override.",
    )
    return parser.parse_args()


def build_eval_command(args: argparse.Namespace) -> list[str]:
    eval_py = args.paddleocr_dir / "tools" / "eval.py"
    _require_files([eval_py, args.config])
    command = [
        sys.executable,
        str(eval_py),
        "-c",
        str(args.config),
        "-o",
        f"Global.checkpoints={args.checkpoint}",
    ]
    command.extend(args.override)
    return command


def main() -> None:
    args = parse_args()
    command = build_eval_command(args)
    print(" ".join(shlex.quote(part) for part in command))
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=args.paddleocr_dir)


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


if __name__ == "__main__":
    main()


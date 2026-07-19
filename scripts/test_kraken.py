"""Evaluate a Kraken model on the validation split with `ketos test`."""

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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--unicode-normalization",
        choices=["NFC", "NFKC", "NFD", "NFKD"],
        default="NFD",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw ketos CLI argument. Repeat for each token.",
    )
    return parser.parse_args()


def build_test_command(args: argparse.Namespace) -> list[str]:
    files = args.dataset_dir / f"{args.split}_files.txt"
    _require_files([args.model, files])
    command = [
        "ketos",
        "--workers",
        str(args.workers),
        "-d",
        args.device,
        "--precision",
        args.precision,
        "test",
        "-f",
        "path",
        "-m",
        str(args.model),
        "-u",
        args.unicode_normalization,
        "-e",
        str(files),
    ]
    command.extend(args.extra_arg)
    return command


def main() -> None:
    args = parse_args()
    output = args.output or KRAKEN_OUTPUTS_DIR / args.run_name / f"{args.split}_ketos_test.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_test_command(args)
    print(" ".join(shlex.quote(part) for part in command))
    print(f"Report output: {output}")
    if args.dry_run:
        return
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    output.write_text(result.stdout, encoding="utf-8")
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


if __name__ == "__main__":
    main()


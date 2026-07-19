"""Run Kraken OCR on prepared line images and save one text file per image."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import KRAKEN_OUTPUTS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--suffix", default=".txt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw Kraken OCR argument. Repeat for each token.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = args.dataset_dir / args.split
    files = args.dataset_dir / f"{args.split}_files.txt"
    _require_files([args.model, files])
    output_dir = args.output_dir or KRAKEN_OUTPUTS_DIR / args.run_name / f"{args.split}_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        Path(line.strip())
        for line in files.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for image_path in tqdm(image_paths, desc=f"kraken {args.split}", leave=False):
        output = output_dir / f"{image_path.stem}{args.suffix}"
        command = [
            "kraken",
            "-i",
            str(image_path),
            str(output),
            "segment",
            "-bl",
            "ocr",
            "-m",
            str(args.model),
        ]
        command.extend(args.extra_arg)
        if args.dry_run:
            print(" ".join(command))
            continue
        subprocess.run(command, check=True)
    print(f"Saved raw Kraken predictions: {output_dir}")
    print(f"Input split dir: {split_dir}")


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


if __name__ == "__main__":
    main()


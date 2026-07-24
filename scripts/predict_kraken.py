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

from src.constants import KRAKEN_MODELS_DIR, KRAKEN_OUTPUTS_DIR  # noqa: E402
from src.htr.kraken.model_registry import resolve_kraken_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument(
        "--model-key",
        default=None,
        help="Known Kraken model alias. Alternative to --model.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--suffix", default=".txt")
    parser.add_argument(
        "--recognition-mode",
        choices=["line", "segment"],
        default="line",
        help=(
            "`line` treats each prepared image as a single text line using "
            "`ocr --no-segmentation`. `segment` runs Kraken's page segmenter "
            "before recognition and is much slower for pre-cropped line images."
        ),
    )
    parser.add_argument(
        "--kraken-process-images",
        type=int,
        default=128,
        help=(
            "Number of images per Kraken process in line mode. Larger values load "
            "the model fewer times but create longer command lines."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument(
        "--ocr-batch-size",
        type=int,
        default=64,
        help="Kraken OCR -B/--batch-size value in line mode.",
    )
    parser.add_argument(
        "--num-line-workers",
        type=int,
        default=4,
        help="Kraken OCR --num-line-workers value in line mode.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Only process the first N images after --start-index. Useful for memory-safe screening.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing at this 0-based image index.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip images whose output text file already exists.",
    )
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
    model = _resolve_model_path(args.model, args.model_key)
    _require_files([model, files])
    output_dir = args.output_dir or KRAKEN_OUTPUTS_DIR / args.run_name / f"{args.split}_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        Path(line.strip())
        for line in files.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    image_paths = image_paths[args.start_index :]
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max-images must be positive when provided.")
        image_paths = image_paths[: args.max_images]

    if args.recognition_mode == "line":
        _run_line_ocr(args, image_paths, output_dir, model)
    else:
        _run_segment_ocr(args, image_paths, output_dir, model)
    print(f"Saved raw Kraken predictions: {output_dir}")
    print(f"Input split dir: {split_dir}")


def _run_line_ocr(
    args: argparse.Namespace,
    image_paths: list[Path],
    output_dir: Path,
    model: Path,
) -> None:
    if args.kraken_process_images <= 0:
        raise ValueError("--kraken-process-images must be positive.")

    pending_pairs: list[tuple[Path, Path]] = []
    for image_path in image_paths:
        output = output_dir / f"{image_path.stem}{args.suffix}"
        if args.skip_existing and output.exists():
            continue
        pending_pairs.append((image_path, output))

    chunks = [
        pending_pairs[index : index + args.kraken_process_images]
        for index in range(0, len(pending_pairs), args.kraken_process_images)
    ]

    for chunk in tqdm(chunks, desc=f"kraken {args.split} line chunks", leave=False):
        command = ["kraken"]
        for image_path, output in chunk:
            command.extend(["-i", str(image_path), str(output)])
        _extend_global_runtime_args(command, args)
        command.extend(
            [
                "ocr",
                "--no-segmentation",
                "-m",
                str(model),
                "-B",
                str(args.ocr_batch_size),
                "--num-line-workers",
                str(args.num_line_workers),
            ]
        )
        command.extend(args.extra_arg)
        if args.dry_run:
            print(" ".join(command))
            continue
        subprocess.run(command, check=True)


def _run_segment_ocr(
    args: argparse.Namespace,
    image_paths: list[Path],
    output_dir: Path,
    model: Path,
) -> None:
    for image_path in tqdm(image_paths, desc=f"kraken {args.split} segment", leave=False):
        output = output_dir / f"{image_path.stem}{args.suffix}"
        if args.skip_existing and output.exists():
            continue
        command = [
            "kraken",
            "-i",
            str(image_path),
            str(output),
        ]
        _extend_global_runtime_args(command, args)
        command.extend(["segment", "-bl", "ocr", "-m", str(model)])
        command.extend(args.extra_arg)
        if args.dry_run:
            print(" ".join(command))
            continue
        subprocess.run(command, check=True)


def _extend_global_runtime_args(command: list[str], args: argparse.Namespace) -> None:
    if args.device:
        command.extend(["--device", args.device])
    if args.precision:
        command.extend(["--precision", args.precision])


def _resolve_model_path(model: Path | None, model_key: str | None) -> Path:
    if model is not None:
        return model
    if model_key is None:
        raise ValueError("Provide either --model or --model-key.")
    model_info = resolve_kraken_model(model_key)
    return KRAKEN_MODELS_DIR / model_key / model_info.filename


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


if __name__ == "__main__":
    main()

"""Run PaddleOCR recognition inference via `tools/infer_rec.py`."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import PADDLEOCR_OUTPUTS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paddleocr-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--infer-img", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional PaddleOCR -o override.",
    )
    return parser.parse_args()


def build_predict_command(args: argparse.Namespace, output: Path) -> list[str]:
    infer_py = args.paddleocr_dir / "tools" / "infer_rec.py"
    _require_files([infer_py, args.config])
    command = [
        sys.executable,
        str(infer_py),
        "-c",
        str(args.config),
        "-o",
        f"Global.pretrained_model={args.checkpoint}",
        f"Global.infer_img={args.infer_img}",
        f"Global.save_res_path={output}",
    ]
    command.extend(args.override)
    return command


def main() -> None:
    args = parse_args()
    output = args.output or PADDLEOCR_OUTPUTS_DIR / args.run_name / "test_raw.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_predict_command(args, output)
    print(" ".join(shlex.quote(part) for part in command))
    print(f"Raw prediction output: {output}")
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=args.paddleocr_dir)


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


if __name__ == "__main__":
    main()

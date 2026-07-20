"""Download a known Kraken HTR `.mlmodel` file."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import KRAKEN_MODELS_DIR  # noqa: E402
from src.htr.kraken.model_registry import resolve_kraken_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-key",
        choices=["catmus-medieval", "mccatmus", "tridis"],
        default="catmus-medieval",
    )
    parser.add_argument("--url", default=None)
    parser.add_argument("--filename", default=None)
    parser.add_argument("--output-dir", type=Path, default=KRAKEN_MODELS_DIR)
    parser.add_argument("--local-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_info = resolve_kraken_model(args.model_key)
    url = args.url or model_info.download_url
    filename = args.filename or model_info.filename
    local_dir = args.local_dir or args.output_dir / args.model_key
    local_dir.mkdir(parents=True, exist_ok=True)
    output = local_dir / filename
    if output.exists() and not args.force:
        print(f"Model already exists: {output}")
        return

    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, output.open("wb") as file_obj:
        shutil.copyfileobj(response, file_obj)
    print(f"Saved model: {output}")
    print(f"DOI/source: {model_info.doi}")


if __name__ == "__main__":
    main()


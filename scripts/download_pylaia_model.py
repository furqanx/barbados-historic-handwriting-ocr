"""Download a Teklia PyLaia model from Hugging Face."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import PYLAIA_MODELS_DIR  # noqa: E402
from src.pylaia.model_registry import (  # noqa: E402
    PYLAIA_REQUIRED_MODEL_FILES,
    resolve_model_repo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default="himanis")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=PYLAIA_MODELS_DIR)
    parser.add_argument("--local-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_id = resolve_model_repo(args.model_key, args.repo_id)
    local_dir = args.local_dir or args.output_dir / repo_id.split("/")[-1]
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Install Hugging Face Hub first: `pip install huggingface_hub`."
        ) from exc

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=[
            "model",
            "weights.ckpt",
            "syms.txt",
            "tokens.txt",
            "language_model.arpa.gz",
            "lexicon.txt",
            "README.md",
        ],
    )

    missing = [
        filename for filename in PYLAIA_REQUIRED_MODEL_FILES
        if not (local_dir / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Downloaded model is missing required files: {missing}"
        )
    print(f"Downloaded {repo_id} to {local_dir}")


if __name__ == "__main__":
    main()


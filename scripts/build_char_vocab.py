"""Build and save the character vocabulary used by CTC models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CHAR_VOCAB_JSON, TARGET_COL, TRAIN_CSV  # noqa: E402
from src.data.char_tokenizer import build_tokenizer_from_train_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--output", type=Path, default=CHAR_VOCAB_JSON)
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Build vocab from raw target text without whitespace normalization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = build_tokenizer_from_train_csv(
        args.train_csv,
        target_col=args.target_col,
        normalize=not args.no_normalize,
    )
    output = tokenizer.save(args.output)

    print(f"Saved character vocabulary: {output}")
    print(f"Vocab size including CTC blank: {tokenizer.vocab_size}")
    print(f"CTC blank token/id: {tokenizer.blank_token!r}/{tokenizer.blank_id}")
    print(f"Characters excluding blank: {len(tokenizer.characters)}")


if __name__ == "__main__":
    main()

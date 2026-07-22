"""Train a lightweight character n-gram LM for CTC beam decoding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import OUTPUTS_DIR, TARGET_COL, TRAIN_CSV  # noqa: E402
from src.ctc.decoding import CharNGramLanguageModel  # noqa: E402
from src.ctc.tokenizer import CharacterTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--char-vocab", type=Path, default=None)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--add-k", type=float, default=0.5)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "language_models" / "char_ngram_order4.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    if args.target_col not in train.columns:
        raise ValueError(f"Target column not found: {args.target_col}")

    extra_characters = None
    if args.char_vocab is not None:
        tokenizer = CharacterTokenizer.load(args.char_vocab)
        extra_characters = tokenizer.characters

    language_model = CharNGramLanguageModel.train(
        train[args.target_col].tolist(),
        order=args.order,
        add_k=args.add_k,
        extra_characters=extra_characters,
        normalize=not args.no_normalize,
    )
    output = language_model.save(args.output)
    print(f"Saved character LM: {output}")
    print(f"order={language_model.order} vocab_size={len(language_model.vocab)}")
    print(f"contexts={len(language_model.context_counts)} add_k={language_model.add_k}")


if __name__ == "__main__":
    main()

"""Audit target characters and CTC tokenizer round-trip integrity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CHAR_VOCAB_JSON, TARGET_COL, TRAIN_CSV  # noqa: E402
from src.ctc.tokenizer import CharacterTokenizer  # noqa: E402
from src.diagnostics.charset_audit import run_charset_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--vocab", type=Path, default=CHAR_VOCAB_JSON)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "diagnostics" / "charset")
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--min-char-count", type=int, default=10)
    parser.add_argument("--skip-tokenizer-roundtrip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    tokenizer = None
    if not args.skip_tokenizer_roundtrip:
        tokenizer = CharacterTokenizer.load(args.vocab)

    result = run_charset_audit(
        train,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        target_col=args.target_col,
        min_char_count=args.min_char_count,
    )
    print(f"Saved character frequency: {result.character_frequency}")
    if result.tokenizer_roundtrip is not None:
        roundtrip = pd.read_csv(result.tokenizer_roundtrip)
        failures = int((~roundtrip["roundtrip_ok"]).sum())
        print(f"Saved tokenizer round-trip: {result.tokenizer_roundtrip}")
        print(f"roundtrip_failures={failures}")
    print(f"Saved rare-character lines: {result.suspicious_lines}")


if __name__ == "__main__":
    main()


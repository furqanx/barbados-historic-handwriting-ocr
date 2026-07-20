"""Audit whether CTC models have enough encoder time steps per target."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import TARGET_COL, TRAIN_MANIFEST  # noqa: E402
from src.diagnostics.ctc_alignment import (  # noqa: E402
    CTCAlignmentConfig,
    ctc_alignment_table,
    summarize_ctc_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "diagnostics" / "ctc_alignment")
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--target-height", type=int, default=96)
    parser.add_argument("--max-width", type=int, default=2048)
    parser.add_argument("--no-max-width", action="store_true")
    parser.add_argument("--time-downsample-factor", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.train_manifest)
    max_width = None if args.no_max_width else args.max_width
    config = CTCAlignmentConfig(
        target_height=args.target_height,
        max_width=max_width,
        time_downsample_factor=args.time_downsample_factor,
    )
    table = ctc_alignment_table(manifest, config=config, target_col=args.target_col)
    summary = summarize_ctc_alignment(table)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.sort_values("ctc_margin").to_csv(args.output_dir / "ctc_alignment_audit.csv", index=False)
    (args.output_dir / "ctc_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved: {args.output_dir / 'ctc_alignment_audit.csv'}")


if __name__ == "__main__":
    main()


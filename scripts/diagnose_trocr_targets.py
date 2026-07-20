"""Audit TrOCR tokenizer target lengths and aspect-aware canvas clipping."""

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
from src.diagnostics.trocr_audit import (  # noqa: E402
    TrOCRTargetAuditConfig,
    load_trocr_tokenizer,
    summarize_trocr_audit,
    trocr_target_audit_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "diagnostics" / "trocr")
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--model-name", default="microsoft/trocr-small-handwritten")
    parser.add_argument("--preprocess-mode", default="default", choices=["default", "aspect"])
    parser.add_argument("--target-height", type=int, default=384)
    parser.add_argument("--canvas-width", type=int, default=1536)
    parser.add_argument("--max-label-length", type=int, default=192)
    parser.add_argument("--max-generation-length", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.train_manifest)
    config = TrOCRTargetAuditConfig(
        model_name=args.model_name,
        preprocess_mode=args.preprocess_mode,
        target_height=args.target_height,
        canvas_width=args.canvas_width,
        max_label_length=args.max_label_length,
        max_generation_length=args.max_generation_length,
    )
    tokenizer = load_trocr_tokenizer(args.model_name)
    table = trocr_target_audit_table(
        manifest,
        tokenizer=tokenizer,
        config=config,
        target_col=args.target_col,
    )
    summary = summarize_trocr_audit(table)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.sort_values("token_count", ascending=False).to_csv(
        args.output_dir / "trocr_target_audit.csv",
        index=False,
    )
    (args.output_dir / "trocr_target_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved: {args.output_dir / 'trocr_target_audit.csv'}")


if __name__ == "__main__":
    main()


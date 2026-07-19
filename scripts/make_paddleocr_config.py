"""Patch a PaddleOCR recognition YAML config for this competition dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import PADDLEOCR_OUTPUTS_DIR  # noqa: E402
from src.paddleocr_rec.config import (  # noqa: E402
    load_yaml,
    patch_recognition_config,
    save_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save-model-dir", type=Path, default=None)
    parser.add_argument("--pretrained-model", type=Path, default=None)
    parser.add_argument("--max-text-length", type=int, default=128)
    parser.add_argument("--image-shape", default="3,64,2048")
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--epoch-num", type=int, default=None)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = (
        args.output
        or PADDLEOCR_OUTPUTS_DIR / args.run_name / f"{args.run_name}.yml"
    )
    save_model_dir = (
        args.save_model_dir
        or PADDLEOCR_OUTPUTS_DIR / args.run_name / "checkpoints"
    )
    config = load_yaml(args.base_config)
    patched = patch_recognition_config(
        config,
        train_labels=args.dataset_dir / "rec_gt_train.txt",
        val_labels=args.dataset_dir / "rec_gt_val.txt",
        character_dict=args.dataset_dir / "character_dict.txt",
        save_model_dir=save_model_dir,
        pretrained_model=args.pretrained_model,
        max_text_length=args.max_text_length,
        image_shape=args.image_shape,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        epoch_num=args.epoch_num,
        use_amp=args.use_amp,
    )
    saved = save_yaml(patched, output)
    print(f"Saved PaddleOCR config: {saved}")
    print(f"Model output dir: {save_model_dir}")


if __name__ == "__main__":
    main()


"""Prepare competition manifests as Kraken legacy line-strip datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from src.constants import FOLD_COL, ID_COL, TARGET_COL
from src.kraken_htr.text import normalize_for_kraken


@dataclass(frozen=True)
class KrakenDatasetPaths:
    """Output paths for a prepared Kraken dataset."""

    root: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path
    train_files: Path
    val_files: Path
    test_files: Path
    metadata: Path


def prepare_kraken_dataset(
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    *,
    output_dir: str | Path,
    fold: int,
    image_height: int = 128,
    image_mode: str = "L",
    image_format: str = "png",
    unicode_normalization: str = "preserve",
    force: bool = False,
) -> KrakenDatasetPaths:
    """Create resized images, `.gt.txt` files, and split manifests."""

    output = Path(output_dir)
    paths = KrakenDatasetPaths(
        root=output,
        train_dir=output / "train",
        val_dir=output / "val",
        test_dir=output / "test",
        train_files=output / "train_files.txt",
        val_files=output / "val_files.txt",
        test_files=output / "test_files.txt",
        metadata=output / "metadata.json",
    )
    for directory in (paths.train_dir, paths.val_dir, paths.test_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_df = train_manifest[train_manifest[FOLD_COL] != fold].reset_index(drop=True)
    val_df = train_manifest[train_manifest[FOLD_COL] == fold].reset_index(drop=True)
    test_df = test_manifest.reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError(f"Fold {fold} produced empty train or validation data.")

    train_images = _write_labeled_split(
        train_df,
        split_dir=paths.train_dir,
        image_height=image_height,
        image_mode=image_mode,
        image_format=image_format,
        unicode_normalization=unicode_normalization,
        force=force,
        split="train",
    )
    val_images = _write_labeled_split(
        val_df,
        split_dir=paths.val_dir,
        image_height=image_height,
        image_mode=image_mode,
        image_format=image_format,
        unicode_normalization=unicode_normalization,
        force=force,
        split="val",
    )
    test_images = _write_unlabeled_split(
        test_df,
        split_dir=paths.test_dir,
        image_height=image_height,
        image_mode=image_mode,
        image_format=image_format,
        force=force,
        split="test",
    )
    _write_lines(paths.train_files, [str(path) for path in train_images])
    _write_lines(paths.val_files, [str(path) for path in val_images])
    _write_lines(paths.test_files, [str(path) for path in test_images])

    metadata = {
        "fold": fold,
        "image_height": image_height,
        "image_mode": image_mode,
        "image_format": image_format,
        "unicode_normalization": unicode_normalization,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "paths": {key: str(value) for key, value in asdict(paths).items()},
    }
    paths.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths


def _write_labeled_split(
    df: pd.DataFrame,
    *,
    split_dir: Path,
    image_height: int,
    image_mode: str,
    image_format: str,
    unicode_normalization: str,
    force: bool,
    split: str,
) -> list[Path]:
    extension = _image_extension(image_format)
    image_paths: list[Path] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"prepare {split}", leave=False):
        image_id = str(getattr(row, ID_COL))
        image_path = split_dir / f"{image_id}{extension}"
        source = Path(str(getattr(row, "image_path")))
        if force or not image_path.exists():
            resize_line_image(
                source,
                image_path,
                image_height=image_height,
                image_mode=image_mode,
            )
        gt_path = image_path.with_suffix(".gt.txt")
        text = normalize_for_kraken(
            getattr(row, TARGET_COL),
            unicode_form=unicode_normalization,
        )
        gt_path.write_text(text + "\n", encoding="utf-8")
        image_paths.append(image_path)
    return image_paths


def _write_unlabeled_split(
    df: pd.DataFrame,
    *,
    split_dir: Path,
    image_height: int,
    image_mode: str,
    image_format: str,
    force: bool,
    split: str,
) -> list[Path]:
    extension = _image_extension(image_format)
    image_paths: list[Path] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"prepare {split}", leave=False):
        image_id = str(getattr(row, ID_COL))
        image_path = split_dir / f"{image_id}{extension}"
        source = Path(str(getattr(row, "image_path")))
        if force or not image_path.exists():
            resize_line_image(
                source,
                image_path,
                image_height=image_height,
                image_mode=image_mode,
            )
        image_paths.append(image_path)
    return image_paths


def resize_line_image(
    source: str | Path,
    target: str | Path,
    *,
    image_height: int,
    image_mode: str = "L",
) -> Path:
    """Resize an image to fixed height while preserving aspect ratio."""

    source_path = Path(source)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image = image.convert(image_mode)
        width, height = image.size
        if height <= 0:
            raise ValueError(f"Invalid image height for {source_path}: {height}")
        resized_width = max(1, round(width * image_height / height))
        resized = image.resize((resized_width, image_height), Image.Resampling.BICUBIC)
        resized.save(target_path)
    return target_path


def _write_lines(output: Path, lines: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _image_extension(image_format: str) -> str:
    normalized = image_format.lower().lstrip(".")
    if normalized == "jpeg":
        normalized = "jpg"
    return f".{normalized}"


"""Prepare manifests in PaddleOCR text recognition format."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from src.constants import FOLD_COL, ID_COL, TARGET_COL
from src.features.text_normalization import normalize_text
from src.paddleocr_rec.text import collect_paddleocr_characters, write_character_dict


@dataclass(frozen=True)
class PaddleOCRDatasetPaths:
    """Output paths for a prepared PaddleOCR recognition dataset."""

    root: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path
    train_labels: Path
    val_labels: Path
    test_images: Path
    character_dict: Path
    metadata: Path


def prepare_paddleocr_dataset(
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    *,
    output_dir: str | Path,
    fold: int,
    image_height: int | None = None,
    image_mode: str = "RGB",
    image_format: str = "jpg",
    path_mode: str = "absolute",
    force: bool = False,
) -> PaddleOCRDatasetPaths:
    """Create PaddleOCR recognition image folders and label files."""

    output = Path(output_dir)
    paths = PaddleOCRDatasetPaths(
        root=output,
        train_dir=output / "train",
        val_dir=output / "val",
        test_dir=output / "test",
        train_labels=output / "rec_gt_train.txt",
        val_labels=output / "rec_gt_val.txt",
        test_images=output / "test_images.txt",
        character_dict=output / "character_dict.txt",
        metadata=output / "metadata.json",
    )
    for directory in (paths.train_dir, paths.val_dir, paths.test_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_df = train_manifest[train_manifest[FOLD_COL] != fold].reset_index(drop=True)
    val_df = train_manifest[train_manifest[FOLD_COL] == fold].reset_index(drop=True)
    test_df = test_manifest.reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError(f"Fold {fold} produced empty train or validation data.")

    train_lines = _write_labeled_split(
        train_df,
        split_dir=paths.train_dir,
        dataset_root=output,
        image_height=image_height,
        image_mode=image_mode,
        image_format=image_format,
        path_mode=path_mode,
        force=force,
        split="train",
    )
    val_lines = _write_labeled_split(
        val_df,
        split_dir=paths.val_dir,
        dataset_root=output,
        image_height=image_height,
        image_mode=image_mode,
        image_format=image_format,
        path_mode=path_mode,
        force=force,
        split="val",
    )
    test_lines = _write_unlabeled_split(
        test_df,
        split_dir=paths.test_dir,
        dataset_root=output,
        image_height=image_height,
        image_mode=image_mode,
        image_format=image_format,
        path_mode=path_mode,
        force=force,
        split="test",
    )
    _write_lines(paths.train_labels, train_lines)
    _write_lines(paths.val_labels, val_lines)
    _write_lines(paths.test_images, test_lines)

    characters = collect_paddleocr_characters(train_manifest[TARGET_COL].tolist())
    write_character_dict(characters, paths.character_dict)

    metadata = {
        "fold": fold,
        "image_height": image_height,
        "image_mode": image_mode,
        "image_format": image_format,
        "path_mode": path_mode,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "character_count_without_space": len(characters),
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
    dataset_root: Path,
    image_height: int | None,
    image_mode: str,
    image_format: str,
    path_mode: str,
    force: bool,
    split: str,
) -> list[str]:
    extension = _image_extension(image_format)
    lines: list[str] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"prepare {split}", leave=False):
        image_id = str(getattr(row, ID_COL))
        source = Path(str(getattr(row, "image_path")))
        target = split_dir / f"{image_id}{extension}"
        if force or not target.exists():
            _copy_or_resize_image(
                source,
                target,
                image_height=image_height,
                image_mode=image_mode,
            )
        image_reference = _format_image_reference(
            target,
            dataset_root=dataset_root,
            path_mode=path_mode,
        )
        text = normalize_text(getattr(row, TARGET_COL))
        lines.append(f"{image_reference}\t{text}")
    return lines


def _write_unlabeled_split(
    df: pd.DataFrame,
    *,
    split_dir: Path,
    dataset_root: Path,
    image_height: int | None,
    image_mode: str,
    image_format: str,
    path_mode: str,
    force: bool,
    split: str,
) -> list[str]:
    extension = _image_extension(image_format)
    lines: list[str] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"prepare {split}", leave=False):
        image_id = str(getattr(row, ID_COL))
        source = Path(str(getattr(row, "image_path")))
        target = split_dir / f"{image_id}{extension}"
        if force or not target.exists():
            _copy_or_resize_image(
                source,
                target,
                image_height=image_height,
                image_mode=image_mode,
            )
        lines.append(
            _format_image_reference(
                target,
                dataset_root=dataset_root,
                path_mode=path_mode,
            )
        )
    return lines


def _copy_or_resize_image(
    source: str | Path,
    target: str | Path,
    *,
    image_height: int | None,
    image_mode: str,
) -> Path:
    source_path = Path(source)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if image_height is None:
        if target_path.suffix.lower() == source_path.suffix.lower():
            shutil.copy2(source_path, target_path)
            return target_path
    with Image.open(source_path) as image:
        image = image.convert(image_mode)
        if image_height is not None:
            width, height = image.size
            if height <= 0:
                raise ValueError(f"Invalid image height for {source_path}: {height}")
            resized_width = max(1, round(width * image_height / height))
            image = image.resize((resized_width, image_height), Image.Resampling.BICUBIC)
        image.save(target_path)
    return target_path


def _format_image_reference(
    image_path: Path,
    *,
    dataset_root: Path,
    path_mode: str,
) -> str:
    if path_mode == "absolute":
        return str(image_path.resolve())
    if path_mode == "relative":
        return str(image_path.relative_to(dataset_root))
    raise ValueError(f"Unsupported path_mode: {path_mode}")


def _write_lines(output: Path, lines: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _image_extension(image_format: str) -> str:
    normalized = image_format.lower().lstrip(".")
    if normalized == "jpeg":
        normalized = "jpg"
    return f".{normalized}"


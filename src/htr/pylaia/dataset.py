"""Prepare competition manifests in PyLaia dataset format."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from src.constants import FOLD_COL, ID_COL, TARGET_COL
from src.common.text_normalization import normalize_text
from src.htr.pylaia.charset import (
    PYLAIA_SPACE_TOKEN,
    characters_to_symbols,
    collect_characters,
    tokenize_text,
    write_syms,
)


@dataclass(frozen=True)
class PyLaiaDatasetPaths:
    """Output paths for a prepared PyLaia dataset."""

    root: Path
    images_dir: Path
    syms: Path
    train_txt: Path
    val_txt: Path
    test_txt: Path
    train_text: Path
    val_text: Path
    test_text: Path
    train_ids: Path
    val_ids: Path
    test_ids: Path
    metadata: Path


def prepare_pylaia_dataset(
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    *,
    output_dir: str | Path,
    fold: int,
    image_height: int = 128,
    image_mode: str = "L",
    image_format: str = "jpg",
    space_token: str = PYLAIA_SPACE_TOKEN,
    force: bool = False,
) -> PyLaiaDatasetPaths:
    """Create resized images and PyLaia text tables for one fold."""

    output = Path(output_dir)
    images_dir = output / "images"
    paths = PyLaiaDatasetPaths(
        root=output,
        images_dir=images_dir,
        syms=output / "syms.txt",
        train_txt=output / "train.txt",
        val_txt=output / "val.txt",
        test_txt=output / "test.txt",
        train_text=output / "train_text.txt",
        val_text=output / "val_text.txt",
        test_text=output / "test_text.txt",
        train_ids=output / "train_ids.txt",
        val_ids=output / "val_ids.txt",
        test_ids=output / "test_ids.txt",
        metadata=output / "metadata.json",
    )
    output.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    train_df = train_manifest[train_manifest[FOLD_COL] != fold].reset_index(drop=True)
    val_df = train_manifest[train_manifest[FOLD_COL] == fold].reset_index(drop=True)
    test_df = test_manifest.reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError(f"Fold {fold} produced empty train or validation data.")

    _prepare_images(
        train_df,
        split="train",
        images_dir=images_dir,
        image_height=image_height,
        image_mode=image_mode,
        image_format=image_format,
        force=force,
    )
    _prepare_images(
        val_df,
        split="val",
        images_dir=images_dir,
        image_height=image_height,
        image_mode=image_mode,
        image_format=image_format,
        force=force,
    )
    _prepare_images(
        test_df,
        split="test",
        images_dir=images_dir,
        image_height=image_height,
        image_mode=image_mode,
        image_format=image_format,
        force=force,
    )

    symbols = characters_to_symbols(
        collect_characters(train_manifest[TARGET_COL].tolist()),
        space_token=space_token,
    )
    write_syms(symbols, paths.syms)
    _write_training_tables(train_df, paths.train_txt, paths.train_text, paths.train_ids, "train")
    _write_training_tables(val_df, paths.val_txt, paths.val_text, paths.val_ids, "val")
    _write_test_tables(test_df, paths.test_txt, paths.test_text, paths.test_ids, "test")

    metadata = {
        "fold": fold,
        "image_height": image_height,
        "image_mode": image_mode,
        "image_format": image_format,
        "space_token": space_token,
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


def _prepare_images(
    df: pd.DataFrame,
    *,
    split: str,
    images_dir: Path,
    image_height: int,
    image_mode: str,
    image_format: str,
    force: bool,
) -> None:
    split_dir = images_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    extension = _image_extension(image_format)
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"prepare {split}", leave=False):
        image_id = str(getattr(row, ID_COL))
        source = Path(str(getattr(row, "image_path")))
        target = split_dir / f"{image_id}{extension}"
        if target.exists() and not force:
            continue
        resize_line_image(
            source,
            target,
            image_height=image_height,
            image_mode=image_mode,
        )


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


def _write_training_tables(
    df: pd.DataFrame,
    tokenized_output: Path,
    text_output: Path,
    ids_output: Path,
    split: str,
) -> None:
    tokenized_lines: list[str] = []
    text_lines: list[str] = []
    ids: list[str] = []
    for row in df.itertuples(index=False):
        image_name = f"{split}/{getattr(row, ID_COL)}"
        text = normalize_text(getattr(row, TARGET_COL))
        tokenized_lines.append(f"{image_name} {tokenize_text(text)}")
        text_lines.append(f"{image_name} {text}")
        ids.append(image_name)
    _write_lines(tokenized_output, tokenized_lines)
    _write_lines(text_output, text_lines)
    _write_lines(ids_output, ids)


def _write_test_tables(
    df: pd.DataFrame,
    tokenized_output: Path,
    text_output: Path,
    ids_output: Path,
    split: str,
) -> None:
    ids = [f"{split}/{image_id}" for image_id in df[ID_COL].astype(str).tolist()]
    _write_lines(tokenized_output, ids)
    _write_lines(text_output, ids)
    _write_lines(ids_output, ids)


def _write_lines(output: Path, lines: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _image_extension(image_format: str) -> str:
    normalized = image_format.lower().lstrip(".")
    if normalized == "jpeg":
        normalized = "jpg"
    return f".{normalized}"


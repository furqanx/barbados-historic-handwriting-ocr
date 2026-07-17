"""Build train/test manifests with image metadata and folds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from src.constants import (
    FOLD_COL,
    ID_COL,
    IMAGE_DIR,
    IMAGE_EXT,
    N_FOLDS,
    RANDOM_SEED,
    SAMPLE_SUBMISSION_CSV,
    TARGET_COL,
    TEST_CSV,
    TEST_MANIFEST,
    TRAIN_CSV,
    TRAIN_MANIFEST,
)


@dataclass(frozen=True)
class ManifestPaths:
    """Output paths produced by manifest generation."""

    train_manifest: Path
    test_manifest: Path


def image_path_from_id(image_id: str, image_dir: Path = IMAGE_DIR) -> Path:
    """Return the expected image path for an ID."""

    return image_dir / f"{image_id}{IMAGE_EXT}"


def load_competition_csvs(
    train_path: Path = TRAIN_CSV,
    test_path: Path = TEST_CSV,
    sample_submission_path: Path = SAMPLE_SUBMISSION_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw competition CSV files."""

    train = pd.read_csv(train_path, encoding="utf-8-sig")
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    sample_submission = pd.read_csv(sample_submission_path, encoding="utf-8-sig")
    return train, test, sample_submission


def collect_image_metadata(image_dir: Path = IMAGE_DIR) -> pd.DataFrame:
    """Collect width, height, aspect ratio, area, and mode for all images."""

    records: list[dict[str, object]] = []
    for path in sorted(image_dir.glob(f"*{IMAGE_EXT}")):
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        records.append(
            {
                ID_COL: path.stem,
                "image_path": str(path),
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
                "area": width * height,
                "mode": mode,
            }
        )
    return pd.DataFrame(records)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add target text length features for train rows."""

    result = df.copy()
    target = result[TARGET_COL].fillna("").astype(str)
    result["char_len"] = target.str.len()
    result["word_len"] = target.str.split().str.len()
    return result


def _safe_qcut(values: pd.Series, q: int, prefix: str) -> pd.Series:
    """Create quantile bins, falling back gracefully when bins collapse."""

    binned = pd.qcut(values, q=q, labels=False, duplicates="drop")
    return prefix + binned.astype("Int64").astype(str)


def make_stratification_labels(train_manifest: pd.DataFrame) -> pd.Series:
    """Build labels that preserve text length and image shape distributions."""

    char_bin = _safe_qcut(train_manifest["char_len"], q=5, prefix="c")
    aspect_bin = _safe_qcut(train_manifest["aspect_ratio"], q=5, prefix="a")
    area_bin = _safe_qcut(train_manifest["area"], q=3, prefix="s")
    return char_bin + "_" + aspect_bin + "_" + area_bin


def _coarsen_rare_labels(
    labels: pd.Series,
    fallback: pd.Series,
    *,
    min_count: int,
) -> pd.Series:
    """Replace labels with low support by a coarser fallback label."""

    result = labels.copy()
    counts = result.value_counts()
    rare_labels = counts[counts < min_count].index
    if len(rare_labels) > 0:
        result = result.mask(result.isin(rare_labels), fallback)
    return result


def add_folds(
    train_manifest: pd.DataFrame,
    *,
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Add a fold column using stratified K-fold labels."""

    result = train_manifest.copy()
    result[FOLD_COL] = -1

    char_bin = _safe_qcut(result["char_len"], q=5, prefix="c")
    aspect_bin = _safe_qcut(result["aspect_ratio"], q=5, prefix="a")
    shape_fallback = char_bin + "_" + aspect_bin

    stratify_labels = make_stratification_labels(result)
    stratify_labels = _coarsen_rare_labels(
        stratify_labels,
        shape_fallback,
        min_count=n_folds,
    )
    stratify_labels = _coarsen_rare_labels(
        stratify_labels,
        char_bin,
        min_count=n_folds,
    )
    if stratify_labels.value_counts().min() < n_folds:
        stratify_labels = char_bin

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(splitter.split(result, stratify_labels)):
        result.loc[result.index[valid_idx], FOLD_COL] = fold

    return result


def build_manifests(
    *,
    image_dir: Path = IMAGE_DIR,
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/test manifests with image metadata and train folds."""

    train, test, sample_submission = load_competition_csvs()
    if set(test[ID_COL].astype(str)) != set(sample_submission[ID_COL].astype(str)):
        raise ValueError("Test IDs and sample submission IDs do not match.")

    image_meta = collect_image_metadata(image_dir)
    train_manifest = add_text_features(train).merge(image_meta, on=ID_COL, how="left")
    test_manifest = test.merge(image_meta, on=ID_COL, how="left")

    missing_train = train_manifest["image_path"].isna().sum()
    missing_test = test_manifest["image_path"].isna().sum()
    if missing_train or missing_test:
        raise FileNotFoundError(
            f"Missing images: train={missing_train}, test={missing_test}"
        )

    train_manifest = add_folds(train_manifest, n_folds=n_folds, seed=seed)
    return train_manifest, test_manifest


def save_manifests(
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    *,
    train_output: Path = TRAIN_MANIFEST,
    test_output: Path = TEST_MANIFEST,
) -> ManifestPaths:
    """Save manifests to CSV and return their paths."""

    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)
    train_manifest.to_csv(train_output, index=False)
    test_manifest.to_csv(test_output, index=False)
    return ManifestPaths(train_manifest=train_output, test_manifest=test_output)

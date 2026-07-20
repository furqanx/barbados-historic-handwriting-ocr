"""Utilities for constructing small-subset overfit manifests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import FOLD_COL, ID_COL, TARGET_COL
from src.common.text_normalization import normalize_text


def make_overfit_manifest(
    manifest: pd.DataFrame,
    *,
    n_samples: int = 64,
    valid_fold: int = 0,
    train_fold: int = 1,
    seed: int = 42,
    require_chars: str = "",
    mirror_valid: bool = True,
) -> pd.DataFrame:
    """Create a manifest designed to test whether a model can memorize a tiny subset."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if ID_COL not in manifest.columns or TARGET_COL not in manifest.columns:
        raise ValueError(f"Manifest must contain {ID_COL!r} and {TARGET_COL!r}.")

    candidates = manifest.copy()
    if require_chars:
        required = set(require_chars)
        mask = candidates[TARGET_COL].map(lambda text: bool(set(normalize_text(text)) & required))
        rare_rows = candidates[mask]
        remaining = candidates[~mask]
        rare_take = rare_rows.sample(
            n=min(len(rare_rows), n_samples),
            random_state=seed,
        )
        if len(rare_take) < n_samples:
            rest = remaining.sample(
                n=n_samples - len(rare_take),
                random_state=seed,
            )
            selected = pd.concat([rare_take, rest], ignore_index=True)
        else:
            selected = rare_take
    else:
        selected = candidates.sample(n=min(n_samples, len(candidates)), random_state=seed)

    selected = selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if not mirror_valid:
        selected[FOLD_COL] = valid_fold
        return selected

    train_copy = selected.copy()
    valid_copy = selected.copy()
    train_copy[FOLD_COL] = train_fold
    valid_copy[FOLD_COL] = valid_fold
    train_copy["diagnostic_split"] = "train_overfit"
    valid_copy["diagnostic_split"] = "valid_same_samples"
    return pd.concat([train_copy, valid_copy], ignore_index=True)


def write_overfit_manifest(
    manifest: pd.DataFrame,
    output_path: str | Path,
    *,
    n_samples: int = 64,
    valid_fold: int = 0,
    train_fold: int = 1,
    seed: int = 42,
    require_chars: str = "",
    mirror_valid: bool = True,
) -> Path:
    """Create and write an overfit manifest."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    overfit = make_overfit_manifest(
        manifest,
        n_samples=n_samples,
        valid_fold=valid_fold,
        train_fold=train_fold,
        seed=seed,
        require_chars=require_chars,
        mirror_valid=mirror_valid,
    )
    overfit.to_csv(output, index=False)
    return output

"""Parse Kraken OCR text outputs into project prediction CSV format."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import ID_COL, TARGET_COL
from src.features.text_normalization import normalize_text
from src.kraken_htr.text import normalize_for_kraken


def load_kraken_prediction_dir(
    prediction_dir: str | Path,
    *,
    output_unicode_normalization: str = "NFC",
) -> pd.DataFrame:
    """Load one plain text OCR output per image ID from a directory."""

    rows: list[dict[str, str]] = []
    for path in sorted(Path(prediction_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        text = normalize_text(text)
        text = normalize_for_kraken(text, unicode_form=output_unicode_normalization)
        rows.append({ID_COL: path.stem, TARGET_COL: text})
    return pd.DataFrame(rows, columns=[ID_COL, TARGET_COL])


def align_predictions_to_sample(
    predictions: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """Align predictions to sample submission ID order."""

    prediction_map = predictions.set_index(ID_COL)[TARGET_COL].to_dict()
    missing_ids = sorted(set(sample_submission[ID_COL].astype(str)) - set(prediction_map))
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(f"Missing predictions for {len(missing_ids)} IDs: {preview}")
    submission = sample_submission.copy()
    submission[TARGET_COL] = submission[ID_COL].astype(str).map(prediction_map).fillna("")
    return submission


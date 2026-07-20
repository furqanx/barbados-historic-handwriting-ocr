"""Parse PaddleOCR recognition output into project CSV format."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.constants import ID_COL, TARGET_COL
from src.common.text_normalization import normalize_text


def load_paddleocr_prediction_file(path: str | Path) -> pd.DataFrame:
    """Load a PaddleOCR prediction text file.

    Supported line styles:
    - `image_path\tprediction`
    - `image_path\t[('prediction', 0.98)]`
    - `image_path\t{"rec_text": "prediction"}`
    """

    rows: list[dict[str, str]] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        image_name, prediction = _split_prediction_line(line)
        rows.append(
            {
                ID_COL: Path(image_name).stem,
                TARGET_COL: normalize_text(prediction),
            }
        )
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


def _split_prediction_line(line: str) -> tuple[str, str]:
    if "\t" in line:
        image_name, raw_prediction = line.split("\t", 1)
    else:
        parts = line.split(maxsplit=1)
        image_name = parts[0]
        raw_prediction = parts[1] if len(parts) > 1 else ""
    return image_name, _extract_prediction_text(raw_prediction.strip())


def _extract_prediction_text(raw_prediction: str) -> str:
    if not raw_prediction:
        return ""
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(raw_prediction)
        except Exception:
            continue
        extracted = _extract_text_from_object(value)
        if extracted is not None:
            return extracted
    return raw_prediction


def _extract_text_from_object(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("rec_text", "text", "label", "transcription"):
            if key in value:
                return str(value[key])
        return None
    if isinstance(value, list) and value:
        return _extract_text_from_object(value[0])
    if isinstance(value, tuple) and value:
        return _extract_text_from_object(value[0])
    return None


"""Parse PyLaia decode output and convert it to project CSV format."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import ID_COL, TARGET_COL
from src.htr.pylaia.charset import detokenize_text


def parse_pylaia_decode_output(
    raw_output: str,
    *,
    id_prefix_to_strip: str | None = None,
) -> pd.DataFrame:
    """Parse stdout produced by `pylaia-htr-decode-ctc`."""

    rows: list[dict[str, str]] = []
    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("["):
            continue
        parts = line.split(maxsplit=1)
        image_name = parts[0]
        tokenized = parts[1] if len(parts) > 1 else ""
        image_id = normalize_prediction_id(image_name, id_prefix_to_strip=id_prefix_to_strip)
        rows.append({ID_COL: image_id, TARGET_COL: detokenize_text(tokenized)})
    return pd.DataFrame(rows, columns=[ID_COL, TARGET_COL])


def load_pylaia_predictions(
    path: str | Path,
    *,
    id_prefix_to_strip: str | None = None,
) -> pd.DataFrame:
    """Load and parse a PyLaia raw prediction text file."""

    return parse_pylaia_decode_output(
        Path(path).read_text(encoding="utf-8"),
        id_prefix_to_strip=id_prefix_to_strip,
    )


def normalize_prediction_id(
    image_name: str,
    *,
    id_prefix_to_strip: str | None = None,
) -> str:
    """Normalize PyLaia image names back to competition IDs."""

    value = image_name
    if id_prefix_to_strip and value.startswith(id_prefix_to_strip):
        value = value[len(id_prefix_to_strip):]
    value = Path(value).name
    return Path(value).stem


def align_predictions_to_sample(
    predictions: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """Align predictions to the sample submission ID order."""

    prediction_map = predictions.set_index(ID_COL)[TARGET_COL].to_dict()
    missing_ids = sorted(set(sample_submission[ID_COL].astype(str)) - set(prediction_map))
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(f"Missing predictions for {len(missing_ids)} IDs: {preview}")
    submission = sample_submission.copy()
    submission[TARGET_COL] = submission[ID_COL].astype(str).map(prediction_map).fillna("")
    return submission


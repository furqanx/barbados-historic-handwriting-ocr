"""Utilities for ensembling transcription prediction CSV files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.constants import ID_COL, TARGET_COL
from src.evaluation.metrics import TranscriptionScore, score_transcriptions


@dataclass(frozen=True)
class NamedPredictions:
    """Predictions loaded from one model run."""

    name: str
    predictions: pd.DataFrame


def load_named_predictions(spec: str) -> NamedPredictions:
    """Load one `name:path.csv` prediction spec."""

    if ":" not in spec:
        raise ValueError("Prediction spec must use format name:path.csv")
    name, raw_path = spec.split(":", maxsplit=1)
    path = Path(raw_path)
    predictions = pd.read_csv(path)
    _validate_prediction_frame(predictions, source=name)
    return NamedPredictions(name=name, predictions=predictions)


def ensemble_predictions(
    prediction_sets: list[NamedPredictions],
    *,
    priority: list[str] | None = None,
) -> pd.DataFrame:
    """Combine prediction CSVs with majority voting and priority tie-breaks."""

    if not prediction_sets:
        raise ValueError("At least one prediction set is required.")

    priority = priority or [prediction_set.name for prediction_set in prediction_sets]
    priority_rank = {name: rank for rank, name in enumerate(priority)}
    base_ids = prediction_sets[0].predictions[ID_COL].astype(str).tolist()
    for prediction_set in prediction_sets[1:]:
        ids = prediction_set.predictions[ID_COL].astype(str).tolist()
        if ids != base_ids:
            raise ValueError(
                "Prediction files must have identical ID order. "
                f"First mismatch source: {prediction_set.name}"
            )

    rows: list[dict[str, str]] = []
    prediction_maps = [
        (
            prediction_set.name,
            prediction_set.predictions[TARGET_COL].fillna("").astype(str).tolist(),
        )
        for prediction_set in prediction_sets
    ]
    for row_idx, image_id in enumerate(base_ids):
        candidates = [
            (name, predictions[row_idx])
            for name, predictions in prediction_maps
        ]
        rows.append(
            {
                ID_COL: image_id,
                TARGET_COL: _choose_prediction(candidates, priority_rank),
            }
        )

    return pd.DataFrame(rows)


def evaluate_prediction_sets(
    prediction_sets: list[NamedPredictions],
) -> pd.DataFrame:
    """Evaluate each prediction set that contains reference text."""

    rows: list[dict[str, float | str]] = []
    for prediction_set in prediction_sets:
        predictions = prediction_set.predictions
        if "reference" not in predictions.columns:
            continue
        score = score_transcriptions(predictions["reference"], predictions[TARGET_COL])
        rows.append(_score_row(prediction_set.name, score))
    return pd.DataFrame(rows)


def evaluate_ensemble(
    ensemble: pd.DataFrame,
    reference_predictions: pd.DataFrame,
    *,
    name: str = "ensemble",
) -> pd.DataFrame:
    """Evaluate ensemble predictions against references from validation CSV."""

    if "reference" not in reference_predictions.columns:
        return pd.DataFrame()
    merged = ensemble.merge(
        reference_predictions[[ID_COL, "reference"]],
        on=ID_COL,
        how="left",
    )
    score = score_transcriptions(merged["reference"], merged[TARGET_COL])
    return pd.DataFrame([_score_row(name, score)])


def make_submission(
    predictions: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """Align predictions to sample submission ID order."""

    prediction_map = predictions.set_index(ID_COL)[TARGET_COL].to_dict()
    submission = sample_submission.copy()
    missing_ids = sorted(set(submission[ID_COL].astype(str)) - set(prediction_map))
    if missing_ids:
        preview = ", ".join(map(str, missing_ids[:5]))
        raise ValueError(f"Missing predictions for {len(missing_ids)} IDs: {preview}")
    submission[TARGET_COL] = submission[ID_COL].astype(str).map(prediction_map).fillna("")
    return submission


def _choose_prediction(
    candidates: list[tuple[str, str]],
    priority_rank: dict[str, int],
) -> str:
    non_empty = [(name, text) for name, text in candidates if text.strip()]
    candidates = non_empty or candidates
    counts = Counter(text for _, text in candidates)
    best_count = max(counts.values())
    tied_texts = {text for text, count in counts.items() if count == best_count}
    if len(tied_texts) == 1:
        return next(iter(tied_texts))

    for name, text in sorted(
        candidates,
        key=lambda item: priority_rank.get(item[0], len(priority_rank)),
    ):
        if text in tied_texts:
            return text
    return candidates[0][1]


def _validate_prediction_frame(predictions: pd.DataFrame, *, source: str) -> None:
    missing = {ID_COL, TARGET_COL} - set(predictions.columns)
    if missing:
        raise ValueError(f"{source} predictions are missing columns: {sorted(missing)}")


def _score_row(name: str, score: TranscriptionScore) -> dict[str, float | str]:
    return {
        "name": name,
        "wer": score.wer,
        "cer": score.cer,
        "score": score.score,
    }

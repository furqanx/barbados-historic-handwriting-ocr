import pandas as pd
import pytest

from src.ensemble.csv_ensemble import (
    NamedPredictions,
    ensemble_predictions,
    make_submission,
)


def test_ensemble_uses_majority_vote() -> None:
    prediction_sets = [
        NamedPredictions("a", _predictions(["one", "two"])),
        NamedPredictions("b", _predictions(["one", "too"])),
        NamedPredictions("c", _predictions(["won", "too"])),
    ]

    ensembled = ensemble_predictions(prediction_sets)

    assert ensembled["Target"].tolist() == ["one", "too"]


def test_ensemble_uses_priority_for_ties() -> None:
    prediction_sets = [
        NamedPredictions("crnn", _predictions(["literal"])),
        NamedPredictions("trocr", _predictions(["language"])),
    ]

    ensembled = ensemble_predictions(prediction_sets, priority=["trocr", "crnn"])

    assert ensembled["Target"].tolist() == ["language"]


def test_ensemble_prefers_non_empty_prediction() -> None:
    prediction_sets = [
        NamedPredictions("crnn", _predictions([""])),
        NamedPredictions("trocr", _predictions(["filled"])),
    ]

    ensembled = ensemble_predictions(prediction_sets, priority=["crnn", "trocr"])

    assert ensembled["Target"].tolist() == ["filled"]


def test_ensemble_rejects_misaligned_ids() -> None:
    prediction_sets = [
        NamedPredictions("a", _predictions(["x"], ids=["id-1"])),
        NamedPredictions("b", _predictions(["x"], ids=["id-2"])),
    ]

    with pytest.raises(ValueError, match="identical ID order"):
        ensemble_predictions(prediction_sets)


def test_make_submission_aligns_to_sample_order() -> None:
    predictions = pd.DataFrame(
        {"ID": ["id-1", "id-2"], "Target": ["first", "second"]}
    )
    sample = pd.DataFrame({"ID": ["id-2", "id-1"], "Target": ["", ""]})

    submission = make_submission(predictions, sample)

    assert submission["Target"].tolist() == ["second", "first"]


def _predictions(values: list[str], ids: list[str] | None = None) -> pd.DataFrame:
    ids = ids or [f"id-{idx}" for idx in range(len(values))]
    return pd.DataFrame({"ID": ids, "Target": values})

import pandas as pd

from src.ensemble.selective import (
    align_prediction_sources,
    consensus_replace,
    keep_anchor,
    weighted_vote,
)


def test_keep_anchor_returns_anchor_predictions() -> None:
    result = keep_anchor("anchor", _predictions(["aa", "bb"]))

    assert result.predictions["Target"].tolist() == ["aa", "bb"]
    assert result.audit["changed_from_anchor"].tolist() == [False, False]


def test_consensus_replace_uses_non_anchor_agreement() -> None:
    ids, sources = align_prediction_sources(
        "anchor",
        _predictions(["wrong", "keep"]),
        {
            "fold0": _predictions(["right", "a"]),
            "fold2": _predictions(["right", "b"]),
            "fold3": _predictions(["right", "c"]),
        },
    )

    result = consensus_replace(ids, sources, min_consensus=3)

    assert result.predictions["Target"].tolist() == ["right", "keep"]
    assert result.audit["changed_from_anchor"].tolist() == [True, False]


def test_consensus_replace_can_limit_length_delta() -> None:
    ids, sources = align_prediction_sources(
        "anchor",
        _predictions(["short"]),
        {
            "fold0": _predictions(["much much longer"]),
            "fold2": _predictions(["much much longer"]),
        },
    )

    result = consensus_replace(ids, sources, min_consensus=2, max_length_delta=3)

    assert result.predictions["Target"].tolist() == ["short"]


def test_length_guarded_consensus_only_replaces_outlier_anchor() -> None:
    ids, sources = align_prediction_sources(
        "anchor",
        _predictions(["tiny", "normal text"]),
        {
            "fold0": _predictions(["longer text", "other text"]),
            "fold2": _predictions(["longer text", "other text"]),
        },
    )

    result = consensus_replace(
        ids,
        sources,
        min_consensus=2,
        min_anchor_outlier_delta=4,
    )

    assert result.predictions["Target"].tolist() == ["longer text", "normal text"]


def test_weighted_vote_can_protect_anchor() -> None:
    ids, sources = align_prediction_sources(
        "anchor",
        _predictions(["anchor"]),
        {
            "fold0": _predictions(["other"]),
            "fold2": _predictions(["other"]),
        },
    )

    result = weighted_vote(
        ids,
        sources,
        weights={"anchor": 3.0, "fold0": 1.0, "fold2": 1.0},
    )

    assert result.predictions["Target"].tolist() == ["anchor"]


def _predictions(values: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [f"id-{idx}" for idx in range(len(values))],
            "Target": values,
        }
    )


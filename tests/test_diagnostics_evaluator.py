import pandas as pd

from src.constants import ID_COL, TARGET_COL
from src.diagnostics.evaluator import (
    align_truth_and_predictions,
    edit_counts,
    make_row_error_table,
)


def test_edit_counts_tracks_insert_delete_substitute() -> None:
    counts = edit_counts(list("abc"), list("axcd"))

    assert counts.substitutions == 1
    assert counts.insertions == 1
    assert counts.deletions == 0
    assert counts.distance == 2


def test_row_error_table_aligns_predictions_and_scores_rows() -> None:
    truth = pd.DataFrame(
        [{ID_COL: "b", TARGET_COL: "hello world"}, {ID_COL: "a", TARGET_COL: "exact"}]
    )
    pred = pd.DataFrame(
        [{ID_COL: "a", TARGET_COL: "exact"}, {ID_COL: "b", TARGET_COL: "hello word"}]
    )

    aligned = align_truth_and_predictions(truth, pred)
    rows = make_row_error_table(aligned)

    assert rows[ID_COL].tolist() == ["b", "a"]
    assert rows.loc[rows[ID_COL] == "a", "exact_match"].item() is True
    assert rows.loc[rows[ID_COL] == "b", "row_cer"].item() > 0


from pathlib import Path

import pandas as pd

from src.constants import ID_COL, TARGET_COL
from src.kraken_htr.prediction_parser import (
    align_predictions_to_sample,
    load_kraken_prediction_dir,
)


def test_load_kraken_prediction_dir(tmp_path: Path) -> None:
    (tmp_path / "abc.txt").write_text("By this\n", encoding="utf-8")

    predictions = load_kraken_prediction_dir(tmp_path)

    assert predictions.to_dict("records") == [{ID_COL: "abc", TARGET_COL: "By this"}]


def test_align_predictions_to_sample() -> None:
    predictions = pd.DataFrame([{ID_COL: "b", TARGET_COL: "two"}, {ID_COL: "a", TARGET_COL: "one"}])
    sample = pd.DataFrame([{ID_COL: "a", TARGET_COL: ""}, {ID_COL: "b", TARGET_COL: ""}])

    submission = align_predictions_to_sample(predictions, sample)

    assert submission[TARGET_COL].tolist() == ["one", "two"]


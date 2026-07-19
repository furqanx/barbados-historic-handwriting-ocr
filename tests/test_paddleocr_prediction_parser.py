import pandas as pd

from src.constants import ID_COL, TARGET_COL
from src.paddleocr_rec.prediction_parser import (
    align_predictions_to_sample,
    load_paddleocr_prediction_file,
)


def test_load_paddleocr_plain_prediction_file(tmp_path) -> None:
    output = tmp_path / "pred.txt"
    output.write_text("/tmp/a.jpg\tBy this\n", encoding="utf-8")

    predictions = load_paddleocr_prediction_file(output)

    assert predictions.to_dict("records") == [{ID_COL: "a", TARGET_COL: "By this"}]


def test_load_paddleocr_structured_prediction_file(tmp_path) -> None:
    output = tmp_path / "pred.txt"
    output.write_text("/tmp/a.jpg\t[('By this', 0.99)]\n", encoding="utf-8")

    predictions = load_paddleocr_prediction_file(output)

    assert predictions[TARGET_COL].tolist() == ["By this"]


def test_align_paddleocr_predictions_to_sample() -> None:
    predictions = pd.DataFrame([{ID_COL: "b", TARGET_COL: "two"}, {ID_COL: "a", TARGET_COL: "one"}])
    sample = pd.DataFrame([{ID_COL: "a", TARGET_COL: ""}, {ID_COL: "b", TARGET_COL: ""}])

    submission = align_predictions_to_sample(predictions, sample)

    assert submission[TARGET_COL].tolist() == ["one", "two"]


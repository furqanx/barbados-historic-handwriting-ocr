"""Shared evaluation and row-level error diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.constants import ID_COL, TARGET_COL
from src.evaluation.metrics import score_transcriptions
from src.features.text_normalization import normalize_text


@dataclass(frozen=True)
class EditCounts:
    """Insertion, deletion, and substitution counts."""

    insertions: int = 0
    deletions: int = 0
    substitutions: int = 0
    matches: int = 0

    @property
    def distance(self) -> int:
        return self.insertions + self.deletions + self.substitutions


@dataclass(frozen=True)
class EvaluationSummary:
    """Corpus-level evaluation summary."""

    rows: int
    wer: float
    cer: float
    score: float
    exact_match_rate: float
    mean_row_cer: float
    mean_row_wer: float
    total_char_insertions: int
    total_char_deletions: int
    total_char_substitutions: int
    total_word_insertions: int
    total_word_deletions: int
    total_word_substitutions: int


def align_truth_and_predictions(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    id_col: str = ID_COL,
    truth_col: str = TARGET_COL,
    pred_col: str = TARGET_COL,
    allow_prediction_subset: bool = False,
) -> pd.DataFrame:
    """Align prediction rows to a truth dataframe by ID."""

    required_truth = {id_col, truth_col}
    required_pred = {id_col, pred_col}
    missing_truth = required_truth - set(truth.columns)
    missing_pred = required_pred - set(predictions.columns)
    if missing_truth:
        raise ValueError(f"Truth dataframe is missing columns: {sorted(missing_truth)}")
    if missing_pred:
        raise ValueError(f"Prediction dataframe is missing columns: {sorted(missing_pred)}")
    if truth[id_col].duplicated().any():
        raise ValueError("Truth dataframe contains duplicate IDs.")
    if predictions[id_col].duplicated().any():
        raise ValueError("Prediction dataframe contains duplicate IDs.")

    unknown_prediction_ids = sorted(set(predictions[id_col]) - set(truth[id_col]))
    if unknown_prediction_ids:
        raise ValueError(f"Predictions contain unknown IDs: {unknown_prediction_ids[:10]}")
    if allow_prediction_subset:
        truth = truth[truth[id_col].isin(set(predictions[id_col]))].copy()

    truth_prepared = truth[[id_col, truth_col]].rename(columns={truth_col: "reference"})
    predictions_prepared = predictions[[id_col, pred_col]].rename(
        columns={pred_col: "prediction"}
    )
    predictions_prepared["__has_prediction_row"] = True
    aligned = truth_prepared.merge(
        predictions_prepared,
        on=id_col,
        how="left",
        validate="one_to_one",
    )
    if aligned["__has_prediction_row"].isna().any():
        missing = aligned.loc[aligned["__has_prediction_row"].isna(), id_col].head(10).tolist()
        raise ValueError(f"Missing predictions for IDs: {missing}")
    aligned = aligned.drop(columns=["__has_prediction_row"])

    return aligned


def make_row_error_table(aligned: pd.DataFrame, *, normalize: bool = True) -> pd.DataFrame:
    """Compute row-level CER/WER and edit counts."""

    rows: list[dict[str, object]] = []
    for row in aligned.itertuples(index=False):
        image_id = str(getattr(row, ID_COL))
        reference = _prepare(getattr(row, "reference"), normalize=normalize)
        prediction = _prepare(getattr(row, "prediction"), normalize=normalize)

        char_counts = edit_counts(list(reference), list(prediction))
        word_counts = edit_counts(reference.split(), prediction.split())
        rows.append(
            {
                ID_COL: image_id,
                "reference": reference,
                "prediction": prediction,
                "exact_match": reference == prediction,
                "ref_char_len": len(reference),
                "pred_char_len": len(prediction),
                "ref_word_len": len(reference.split()),
                "pred_word_len": len(prediction.split()),
                "char_edits": char_counts.distance,
                "char_insertions": char_counts.insertions,
                "char_deletions": char_counts.deletions,
                "char_substitutions": char_counts.substitutions,
                "row_cer": _safe_rate(char_counts.distance, len(reference)),
                "word_edits": word_counts.distance,
                "word_insertions": word_counts.insertions,
                "word_deletions": word_counts.deletions,
                "word_substitutions": word_counts.substitutions,
                "row_wer": _safe_rate(word_counts.distance, len(reference.split())),
            }
        )
    return pd.DataFrame(rows)


def summarize_row_errors(row_errors: pd.DataFrame) -> EvaluationSummary:
    """Summarize row-level diagnostics into one corpus report."""

    score = score_transcriptions(
        row_errors["reference"].tolist(),
        row_errors["prediction"].tolist(),
        normalize=False,
    )
    return EvaluationSummary(
        rows=len(row_errors),
        wer=score.wer,
        cer=score.cer,
        score=score.score,
        exact_match_rate=float(row_errors["exact_match"].mean()),
        mean_row_cer=float(row_errors["row_cer"].mean()),
        mean_row_wer=float(row_errors["row_wer"].mean()),
        total_char_insertions=int(row_errors["char_insertions"].sum()),
        total_char_deletions=int(row_errors["char_deletions"].sum()),
        total_char_substitutions=int(row_errors["char_substitutions"].sum()),
        total_word_insertions=int(row_errors["word_insertions"].sum()),
        total_word_deletions=int(row_errors["word_deletions"].sum()),
        total_word_substitutions=int(row_errors["word_substitutions"].sum()),
    )


def character_confusion_table(
    references: Iterable[str],
    predictions: Iterable[str],
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """Create a character-level operation table for error analysis."""

    counter: Counter[tuple[str, str, str]] = Counter()
    for reference, prediction in zip(references, predictions, strict=True):
        ref = _prepare(reference, normalize=normalize)
        pred = _prepare(prediction, normalize=normalize)
        for op, src, dst in edit_operations(list(ref), list(pred)):
            if op != "match":
                counter[(op, src, dst)] += 1

    rows = [
        {"operation": op, "reference_char": src, "prediction_char": dst, "count": count}
        for (op, src, dst), count in counter.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["operation", "reference_char", "prediction_char", "count"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def evaluate_predictions(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    output_dir: str | Path,
    id_col: str = ID_COL,
    truth_col: str = TARGET_COL,
    pred_col: str = TARGET_COL,
    normalize: bool = True,
    allow_prediction_subset: bool = False,
) -> tuple[EvaluationSummary, pd.DataFrame, pd.DataFrame]:
    """Run full evaluation and write summary, row errors, and confusions."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aligned = align_truth_and_predictions(
        truth,
        predictions,
        id_col=id_col,
        truth_col=truth_col,
        pred_col=pred_col,
        allow_prediction_subset=allow_prediction_subset,
    )
    row_errors = make_row_error_table(aligned, normalize=normalize)
    summary = summarize_row_errors(row_errors)
    confusions = character_confusion_table(
        row_errors["reference"].tolist(),
        row_errors["prediction"].tolist(),
        normalize=False,
    )

    (output / "evaluation_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    row_errors.sort_values("row_cer", ascending=False).to_csv(
        output / "row_errors.csv",
        index=False,
    )
    confusions.to_csv(output / "character_confusions.csv", index=False)
    return summary, row_errors, confusions


def edit_counts(source: Sequence[object], target: Sequence[object]) -> EditCounts:
    """Compute Levenshtein edit counts with deterministic tie-breaking."""

    operations = edit_operations(source, target)
    counts = Counter(op for op, _, _ in operations)
    return EditCounts(
        insertions=counts["insert"],
        deletions=counts["delete"],
        substitutions=counts["substitute"],
        matches=counts["match"],
    )


def edit_operations(
    source: Sequence[object],
    target: Sequence[object],
) -> list[tuple[str, str, str]]:
    """Return an edit script from source to target."""

    n = len(source)
    m = len(target)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "insert"

    priority = {"match": 0, "substitute": 1, "delete": 2, "insert": 3}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates: list[tuple[int, str]] = []
            if source[i - 1] == target[j - 1]:
                candidates.append((dp[i - 1][j - 1], "match"))
            else:
                candidates.append((dp[i - 1][j - 1] + 1, "substitute"))
            candidates.append((dp[i - 1][j] + 1, "delete"))
            candidates.append((dp[i][j - 1] + 1, "insert"))
            cost, op = min(candidates, key=lambda item: (item[0], priority[item[1]]))
            dp[i][j] = cost
            back[i][j] = op

    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op in {"match", "substitute"}:
            ops.append((op, str(source[i - 1]), str(target[j - 1])))
            i -= 1
            j -= 1
        elif op == "delete":
            ops.append((op, str(source[i - 1]), ""))
            i -= 1
        elif op == "insert":
            ops.append((op, "", str(target[j - 1])))
            j -= 1
        else:
            raise RuntimeError("Invalid edit backtrace state.")
    ops.reverse()
    return ops


def _prepare(text: object, *, normalize: bool) -> str:
    value = "" if text is None else str(text)
    return normalize_text(value) if normalize else value


def _safe_rate(distance: int, reference_length: int) -> float:
    if reference_length == 0:
        return 0.0 if distance == 0 else 1.0
    return distance / reference_length

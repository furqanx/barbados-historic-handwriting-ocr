"""Grouped OCR error analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import ID_COL, TARGET_COL
from src.diagnostics.evaluator import align_truth_and_predictions, make_row_error_table


def attach_error_groups(
    row_errors: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    rare_chars: str = "^*#|\\",
) -> pd.DataFrame:
    """Attach interpretable diagnostic groups to row-level errors."""

    metadata_cols = [ID_COL]
    for col in ["width", "height", "aspect_ratio", "fold"]:
        if col in manifest.columns:
            metadata_cols.append(col)
    enriched = row_errors.merge(
        manifest[metadata_cols],
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )
    enriched["length_group"] = pd.cut(
        enriched["ref_char_len"],
        bins=[-1, 39, 80, 10_000],
        labels=["short", "medium", "long"],
    ).astype(str)
    enriched["has_rare_symbol"] = enriched["reference"].map(
        lambda text: any(char in rare_chars for char in str(text))
    )
    enriched["punctuation_heavy"] = enriched["reference"].map(_punctuation_ratio).ge(0.08)
    enriched["digit_heavy"] = enriched["reference"].map(_digit_ratio).ge(0.08)
    enriched["prediction_shorter"] = enriched["pred_char_len"] < enriched["ref_char_len"]
    enriched["prediction_longer"] = enriched["pred_char_len"] > enriched["ref_char_len"]
    if "aspect_ratio" in enriched.columns:
        enriched["aspect_group"] = pd.cut(
            enriched["aspect_ratio"],
            bins=[-1, 12, 22, 10_000],
            labels=["low_ar", "mid_ar", "high_ar"],
        ).astype(str)
    return enriched


def grouped_error_summary(enriched_errors: pd.DataFrame) -> pd.DataFrame:
    """Summarize CER/WER by diagnostic groups."""

    group_cols = [
        "length_group",
        "has_rare_symbol",
        "punctuation_heavy",
        "digit_heavy",
        "prediction_shorter",
        "prediction_longer",
    ]
    if "aspect_group" in enriched_errors.columns:
        group_cols.append("aspect_group")

    frames = []
    for col in group_cols:
        summary = (
            enriched_errors.groupby(col, dropna=False)
            .agg(
                rows=(ID_COL, "count"),
                mean_row_cer=("row_cer", "mean"),
                mean_row_wer=("row_wer", "mean"),
                exact_match_rate=("exact_match", "mean"),
                mean_char_edits=("char_edits", "mean"),
            )
            .reset_index()
            .rename(columns={col: "group_value"})
        )
        summary.insert(0, "group_name", col)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def run_grouped_error_analysis(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    output_dir: str | Path,
    truth_col: str = TARGET_COL,
    pred_col: str = TARGET_COL,
    rare_chars: str = "^*#|\\",
    allow_prediction_subset: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate predictions and write grouped error reports."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aligned = align_truth_and_predictions(
        manifest,
        predictions,
        truth_col=truth_col,
        pred_col=pred_col,
        allow_prediction_subset=allow_prediction_subset,
    )
    row_errors = make_row_error_table(aligned)
    enriched = attach_error_groups(row_errors, manifest, rare_chars=rare_chars)
    summary = grouped_error_summary(enriched)
    enriched.sort_values("row_cer", ascending=False).to_csv(
        output / "grouped_row_errors.csv",
        index=False,
    )
    summary.to_csv(output / "grouped_error_summary.csv", index=False)
    return enriched, summary


def _punctuation_ratio(text: object) -> float:
    value = "" if text is None else str(text)
    return sum(not char.isalnum() and not char.isspace() for char in value) / max(len(value), 1)


def _digit_ratio(text: object) -> float:
    value = "" if text is None else str(text)
    return sum(char.isdigit() for char in value) / max(len(value), 1)

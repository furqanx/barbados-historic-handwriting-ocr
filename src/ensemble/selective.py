"""Anchor-based selective ensembling for transcription CSV files."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import pandas as pd

from src.constants import ID_COL, TARGET_COL


@dataclass(frozen=True)
class SelectionResult:
    """Selected predictions and row-level decision audit."""

    predictions: pd.DataFrame
    audit: pd.DataFrame


@dataclass(frozen=True)
class PredictionTable:
    """Aligned text predictions from one named source."""

    name: str
    texts: list[str]


def load_prediction_csv(path: str | Path, *, name: str | None = None) -> pd.DataFrame:
    """Load and validate one prediction/submission CSV."""

    csv_path = Path(path)
    predictions = pd.read_csv(csv_path).fillna("")
    missing = {ID_COL, TARGET_COL} - set(predictions.columns)
    if missing:
        source = name or str(csv_path)
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    if predictions[ID_COL].duplicated().any():
        source = name or str(csv_path)
        raise ValueError(f"{source} contains duplicate IDs.")
    return predictions[[ID_COL, TARGET_COL]].copy()


def parse_prediction_spec(spec: str) -> tuple[str, Path]:
    """Parse `name:path.csv` prediction spec."""

    if ":" not in spec:
        raise ValueError("Prediction spec must use format name:path.csv")
    name, raw_path = spec.split(":", maxsplit=1)
    if not name.strip():
        raise ValueError("Prediction spec name cannot be empty.")
    return name, Path(raw_path)


def align_prediction_sources(
    anchor_name: str,
    anchor: pd.DataFrame,
    candidates: dict[str, pd.DataFrame],
) -> tuple[list[str], list[PredictionTable]]:
    """Align anchor and candidate predictions to the anchor ID order."""

    ids = anchor[ID_COL].astype(str).tolist()
    tables = [
        PredictionTable(
            name=anchor_name,
            texts=anchor[TARGET_COL].fillna("").astype(str).tolist(),
        )
    ]
    expected_ids = set(ids)
    for name, predictions in candidates.items():
        candidate_ids = set(predictions[ID_COL].astype(str))
        if candidate_ids != expected_ids:
            missing = sorted(expected_ids - candidate_ids)
            extra = sorted(candidate_ids - expected_ids)
            raise ValueError(
                f"{name} IDs do not match anchor IDs. "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        aligned = anchor[[ID_COL]].merge(
            predictions[[ID_COL, TARGET_COL]],
            on=ID_COL,
            how="left",
            validate="one_to_one",
        )
        tables.append(
            PredictionTable(
                name=name,
                texts=aligned[TARGET_COL].fillna("").astype(str).tolist(),
            )
        )
    return ids, tables


def keep_anchor(
    anchor_name: str,
    anchor: pd.DataFrame,
) -> SelectionResult:
    """Return the anchor unchanged with an audit table."""

    predictions = anchor[[ID_COL, TARGET_COL]].copy()
    audit = pd.DataFrame(
        {
            ID_COL: predictions[ID_COL].astype(str),
            "anchor_text": predictions[TARGET_COL].fillna("").astype(str),
            "chosen_text": predictions[TARGET_COL].fillna("").astype(str),
            "chosen_source": anchor_name,
            "strategy": "anchor",
            "changed_from_anchor": False,
            "top_non_anchor_count": 0,
            "candidate_count": 0,
            "anchor_len": predictions[TARGET_COL].fillna("").astype(str).str.len(),
            "chosen_len": predictions[TARGET_COL].fillna("").astype(str).str.len(),
        }
    )
    return SelectionResult(predictions=predictions, audit=audit)


def consensus_replace(
    ids: list[str],
    sources: list[PredictionTable],
    *,
    min_consensus: int,
    max_length_delta: int | None = None,
    min_anchor_outlier_delta: int | None = None,
) -> SelectionResult:
    """Replace anchor only when non-anchor candidates strongly agree."""

    if len(sources) < 2:
        raise ValueError("Consensus replacement requires at least one candidate.")
    anchor = sources[0]
    candidates = sources[1:]
    output_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, object]] = []
    for row_idx, image_id in enumerate(ids):
        anchor_text = anchor.texts[row_idx]
        candidate_items = [
            (candidate.name, candidate.texts[row_idx])
            for candidate in candidates
            if candidate.texts[row_idx].strip()
        ]
        chosen_text = anchor_text
        chosen_source = anchor.name
        strategy = "anchor"
        top_text, top_count, top_sources = _top_consensus(candidate_items)
        candidate_lengths = [len(text) for _, text in candidate_items]
        candidate_median_len = median(candidate_lengths) if candidate_lengths else len(anchor_text)

        can_replace = top_text is not None and top_count >= min_consensus
        if can_replace and max_length_delta is not None:
            can_replace = abs(len(top_text) - len(anchor_text)) <= max_length_delta
        if can_replace and min_anchor_outlier_delta is not None:
            can_replace = (
                abs(len(anchor_text) - candidate_median_len) >= min_anchor_outlier_delta
            )
        if can_replace and top_text != anchor_text:
            chosen_text = top_text
            chosen_source = "+".join(top_sources)
            strategy = "consensus_replace"

        output_rows.append({ID_COL: image_id, TARGET_COL: chosen_text})
        audit_rows.append(
            _audit_row(
                image_id=image_id,
                anchor_text=anchor_text,
                chosen_text=chosen_text,
                chosen_source=chosen_source,
                strategy=strategy,
                top_text=top_text or "",
                top_count=top_count,
                candidate_count=len(candidate_items),
                candidate_median_len=float(candidate_median_len),
            )
        )
    return SelectionResult(
        predictions=pd.DataFrame(output_rows),
        audit=pd.DataFrame(audit_rows),
    )


def weighted_vote(
    ids: list[str],
    sources: list[PredictionTable],
    *,
    weights: dict[str, float],
    priority: list[str] | None = None,
) -> SelectionResult:
    """Select text by weighted exact-string vote."""

    if not sources:
        raise ValueError("At least one source is required.")
    priority = priority or [source.name for source in sources]
    priority_rank = {name: rank for rank, name in enumerate(priority)}
    anchor = sources[0]
    output_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, object]] = []
    for row_idx, image_id in enumerate(ids):
        scores: defaultdict[str, float] = defaultdict(float)
        text_sources: defaultdict[str, list[str]] = defaultdict(list)
        for source in sources:
            text = source.texts[row_idx]
            if not text.strip():
                continue
            scores[text] += weights.get(source.name, 1.0)
            text_sources[text].append(source.name)

        if not scores:
            chosen_text = anchor.texts[row_idx]
            chosen_sources = [anchor.name]
            top_score = 0.0
        else:
            chosen_text = min(
                scores,
                key=lambda text: (
                    -scores[text],
                    min(priority_rank.get(name, len(priority_rank)) for name in text_sources[text]),
                ),
            )
            chosen_sources = text_sources[chosen_text]
            top_score = scores[chosen_text]

        output_rows.append({ID_COL: image_id, TARGET_COL: chosen_text})
        audit_rows.append(
            _audit_row(
                image_id=image_id,
                anchor_text=anchor.texts[row_idx],
                chosen_text=chosen_text,
                chosen_source="+".join(chosen_sources),
                strategy="weighted_vote",
                top_text=chosen_text,
                top_count=top_score,
                candidate_count=len(sources),
                candidate_median_len=float(
                    median([len(source.texts[row_idx]) for source in sources])
                ),
            )
        )
    return SelectionResult(
        predictions=pd.DataFrame(output_rows),
        audit=pd.DataFrame(audit_rows),
    )


def _top_consensus(candidate_items: list[tuple[str, str]]) -> tuple[str | None, int, list[str]]:
    if not candidate_items:
        return None, 0, []
    counts = Counter(text for _, text in candidate_items)
    top_text, top_count = counts.most_common(1)[0]
    top_sources = [name for name, text in candidate_items if text == top_text]
    return top_text, top_count, top_sources


def _audit_row(
    *,
    image_id: str,
    anchor_text: str,
    chosen_text: str,
    chosen_source: str,
    strategy: str,
    top_text: str,
    top_count: int | float,
    candidate_count: int,
    candidate_median_len: float,
) -> dict[str, object]:
    return {
        ID_COL: image_id,
        "anchor_text": anchor_text,
        "chosen_text": chosen_text,
        "chosen_source": chosen_source,
        "strategy": strategy,
        "changed_from_anchor": chosen_text != anchor_text,
        "top_non_anchor_text": top_text,
        "top_non_anchor_count": top_count,
        "candidate_count": candidate_count,
        "anchor_len": len(anchor_text),
        "chosen_len": len(chosen_text),
        "candidate_median_len": candidate_median_len,
        "chosen_minus_anchor_len": len(chosen_text) - len(anchor_text),
    }

"""Create anchor-based selective ensemble prediction/submission CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import OUTPUTS_DIR, SAMPLE_SUBMISSION_CSV  # noqa: E402
from src.diagnostics.evaluator import evaluate_predictions  # noqa: E402
from src.ensemble.csv_ensemble import make_submission  # noqa: E402
from src.ensemble.selective import (  # noqa: E402
    align_prediction_sources,
    consensus_replace,
    keep_anchor,
    load_prediction_csv,
    parse_prediction_spec,
    weighted_vote,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor",
        required=True,
        help="Anchor prediction in format name:path.csv.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate prediction in format name:path.csv. Repeat for each source.",
    )
    parser.add_argument(
        "--strategy",
        choices=["anchor", "consensus", "length_guarded_consensus", "weighted_vote"],
        default="consensus",
    )
    parser.add_argument(
        "--min-consensus",
        type=int,
        default=3,
        help="Minimum non-anchor exact-string votes needed for consensus replacement.",
    )
    parser.add_argument(
        "--max-length-delta",
        type=int,
        default=None,
        help="Only replace anchor when chosen text length is within this delta.",
    )
    parser.add_argument(
        "--min-anchor-outlier-delta",
        type=int,
        default=4,
        help=(
            "For length_guarded_consensus, replace only when anchor length differs "
            "from candidate median by at least this many characters."
        ),
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        help="Weighted vote setting in format name=value. Repeat as needed.",
    )
    parser.add_argument(
        "--priority",
        nargs="*",
        default=None,
        help="Source priority for weighted-vote ties.",
    )
    parser.add_argument("--run-name", default="selective_ensemble")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--truth", type=Path, default=None)
    parser.add_argument("--evaluation-output-dir", type=Path, default=None)
    parser.add_argument("--sample-submission", type=Path, default=None)
    parser.add_argument("--submission-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anchor_name, anchor_path = parse_prediction_spec(args.anchor)
    anchor = load_prediction_csv(anchor_path, name=anchor_name)
    candidates = {}
    for spec in args.candidate:
        name, path = parse_prediction_spec(spec)
        candidates[name] = load_prediction_csv(path, name=name)

    ids, sources = align_prediction_sources(anchor_name, anchor, candidates)
    result = _select(args, anchor_name, anchor, ids, sources)

    output = args.output or args.output_dir / "predictions" / f"{args.run_name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(output, index=False)
    print(f"Saved predictions: {output}")

    audit_output = args.audit_output or args.output_dir / "predictions" / f"{args.run_name}_audit.csv"
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    result.audit.to_csv(audit_output, index=False)
    changed = int(result.audit["changed_from_anchor"].sum())
    print(f"Saved audit: {audit_output}")
    print(f"Changed rows from anchor: {changed}/{len(result.audit)}")

    if args.truth is not None:
        truth = pd.read_csv(args.truth)
        eval_dir = (
            args.evaluation_output_dir
            or PROJECT_ROOT / "reports" / "diagnostics" / "evaluation" / args.run_name
        )
        summary, _, _ = evaluate_predictions(
            truth,
            result.predictions,
            output_dir=eval_dir,
            allow_prediction_subset=True,
        )
        print(
            "Evaluation: "
            f"rows={summary.rows} "
            f"wer={summary.wer:.5f} "
            f"cer={summary.cer:.5f} "
            f"score={summary.score:.5f} "
            f"exact_match_rate={summary.exact_match_rate:.5f}"
        )

    sample_submission = args.sample_submission
    if sample_submission is None and SAMPLE_SUBMISSION_CSV.exists():
        sample_submission = SAMPLE_SUBMISSION_CSV
    if sample_submission is not None:
        sample = pd.read_csv(sample_submission)
        submission = make_submission(result.predictions, sample)
        submission_output = (
            args.submission_output
            or args.output_dir / "submissions" / f"{args.run_name}_submission.csv"
        )
        submission_output.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(submission_output, index=False)
        print(f"Saved submission: {submission_output}")


def _select(args: argparse.Namespace, anchor_name: str, anchor, ids, sources):
    if args.strategy == "anchor":
        return keep_anchor(anchor_name, anchor)
    if args.strategy == "consensus":
        return consensus_replace(
            ids,
            sources,
            min_consensus=args.min_consensus,
            max_length_delta=args.max_length_delta,
        )
    if args.strategy == "length_guarded_consensus":
        return consensus_replace(
            ids,
            sources,
            min_consensus=args.min_consensus,
            max_length_delta=args.max_length_delta,
            min_anchor_outlier_delta=args.min_anchor_outlier_delta,
        )
    if args.strategy == "weighted_vote":
        return weighted_vote(
            ids,
            sources,
            weights=_parse_weights(args.weight),
            priority=args.priority,
        )
    raise ValueError(f"Unsupported strategy: {args.strategy}")


def _parse_weights(raw_weights: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for raw_weight in raw_weights:
        if "=" not in raw_weight:
            raise ValueError("--weight must use format name=value")
        name, value = raw_weight.split("=", maxsplit=1)
        weights[name] = float(value)
    return weights


if __name__ == "__main__":
    main()


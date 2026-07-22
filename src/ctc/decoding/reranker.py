"""Conservative reranking utilities for CTC beam candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RerankConfig:
    """Penalties applied after beam search candidate generation."""

    short_text_penalty: float = 0.0
    min_chars: int = 0
    repeated_whitespace_penalty: float = 0.0
    repeated_punctuation_penalty: float = 0.0
    edge_space_penalty: float = 0.0


def rerank_text_score(text: str, base_score: float, config: RerankConfig) -> float:
    """Return an adjusted score for a decoded candidate."""

    score = float(base_score)
    if config.min_chars > 0 and len(text) < config.min_chars:
        score -= config.short_text_penalty * (config.min_chars - len(text))
    if config.repeated_whitespace_penalty and re.search(r"\s{2,}", text):
        score -= config.repeated_whitespace_penalty
    if config.repeated_punctuation_penalty and re.search(r"([.,:;!?\-])\1{1,}", text):
        score -= config.repeated_punctuation_penalty
    if config.edge_space_penalty and text != text.strip():
        score -= config.edge_space_penalty
    return score


def rerank_candidates(candidates, config: RerankConfig):
    """Return candidates sorted by conservative reranking score."""

    return sorted(
        candidates,
        key=lambda candidate: rerank_text_score(candidate.text, candidate.score, config),
        reverse=True,
    )

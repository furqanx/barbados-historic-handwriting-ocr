"""Configuration objects for CTC decoding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CTCDecoderConfig:
    """Runtime options for converting CTC logits into text."""

    decoder: str = "greedy"
    beam_size: int = 10
    top_tokens_per_step: int | None = 20
    lm_weight: float = 0.0
    length_bonus: float = 0.0
    candidates_top_k: int = 1
    rerank_short_text_penalty: float = 0.0
    rerank_min_chars: int = 0
    rerank_repeated_whitespace_penalty: float = 0.0
    rerank_repeated_punctuation_penalty: float = 0.0
    rerank_edge_space_penalty: float = 0.0

    def __post_init__(self) -> None:
        valid = {"greedy", "beam", "beam_lm", "beam_lm_rerank"}
        if self.decoder not in valid:
            raise ValueError(f"Unsupported CTC decoder: {self.decoder}")
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive.")
        if self.top_tokens_per_step is not None and self.top_tokens_per_step <= 0:
            raise ValueError("top_tokens_per_step must be positive or None.")
        if self.candidates_top_k <= 0:
            raise ValueError("candidates_top_k must be positive.")
        if self.rerank_min_chars < 0:
            raise ValueError("rerank_min_chars cannot be negative.")

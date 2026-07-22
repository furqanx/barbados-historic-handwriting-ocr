"""Prefix beam search for CTC logits."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from src.ctc.decoding.language_model import CharNGramLanguageModel
from src.ctc.tokenizer import CharacterTokenizer


NEG_INF = -1.0e30


@dataclass(frozen=True)
class BeamSearchConfig:
    """Options for CTC prefix beam search."""

    beam_size: int = 10
    top_tokens_per_step: int | None = 20
    lm_weight: float = 0.0
    length_bonus: float = 0.0
    candidates_top_k: int = 1

    def __post_init__(self) -> None:
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive.")
        if self.top_tokens_per_step is not None and self.top_tokens_per_step <= 0:
            raise ValueError("top_tokens_per_step must be positive or None.")
        if self.candidates_top_k <= 0:
            raise ValueError("candidates_top_k must be positive.")


@dataclass(frozen=True)
class CTCCandidate:
    """One decoded CTC candidate."""

    text: str
    token_ids: tuple[int, ...]
    acoustic_score: float
    lm_score: float
    score: float


@dataclass(frozen=True)
class _BeamState:
    blank_score: float = NEG_INF
    nonblank_score: float = NEG_INF
    lm_score: float = 0.0

    @property
    def acoustic_score(self) -> float:
        return _logaddexp(self.blank_score, self.nonblank_score)


def ctc_prefix_beam_search(
    logits: Tensor,
    tokenizer: CharacterTokenizer,
    *,
    config: BeamSearchConfig | None = None,
    language_model: CharNGramLanguageModel | None = None,
) -> list[CTCCandidate]:
    """Decode one CTC logit matrix shaped ``[time, vocab]``."""

    config = config or BeamSearchConfig()
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shaped [time, vocab], got {tuple(logits.shape)}")
    if logits.shape[-1] != tokenizer.vocab_size:
        raise ValueError(
            "Logit vocabulary size does not match tokenizer: "
            f"{logits.shape[-1]} != {tokenizer.vocab_size}"
        )

    log_probs = logits.float().log_softmax(dim=-1).detach().cpu()
    blank_id = tokenizer.blank_id
    id_to_char = tokenizer.id_to_char
    beams: dict[tuple[int, ...], _BeamState] = {
        (): _BeamState(blank_score=0.0, nonblank_score=NEG_INF, lm_score=0.0)
    }

    for timestep in range(log_probs.shape[0]):
        step = log_probs[timestep]
        token_ids = _candidate_token_ids(step, blank_id, config.top_tokens_per_step)
        next_beams: dict[tuple[int, ...], _BeamState] = {}

        for prefix, state in beams.items():
            total_score = state.acoustic_score
            for token_id in token_ids:
                logp = float(step[token_id].item())
                if token_id == blank_id:
                    current = next_beams.get(prefix, _BeamState(lm_score=state.lm_score))
                    next_beams[prefix] = _BeamState(
                        blank_score=_logaddexp(
                            current.blank_score,
                            total_score + logp,
                        ),
                        nonblank_score=current.nonblank_score,
                        lm_score=state.lm_score,
                    )
                    continue

                last_token_id = prefix[-1] if prefix else None
                if token_id == last_token_id:
                    current = next_beams.get(prefix, _BeamState(lm_score=state.lm_score))
                    next_beams[prefix] = _BeamState(
                        blank_score=current.blank_score,
                        nonblank_score=_logaddexp(
                            current.nonblank_score,
                            state.nonblank_score + logp,
                        ),
                        lm_score=state.lm_score,
                    )

                    extended = prefix + (token_id,)
                    extended_lm = _extend_lm_score(
                        prefix,
                        token_id,
                        tokenizer,
                        state.lm_score,
                        language_model,
                    )
                    current = next_beams.get(extended, _BeamState(lm_score=extended_lm))
                    next_beams[extended] = _BeamState(
                        blank_score=current.blank_score,
                        nonblank_score=_logaddexp(
                            current.nonblank_score,
                            state.blank_score + logp,
                        ),
                        lm_score=extended_lm,
                    )
                    continue

                extended = prefix + (token_id,)
                extended_lm = _extend_lm_score(
                    prefix,
                    token_id,
                    tokenizer,
                    state.lm_score,
                    language_model,
                )
                current = next_beams.get(extended, _BeamState(lm_score=extended_lm))
                next_beams[extended] = _BeamState(
                    blank_score=current.blank_score,
                    nonblank_score=_logaddexp(
                        current.nonblank_score,
                        total_score + logp,
                    ),
                    lm_score=extended_lm,
                )

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: _combined_score(item[0], item[1], config, tokenizer, language_model),
                reverse=True,
            )[: config.beam_size]
        )

    candidates = [
        _make_candidate(prefix, state, config, tokenizer, language_model)
        for prefix, state in beams.items()
    ]
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[
        : config.candidates_top_k
    ]


def ctc_prefix_beam_search_batch(
    logits: Tensor,
    input_lengths: Tensor,
    tokenizer: CharacterTokenizer,
    *,
    config: BeamSearchConfig | None = None,
    language_model: CharNGramLanguageModel | None = None,
) -> list[list[CTCCandidate]]:
    """Decode a CTC batch with logits shaped ``[time, batch, vocab]``."""

    if logits.ndim != 3:
        raise ValueError(f"Expected logits shaped [time, batch, vocab], got {tuple(logits.shape)}")
    lengths = input_lengths.detach().cpu().tolist()
    logit_batch = logits.detach().cpu()
    decoded: list[list[CTCCandidate]] = []
    for batch_idx, length in enumerate(lengths):
        decoded.append(
            ctc_prefix_beam_search(
                logit_batch[: int(length), batch_idx, :],
                tokenizer,
                config=config,
                language_model=language_model,
            )
        )
    return decoded


def _make_candidate(
    prefix: tuple[int, ...],
    state: _BeamState,
    config: BeamSearchConfig,
    tokenizer: CharacterTokenizer,
    language_model: CharNGramLanguageModel | None,
) -> CTCCandidate:
    text = "".join(tokenizer.id_to_char[idx] for idx in prefix)
    lm_score = state.lm_score
    if language_model is not None:
        lm_score += language_model.log_prob_end(text)
    acoustic_score = state.acoustic_score
    score = acoustic_score + config.lm_weight * lm_score + config.length_bonus * len(prefix)
    return CTCCandidate(
        text=text,
        token_ids=prefix,
        acoustic_score=acoustic_score,
        lm_score=lm_score,
        score=score,
    )


def _combined_score(
    prefix: tuple[int, ...],
    state: _BeamState,
    config: BeamSearchConfig,
    tokenizer: CharacterTokenizer,
    language_model: CharNGramLanguageModel | None,
) -> float:
    del tokenizer, language_model
    text_length = len(prefix)
    return (
        state.acoustic_score
        + config.lm_weight * state.lm_score
        + config.length_bonus * text_length
    )


def _extend_lm_score(
    prefix: tuple[int, ...],
    token_id: int,
    tokenizer: CharacterTokenizer,
    current_lm_score: float,
    language_model: CharNGramLanguageModel | None,
) -> float:
    if language_model is None:
        return current_lm_score
    text = "".join(tokenizer.id_to_char[idx] for idx in prefix)
    token = tokenizer.id_to_char[token_id]
    return current_lm_score + language_model.log_prob_next(text, token)


def _candidate_token_ids(
    log_probs: Tensor,
    blank_id: int,
    top_tokens_per_step: int | None,
) -> list[int]:
    if top_tokens_per_step is None or top_tokens_per_step >= log_probs.numel():
        token_ids = list(range(log_probs.numel()))
    else:
        token_ids = torch.topk(log_probs, k=top_tokens_per_step).indices.tolist()
    if blank_id not in token_ids:
        token_ids.append(blank_id)
    return [int(token_id) for token_id in token_ids]


def _logaddexp(a: float, b: float) -> float:
    if a <= NEG_INF / 2:
        return b
    if b <= NEG_INF / 2:
        return a
    high = max(a, b)
    low = min(a, b)
    return high + math.log1p(math.exp(low - high))

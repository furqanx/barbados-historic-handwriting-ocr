import torch

from src.ctc.decoding import (
    BeamSearchConfig,
    CTCDecoderConfig,
    CharNGramLanguageModel,
    RerankConfig,
    ctc_prefix_beam_search,
    rerank_candidates,
)
from src.ctc.predictor import decode_ctc_batch
from src.ctc.tokenizer import build_tokenizer_from_texts


def test_prefix_beam_search_decodes_simple_sequence() -> None:
    tokenizer = build_tokenizer_from_texts(["AB"])
    blank = tokenizer.blank_id
    a = tokenizer.char_to_id["A"]
    b = tokenizer.char_to_id["B"]
    logits = torch.full((4, tokenizer.vocab_size), -5.0)
    logits[0, a] = 5.0
    logits[1, blank] = 5.0
    logits[2, b] = 5.0
    logits[3, blank] = 5.0

    candidates = ctc_prefix_beam_search(
        logits,
        tokenizer,
        config=BeamSearchConfig(beam_size=3, top_tokens_per_step=3),
    )

    assert candidates[0].text == "AB"


def test_character_lm_can_bias_equal_acoustic_candidates() -> None:
    tokenizer = build_tokenizer_from_texts(["AB", "AC"])
    blank = tokenizer.blank_id
    a = tokenizer.char_to_id["A"]
    b = tokenizer.char_to_id["B"]
    c = tokenizer.char_to_id["C"]
    logits = torch.full((3, tokenizer.vocab_size), -8.0)
    logits[0, a] = 5.0
    logits[1, b] = 5.0
    logits[1, c] = 5.0
    logits[2, blank] = 5.0
    language_model = CharNGramLanguageModel.train(["AB"] * 20 + ["AC"], order=2)

    candidates = ctc_prefix_beam_search(
        logits,
        tokenizer,
        config=BeamSearchConfig(
            beam_size=5,
            top_tokens_per_step=4,
            lm_weight=2.0,
        ),
        language_model=language_model,
    )

    assert candidates[0].text == "AB"


def test_decode_ctc_batch_returns_candidate_payloads_for_beam() -> None:
    tokenizer = build_tokenizer_from_texts(["AB"])
    a = tokenizer.char_to_id["A"]
    b = tokenizer.char_to_id["B"]
    logits = torch.full((2, 1, tokenizer.vocab_size), -5.0)
    logits[0, 0, a] = 5.0
    logits[1, 0, b] = 5.0

    predictions, candidates = decode_ctc_batch(
        logits,
        torch.tensor([2]),
        tokenizer,
        decoder_config=CTCDecoderConfig(
            decoder="beam",
            beam_size=3,
            top_tokens_per_step=3,
            candidates_top_k=2,
        ),
    )

    assert predictions == ["AB"]
    assert candidates is not None
    assert "AB" in candidates[0]


def test_reranker_penalizes_too_short_candidate() -> None:
    tokenizer = build_tokenizer_from_texts(["AB"])
    a = tokenizer.char_to_id["A"]
    b = tokenizer.char_to_id["B"]
    logits = torch.full((2, tokenizer.vocab_size), -5.0)
    logits[0, a] = 5.0
    logits[1, b] = 5.0
    candidates = ctc_prefix_beam_search(
        logits,
        tokenizer,
        config=BeamSearchConfig(beam_size=3, top_tokens_per_step=3, candidates_top_k=3),
    )

    reranked = rerank_candidates(
        candidates,
        RerankConfig(short_text_penalty=10.0, min_chars=2),
    )

    assert reranked[0].text == "AB"

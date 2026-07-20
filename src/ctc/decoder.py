"""CTC decoding helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from src.ctc.tokenizer import CharacterTokenizer


def greedy_decode_ids(
    token_ids: Iterable[int],
    tokenizer: CharacterTokenizer,
) -> str:
    """Decode one CTC token-id sequence with greedy collapse."""

    return tokenizer.decode(token_ids, collapse_ctc=True)


def greedy_decode_batch(
    logits,
    input_lengths,
    tokenizer: CharacterTokenizer,
) -> list[str]:
    """Greedy-decode model logits shaped [time, batch, vocab]."""

    predictions = logits.argmax(dim=-1).transpose(0, 1).detach().cpu()
    lengths = input_lengths.detach().cpu().tolist()
    decoded: list[str] = []

    for token_ids, length in zip(predictions, lengths):
        decoded.append(
            greedy_decode_ids(
                token_ids[: int(length)].tolist(),
                tokenizer,
            )
        )

    return decoded


def decode_token_id_batches(
    batches: Sequence[Iterable[int]],
    tokenizer: CharacterTokenizer,
) -> list[str]:
    """Decode already-argmaxed CTC token-id sequences."""

    return [greedy_decode_ids(token_ids, tokenizer) for token_ids in batches]

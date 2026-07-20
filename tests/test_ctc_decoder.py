from src.ctc.tokenizer import build_tokenizer_from_texts
from src.ctc.decoder import decode_token_id_batches, greedy_decode_ids


def test_greedy_decode_ids_collapses_repeats_and_removes_blank() -> None:
    tokenizer = build_tokenizer_from_texts(["AB"])
    blank = tokenizer.blank_id
    a = tokenizer.char_to_id["A"]
    b = tokenizer.char_to_id["B"]

    assert greedy_decode_ids([blank, a, a, blank, b, b], tokenizer) == "AB"


def test_decode_token_id_batches() -> None:
    tokenizer = build_tokenizer_from_texts(["AB"])
    a = tokenizer.char_to_id["A"]
    b = tokenizer.char_to_id["B"]

    assert decode_token_id_batches([[a, b], [b, a]], tokenizer) == ["AB", "BA"]

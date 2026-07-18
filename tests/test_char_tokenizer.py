import pytest

from src.data.char_tokenizer import CharacterTokenizer, build_tokenizer_from_texts


def test_build_tokenizer_reserves_blank_id_zero() -> None:
    tokenizer = build_tokenizer_from_texts(["BA", "A B"])

    assert tokenizer.blank_id == 0
    assert tokenizer.vocab_size == 4
    assert tokenizer.char_to_id[" "] == 1
    assert tokenizer.char_to_id["A"] == 2
    assert tokenizer.char_to_id["B"] == 3


def test_encode_normalizes_whitespace_by_default() -> None:
    tokenizer = build_tokenizer_from_texts(["A B"])

    assert tokenizer.encode("  A   B  ") == [
        tokenizer.char_to_id["A"],
        tokenizer.char_to_id[" "],
        tokenizer.char_to_id["B"],
    ]


def test_decode_collapses_ctc_repeats_and_blanks() -> None:
    tokenizer = build_tokenizer_from_texts(["AB"])
    blank = tokenizer.blank_id
    a = tokenizer.char_to_id["A"]
    b = tokenizer.char_to_id["B"]

    assert tokenizer.decode([blank, a, a, blank, b, b]) == "AB"


def test_decode_can_skip_ctc_collapse() -> None:
    tokenizer = build_tokenizer_from_texts(["AB"])
    a = tokenizer.char_to_id["A"]

    assert tokenizer.decode([a, a], collapse_ctc=False) == "AA"


def test_unknown_character_raises() -> None:
    tokenizer = build_tokenizer_from_texts(["AB"])

    with pytest.raises(ValueError, match="unknown characters"):
        tokenizer.encode("AC")


def test_save_and_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "char_vocab.json"
    tokenizer = build_tokenizer_from_texts(["W^m Olivant"])
    tokenizer.save(path)

    loaded = CharacterTokenizer.load(path)

    assert loaded.char_to_id == tokenizer.char_to_id
    assert loaded.decode(tokenizer.encode("W^m Olivant")) == "W^m Olivant"

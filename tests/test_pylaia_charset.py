from pathlib import Path

from src.htr.pylaia.charset import (
    PYLAIA_CTC_TOKEN,
    PYLAIA_SPACE_TOKEN,
    characters_to_symbols,
    detokenize_text,
    load_syms,
    tokenize_text,
    write_syms,
)


def test_tokenize_and_detokenize_text() -> None:
    tokenized = tokenize_text("By this")

    assert tokenized == f"B y {PYLAIA_SPACE_TOKEN} t h i s"
    assert detokenize_text(tokenized) == "By this"


def test_symbols_start_with_ctc_and_end_with_space(tmp_path: Path) -> None:
    symbols = characters_to_symbols(["B", "y", " "])
    output = write_syms(symbols, tmp_path / "syms.txt")

    loaded = load_syms(output)

    assert symbols[0] == PYLAIA_CTC_TOKEN
    assert symbols[-1] == PYLAIA_SPACE_TOKEN
    assert loaded[PYLAIA_CTC_TOKEN] == 0
    assert loaded[PYLAIA_SPACE_TOKEN] == len(symbols) - 1


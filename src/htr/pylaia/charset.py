"""Character and token helpers for PyLaia HTR workflows."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.common.text_normalization import normalize_text


PYLAIA_CTC_TOKEN = "<ctc>"
PYLAIA_SPACE_TOKEN = "<space>"
PYLAIA_UNKNOWN_TOKEN = "<unk>"


@dataclass(frozen=True)
class SymbolAudit:
    """Summary of target characters not covered by a PyLaia symbol file."""

    missing_characters: tuple[str, ...]
    covered_characters: tuple[str, ...]

    @property
    def has_missing(self) -> bool:
        """Whether at least one target character is not present in the symbols."""

        return bool(self.missing_characters)


def tokenize_text(text: object, *, space_token: str = PYLAIA_SPACE_TOKEN) -> str:
    """Convert plain text into PyLaia character tokens."""

    value = normalize_text(text)
    return " ".join(space_token if char == " " else char for char in value)


def detokenize_text(
    tokenized_text: str,
    *,
    space_token: str = PYLAIA_SPACE_TOKEN,
    unknown_token: str = PYLAIA_UNKNOWN_TOKEN,
    keep_unknown: bool = False,
) -> str:
    """Convert PyLaia tokenized text back to a plain transcription."""

    chars: list[str] = []
    for token in tokenized_text.split():
        if token == space_token:
            chars.append(" ")
        elif token == unknown_token and not keep_unknown:
            continue
        else:
            chars.append(token)
    return normalize_text("".join(chars))


def collect_characters(
    texts: Iterable[object],
    *,
    include_space: bool = True,
) -> list[str]:
    """Collect a deterministic sorted list of characters from text values."""

    characters = {
        char
        for text in texts
        for char in normalize_text(text)
        if include_space or char != " "
    }
    return sorted(characters)


def characters_to_symbols(
    characters: Iterable[str],
    *,
    ctc_token: str = PYLAIA_CTC_TOKEN,
    space_token: str = PYLAIA_SPACE_TOKEN,
    include_unknown: bool = True,
    unknown_token: str = PYLAIA_UNKNOWN_TOKEN,
) -> list[str]:
    """Build PyLaia symbols from characters, reserving index 0 for CTC."""

    normalized = sorted({char for char in characters if char != " "})
    reserved = {ctc_token, space_token, unknown_token}
    collisions = reserved & set(normalized)
    if collisions:
        raise ValueError(f"Reserved PyLaia tokens appear as real characters: {collisions}")

    symbols = [ctc_token, *normalized]
    if include_unknown:
        symbols.append(unknown_token)
    symbols.append(space_token)
    return symbols


def write_syms(symbols: Iterable[str], output_path: str | Path) -> Path:
    """Write a PyLaia `syms.txt` file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{symbol} {idx}" for idx, symbol in enumerate(symbols)]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def load_syms(path: str | Path) -> dict[str, int]:
    """Load a PyLaia `syms.txt` file."""

    symbols: dict[str, int] = {}
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            symbol, raw_idx = line.rsplit(" ", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid syms line {line_number}: {raw_line!r}") from exc
        symbols[symbol] = int(raw_idx)
    return symbols


def audit_symbols(
    texts: Iterable[object],
    symbols: Iterable[str],
    *,
    space_token: str = PYLAIA_SPACE_TOKEN,
) -> SymbolAudit:
    """Compare target characters against an existing PyLaia symbol set."""

    symbol_set = set(symbols)
    target_characters = collect_characters(texts)
    missing: list[str] = []
    covered: list[str] = []
    for char in target_characters:
        symbol = space_token if char == " " else char
        if symbol in symbol_set:
            covered.append(char)
        else:
            missing.append(char)
    return SymbolAudit(
        missing_characters=tuple(missing),
        covered_characters=tuple(covered),
    )


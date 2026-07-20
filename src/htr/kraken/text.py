"""Text normalization helpers for Kraken HTR workflows."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from src.common.text_normalization import normalize_text


UNICODE_FORMS = ("preserve", "NFC", "NFKC", "NFD", "NFKD")


@dataclass(frozen=True)
class CharacterAudit:
    """Summary of characters before and after Kraken text normalization."""

    unicode_form: str
    character_count: int
    characters: tuple[str, ...]


def normalize_for_kraken(text: object, *, unicode_form: str = "preserve") -> str:
    """Normalize text for Kraken training or submission postprocessing."""

    value = normalize_text(text)
    if unicode_form not in UNICODE_FORMS:
        raise ValueError(f"Unsupported unicode normalization: {unicode_form}")
    if unicode_form == "preserve":
        return value
    return unicodedata.normalize(unicode_form, value)


def normalize_texts_for_kraken(
    texts: Iterable[object],
    *,
    unicode_form: str = "preserve",
) -> list[str]:
    """Normalize a sequence of text values for Kraken."""

    return [normalize_for_kraken(text, unicode_form=unicode_form) for text in texts]


def audit_characters(
    texts: Iterable[object],
    *,
    unicode_form: str = "preserve",
) -> CharacterAudit:
    """Collect target characters after a Kraken normalization choice."""

    characters = sorted(
        {
            char
            for text in normalize_texts_for_kraken(texts, unicode_form=unicode_form)
            for char in text
        }
    )
    return CharacterAudit(
        unicode_form=unicode_form,
        character_count=len(characters),
        characters=tuple(characters),
    )


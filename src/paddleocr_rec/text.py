"""Text helpers for PaddleOCR recognition training."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from src.features.text_normalization import normalize_text


@dataclass(frozen=True)
class PaddleOCRCharacterAudit:
    """Summary of the PaddleOCR recognition character dictionary."""

    character_count: int
    has_space: bool
    characters: tuple[str, ...]


def collect_paddleocr_characters(texts: Iterable[object]) -> list[str]:
    """Collect characters for PaddleOCR `character_dict_path`.

    PaddleOCR handles spaces through `Global.use_space_char=True`, so the
    dictionary file should contain non-space characters only.
    """

    return sorted(
        {
            char
            for text in texts
            for char in normalize_text(text)
            if char != " "
        }
    )


def write_character_dict(
    characters: Iterable[str],
    output_path,
) -> None:
    """Write one PaddleOCR recognition character per line."""

    from pathlib import Path

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(characters) + "\n", encoding="utf-8")


def audit_paddleocr_characters(texts: Iterable[object]) -> PaddleOCRCharacterAudit:
    """Audit target characters for PaddleOCR recognition."""

    raw_characters = sorted({char for text in texts for char in normalize_text(text)})
    return PaddleOCRCharacterAudit(
        character_count=len(raw_characters),
        has_space=" " in raw_characters,
        characters=tuple(raw_characters),
    )


"""Character vocabulary and tokenizer utilities for CTC OCR models."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.constants import TARGET_COL
from src.features.text_normalization import normalize_text


DEFAULT_BLANK_TOKEN = "<blank>"


@dataclass(frozen=True)
class CharacterTokenizer:
    """Character-level tokenizer for CTC training and decoding."""

    char_to_id: dict[str, int]
    blank_token: str = DEFAULT_BLANK_TOKEN

    def __post_init__(self) -> None:
        if self.blank_token not in self.char_to_id:
            raise ValueError(f"blank_token is missing from vocabulary: {self.blank_token}")
        if self.blank_id != 0:
            raise ValueError("CTC blank token must have id 0.")
        if len(set(self.char_to_id.values())) != len(self.char_to_id):
            raise ValueError("Vocabulary ids must be unique.")

    @property
    def blank_id(self) -> int:
        """ID reserved for the CTC blank token."""

        return self.char_to_id[self.blank_token]

    @property
    def id_to_char(self) -> dict[int, str]:
        """Reverse vocabulary mapping."""

        return {idx: char for char, idx in self.char_to_id.items()}

    @property
    def vocab_size(self) -> int:
        """Total number of output classes, including CTC blank."""

        return len(self.char_to_id)

    @property
    def characters(self) -> list[str]:
        """Characters excluding the CTC blank token."""

        return [
            char
            for char, idx in sorted(self.char_to_id.items(), key=lambda item: item[1])
            if char != self.blank_token
        ]

    def encode(self, text: object, *, normalize: bool = True) -> list[int]:
        """Encode a text string into character ids."""

        value = normalize_text(text) if normalize else _stringify_text(text)
        unknown = sorted({char for char in value if char not in self.char_to_id})
        if unknown:
            raise ValueError(f"Text contains unknown characters: {unknown}")
        return [self.char_to_id[char] for char in value]

    def decode(self, ids: Iterable[int], *, collapse_ctc: bool = True) -> str:
        """Decode ids into text, optionally applying greedy CTC collapse."""

        id_to_char = self.id_to_char
        chars: list[str] = []
        previous_id: int | None = None

        for raw_id in ids:
            idx = int(raw_id)
            if idx not in id_to_char:
                raise ValueError(f"Unknown token id: {idx}")

            if collapse_ctc:
                if idx == self.blank_id:
                    previous_id = idx
                    continue
                if idx == previous_id:
                    previous_id = idx
                    continue

            chars.append(id_to_char[idx])
            previous_id = idx

        return "".join(chars)

    def to_dict(self) -> dict[str, Any]:
        """Serialize tokenizer metadata."""

        return {
            "blank_token": self.blank_token,
            "blank_id": self.blank_id,
            "vocab_size": self.vocab_size,
            "char_to_id": self.char_to_id,
            "id_to_char": {str(idx): char for idx, char in self.id_to_char.items()},
        }

    def save(self, path: str | Path) -> Path:
        """Save tokenizer to JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CharacterTokenizer":
        """Create tokenizer from serialized metadata."""

        return cls(
            char_to_id={str(char): int(idx) for char, idx in payload["char_to_id"].items()},
            blank_token=str(payload.get("blank_token", DEFAULT_BLANK_TOKEN)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        """Load tokenizer from JSON."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def build_char_to_id(
    texts: Iterable[object],
    *,
    blank_token: str = DEFAULT_BLANK_TOKEN,
    normalize: bool = True,
) -> dict[str, int]:
    """Build a deterministic CTC vocabulary from text values."""

    normalized_texts = [
        normalize_text(text) if normalize else _stringify_text(text) for text in texts
    ]
    characters = sorted({char for text in normalized_texts for char in text})
    if blank_token in characters:
        raise ValueError(f"blank_token collides with a real character: {blank_token}")
    return {blank_token: 0, **{char: idx for idx, char in enumerate(characters, start=1)}}


def build_tokenizer_from_texts(
    texts: Iterable[object],
    *,
    blank_token: str = DEFAULT_BLANK_TOKEN,
    normalize: bool = True,
) -> CharacterTokenizer:
    """Build a character tokenizer from target texts."""

    return CharacterTokenizer(
        char_to_id=build_char_to_id(
            texts,
            blank_token=blank_token,
            normalize=normalize,
        ),
        blank_token=blank_token,
    )


def build_tokenizer_from_train_csv(
    train_csv: str | Path,
    *,
    target_col: str = TARGET_COL,
    blank_token: str = DEFAULT_BLANK_TOKEN,
    normalize: bool = True,
) -> CharacterTokenizer:
    """Build a character tokenizer from the competition Train.csv."""

    train = pd.read_csv(train_csv, encoding="utf-8-sig")
    if target_col not in train.columns:
        raise ValueError(f"Target column not found in train CSV: {target_col}")
    return build_tokenizer_from_texts(
        train[target_col].tolist(),
        blank_token=blank_token,
        normalize=normalize,
    )


def _stringify_text(text: object) -> str:
    """Convert nullable text values into strings."""

    return "" if text is None else str(text)

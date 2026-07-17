"""Text normalization helpers for OCR targets and predictions.

Default behavior is intentionally conservative: preserve case, punctuation,
digits, and historical transcription symbols while cleaning accidental
whitespace.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TextNormalizer:
    """Configurable text normalizer for OCR workflows."""

    strip: bool = True
    collapse_whitespace: bool = True
    lowercase: bool = False
    remove_punctuation: bool = False

    def __call__(self, text: object) -> str:
        value = "" if text is None else str(text)
        if self.collapse_whitespace:
            value = WHITESPACE_RE.sub(" ", value)
        if self.strip:
            value = value.strip()
        if self.lowercase:
            value = value.lower()
        if self.remove_punctuation:
            value = re.sub(r"[^A-Za-z0-9\s]", "", value)
            if self.collapse_whitespace:
                value = WHITESPACE_RE.sub(" ", value)
            if self.strip:
                value = value.strip()
        return value


PRESERVE_TEXT_NORMALIZER = TextNormalizer(
    strip=True,
    collapse_whitespace=True,
    lowercase=False,
    remove_punctuation=False,
)


def normalize_text(text: object) -> str:
    """Safely normalize text for metric comparison and baseline cleanup."""

    return PRESERVE_TEXT_NORMALIZER(text)


def normalize_texts(
    texts: Iterable[object],
    normalizer: TextNormalizer = PRESERVE_TEXT_NORMALIZER,
) -> list[str]:
    """Normalize a sequence of texts."""

    return [normalizer(text) for text in texts]


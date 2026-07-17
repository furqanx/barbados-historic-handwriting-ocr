"""Local evaluation metrics for handwriting transcription.

The competition metric combines WER and CER. Computing these over the full
corpus naturally weights examples by their reference word/character lengths,
which matches the intended behavior better than averaging per-row errors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from jiwer import cer, wer


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TranscriptionScore:
    """Container for OCR metric components."""

    wer: float
    cer: float

    @property
    def score(self) -> float:
        """Competition-style combined score. Lower is better."""

        return 0.5 * self.wer + 0.5 * self.cer


def normalize_whitespace(text: str) -> str:
    """Normalize repeated whitespace without changing case or punctuation."""

    return _WHITESPACE_RE.sub(" ", str(text)).strip()


def prepare_texts(texts: Iterable[str], *, normalize: bool = True) -> list[str]:
    """Convert an iterable of text values into metric-ready strings."""

    prepared = ["" if text is None else str(text) for text in texts]
    if normalize:
        prepared = [normalize_whitespace(text) for text in prepared]
    return prepared


def score_transcriptions(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    *,
    normalize: bool = True,
) -> TranscriptionScore:
    """Compute corpus-level WER, CER, and combined score."""

    references = prepare_texts(y_true, normalize=normalize)
    predictions = prepare_texts(y_pred, normalize=normalize)

    if len(references) != len(predictions):
        raise ValueError(
            "y_true and y_pred must have the same length: "
            f"{len(references)} != {len(predictions)}"
        )

    return TranscriptionScore(
        wer=wer(references, predictions),
        cer=cer(references, predictions),
    )


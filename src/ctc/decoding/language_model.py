"""Lightweight character n-gram language model for CTC beam search."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.text_normalization import normalize_text


START_TOKEN = "<s>"
END_TOKEN = "</s>"


@dataclass(frozen=True)
class CharNGramLanguageModel:
    """Add-k smoothed character n-gram language model."""

    order: int
    vocab: tuple[str, ...]
    context_counts: dict[tuple[str, ...], dict[str, int]]
    add_k: float = 0.5

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("order must be at least 1.")
        if self.add_k <= 0:
            raise ValueError("add_k must be positive.")
        if END_TOKEN not in self.vocab:
            raise ValueError(f"vocab must include {END_TOKEN!r}.")

    @classmethod
    def train(
        cls,
        texts: Iterable[object],
        *,
        order: int = 4,
        add_k: float = 0.5,
        extra_characters: Iterable[str] | None = None,
        normalize: bool = True,
    ) -> "CharNGramLanguageModel":
        """Fit a character n-gram LM from training transcriptions."""

        prepared = [
            normalize_text(text) if normalize else ("" if text is None else str(text))
            for text in texts
        ]
        characters = sorted({char for text in prepared for char in text})
        if extra_characters is not None:
            characters = sorted(set(characters).union(str(char) for char in extra_characters))
        vocab = tuple(characters + [END_TOKEN])

        counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        prefix_pad = [START_TOKEN] * max(order - 1, 0)
        for text in prepared:
            sequence = prefix_pad + list(text) + [END_TOKEN]
            for idx in range(order - 1, len(sequence)):
                context = tuple(sequence[idx - order + 1 : idx]) if order > 1 else ()
                token = sequence[idx]
                counts[context][token] += 1

        return cls(
            order=order,
            vocab=vocab,
            context_counts={context: dict(counter) for context, counter in counts.items()},
            add_k=add_k,
        )

    def log_prob_next(self, prefix: str, token: str) -> float:
        """Return log P(token | prefix)."""

        if token not in self.vocab:
            return math.log(self.add_k / (self.add_k * len(self.vocab)))

        context = self._context(prefix)
        counter = self.context_counts.get(context)
        if counter is None and self.order > 1:
            counter = self.context_counts.get(())
        if counter is None:
            counter = {}

        total = sum(counter.values())
        numerator = counter.get(token, 0) + self.add_k
        denominator = total + self.add_k * len(self.vocab)
        return math.log(numerator / denominator)

    def log_prob_end(self, prefix: str) -> float:
        """Return log probability of ending after prefix."""

        return self.log_prob_next(prefix, END_TOKEN)

    def save(self, path: str | Path) -> Path:
        """Write the language model to JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output

    def to_dict(self) -> dict[str, Any]:
        """Serialize model data."""

        return {
            "order": self.order,
            "vocab": list(self.vocab),
            "add_k": self.add_k,
            "context_counts": {
                _encode_context(context): counts
                for context, counts in self.context_counts.items()
            },
        }

    @classmethod
    def load(cls, path: str | Path) -> "CharNGramLanguageModel":
        """Load a model saved by :meth:`save`."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CharNGramLanguageModel":
        """Create model from a serialized payload."""

        return cls(
            order=int(payload["order"]),
            vocab=tuple(str(token) for token in payload["vocab"]),
            add_k=float(payload.get("add_k", 0.5)),
            context_counts={
                _decode_context(key): {str(token): int(count) for token, count in counts.items()}
                for key, counts in payload["context_counts"].items()
            },
        )

    def _context(self, prefix: str) -> tuple[str, ...]:
        if self.order == 1:
            return ()
        padded = [START_TOKEN] * (self.order - 1) + list(prefix)
        return tuple(padded[-self.order + 1 :])


def _encode_context(context: tuple[str, ...]) -> str:
    return json.dumps(list(context), ensure_ascii=False, separators=(",", ":"))


def _decode_context(payload: str) -> tuple[str, ...]:
    return tuple(json.loads(payload))

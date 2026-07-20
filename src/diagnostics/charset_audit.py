"""Character vocabulary and tokenizer integrity diagnostics."""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.constants import ID_COL, TARGET_COL
from src.data.char_tokenizer import CharacterTokenizer
from src.features.text_normalization import normalize_text


@dataclass(frozen=True)
class CharsetAuditResult:
    """Paths written by a charset audit run."""

    character_frequency: Path
    tokenizer_roundtrip: Path | None
    suspicious_lines: Path


def character_frequency_table(texts: list[object], *, normalize: bool = True) -> pd.DataFrame:
    """Count characters and attach Unicode metadata."""

    counter: Counter[str] = Counter()
    line_counter: Counter[str] = Counter()
    for raw in texts:
        text = normalize_text(raw) if normalize else ("" if raw is None else str(raw))
        seen = set(text)
        counter.update(text)
        line_counter.update(seen)

    rows = []
    for char, count in counter.items():
        rows.append(
            {
                "char": char,
                "repr": repr(char),
                "count": count,
                "line_count": line_counter[char],
                "unicode_name": unicodedata.name(char, "UNKNOWN"),
                "category": unicodedata.category(char),
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "char"]).reset_index(drop=True)


def tokenizer_roundtrip_table(
    df: pd.DataFrame,
    tokenizer: CharacterTokenizer,
    *,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Check encode/decode round-trip for every target line."""

    rows = []
    for row in df[[ID_COL, target_col]].itertuples(index=False):
        image_id = str(getattr(row, ID_COL))
        text = normalize_text(getattr(row, target_col))
        try:
            ids = tokenizer.encode(text, normalize=False)
            decoded = tokenizer.decode(ids, collapse_ctc=False)
            error = ""
        except Exception as exc:  # noqa: BLE001 - diagnostic report should capture all failures
            ids = []
            decoded = ""
            error = str(exc)
        rows.append(
            {
                ID_COL: image_id,
                "target": text,
                "encoded_length": len(ids),
                "decoded": decoded,
                "roundtrip_ok": decoded == text and not error,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def suspicious_character_lines(
    df: pd.DataFrame,
    *,
    rare_chars: set[str] | None = None,
    min_char_count: int = 10,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Find lines containing rare or unusual characters."""

    texts = [normalize_text(text) for text in df[target_col].tolist()]
    counts = Counter(char for text in texts for char in text)
    if rare_chars is None:
        rare_chars = {char for char, count in counts.items() if count <= min_char_count}

    rows = []
    for row in df[[ID_COL, target_col]].itertuples(index=False):
        text = normalize_text(getattr(row, target_col))
        present = sorted(set(text) & rare_chars)
        if present:
            rows.append(
                {
                    ID_COL: str(getattr(row, ID_COL)),
                    target_col: text,
                    "rare_chars": "".join(present),
                    "rare_char_count": sum(text.count(char) for char in present),
                    "char_len": len(text),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["rare_char_count", "char_len"],
        ascending=[False, False],
    )


def run_charset_audit(
    train: pd.DataFrame,
    *,
    output_dir: str | Path,
    tokenizer: CharacterTokenizer | None = None,
    target_col: str = TARGET_COL,
    min_char_count: int = 10,
) -> CharsetAuditResult:
    """Write character frequency and tokenizer integrity reports."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frequency = character_frequency_table(train[target_col].tolist())
    frequency_path = output / "character_frequency.csv"
    frequency.to_csv(frequency_path, index=False)

    roundtrip_path = None
    if tokenizer is not None:
        roundtrip = tokenizer_roundtrip_table(train, tokenizer, target_col=target_col)
        roundtrip_path = output / "tokenizer_roundtrip.csv"
        roundtrip.to_csv(roundtrip_path, index=False)

    suspicious = suspicious_character_lines(
        train,
        min_char_count=min_char_count,
        target_col=target_col,
    )
    suspicious_path = output / "rare_character_lines.csv"
    suspicious.to_csv(suspicious_path, index=False)

    return CharsetAuditResult(
        character_frequency=frequency_path,
        tokenizer_roundtrip=roundtrip_path,
        suspicious_lines=suspicious_path,
    )


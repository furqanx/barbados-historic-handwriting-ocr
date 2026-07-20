"""TrOCR tokenization, generation-length, and image canvas diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from src.constants import ID_COL, TARGET_COL
from src.diagnostics.ctc_alignment import resized_width
from src.features.text_normalization import normalize_text


@dataclass(frozen=True)
class TrOCRTargetAuditConfig:
    """Configuration for TrOCR target and image diagnostics."""

    model_name: str = "microsoft/trocr-small-handwritten"
    preprocess_mode: str = "default"
    target_height: int = 384
    canvas_width: int = 1536
    max_label_length: int = 192
    max_generation_length: int = 192


def load_trocr_tokenizer(model_name: str) -> object:
    """Load a TrOCR tokenizer without fast-tokenizer conversion surprises."""

    try:
        from transformers import AutoTokenizer, XLMRobertaTokenizer
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise ImportError("Install transformers to run TrOCR diagnostics.") from exc
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except ValueError:
        # Some Transformers builds still route TrOCR through a fast-tokenizer
        # backend even with use_fast=False. The TrOCR handwritten checkpoints
        # ship a SentencePiece BPE model, so this explicit fallback keeps token
        # length diagnostics aligned with the decoder tokenizer.
        return XLMRobertaTokenizer.from_pretrained(model_name, use_fast=False)


def trocr_target_audit_table(
    manifest: pd.DataFrame,
    *,
    tokenizer: object,
    config: TrOCRTargetAuditConfig,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Audit target token length and aspect-aware preprocessing risk."""

    rows = []
    for row in manifest.itertuples(index=False):
        image_id = str(getattr(row, ID_COL))
        text = normalize_text(getattr(row, target_col))
        token_ids = tokenizer(text, add_special_tokens=True).input_ids
        token_count = len(token_ids)
        image_path = Path(getattr(row, "image_path"))
        original_width, original_height = _get_image_size(row, image_path)
        aspect_resized_width = resized_width(
            original_width,
            original_height,
            target_height=config.target_height,
            max_width=config.canvas_width,
        )
        unclipped_width = resized_width(
            original_width,
            original_height,
            target_height=config.target_height,
            max_width=None,
        )
        rows.append(
            {
                ID_COL: image_id,
                "token_count": token_count,
                "char_len": len(text),
                "word_len": len(text.split()),
                "exceeds_max_label_length": token_count > config.max_label_length,
                "exceeds_max_generation_length": token_count > config.max_generation_length,
                "original_width": original_width,
                "original_height": original_height,
                "aspect_resized_width": aspect_resized_width,
                "unclipped_aspect_resized_width": unclipped_width,
                "would_clip_aspect_canvas": aspect_resized_width < unclipped_width,
                "pixels_per_char": aspect_resized_width / max(len(text), 1),
                target_col: text,
            }
        )
    return pd.DataFrame(rows)


def summarize_trocr_audit(table: pd.DataFrame) -> dict[str, object]:
    """Summarize TrOCR tokenization and preprocessing risks."""

    return {
        "rows": int(len(table)),
        "max_token_count": int(table["token_count"].max()) if len(table) else None,
        "p95_token_count": float(table["token_count"].quantile(0.95)) if len(table) else None,
        "exceeds_max_label_length_rows": int(table["exceeds_max_label_length"].sum()),
        "exceeds_max_generation_length_rows": int(
            table["exceeds_max_generation_length"].sum()
        ),
        "aspect_canvas_clipped_rows": int(table["would_clip_aspect_canvas"].sum()),
        "min_pixels_per_char": float(table["pixels_per_char"].min()) if len(table) else None,
        "median_pixels_per_char": float(table["pixels_per_char"].median()) if len(table) else None,
    }


def _get_image_size(row: object, image_path: Path) -> tuple[int, int]:
    if hasattr(row, "width") and hasattr(row, "height"):
        width = getattr(row, "width")
        height = getattr(row, "height")
        if pd.notna(width) and pd.notna(height):
            return int(width), int(height)
    with Image.open(image_path) as image:
        return image.size

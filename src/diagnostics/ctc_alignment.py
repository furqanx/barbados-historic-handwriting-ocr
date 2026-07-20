"""CTC sequence-length and image geometry diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from src.constants import ID_COL, TARGET_COL
from src.common.text_normalization import normalize_text


@dataclass(frozen=True)
class CTCAlignmentConfig:
    """Geometry configuration used by CTC image preprocessing."""

    target_height: int = 96
    max_width: int | None = 2048
    time_downsample_factor: int = 4


def adjacent_repeat_count(text: str) -> int:
    """Count adjacent repeated characters that require extra CTC separation."""

    return sum(1 for left, right in zip(text, text[1:]) if left == right)


def resized_width(
    original_width: int,
    original_height: int,
    *,
    target_height: int,
    max_width: int | None,
) -> int:
    """Compute width after keep-aspect resizing."""

    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"Invalid image size: {(original_width, original_height)}")
    width = max(1, int(round(original_width * (target_height / original_height))))
    if max_width is not None:
        width = min(width, max_width)
    return width


def ctc_alignment_table(
    manifest: pd.DataFrame,
    *,
    config: CTCAlignmentConfig,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Audit whether each line has enough CTC time steps."""

    rows = []
    for row in manifest.itertuples(index=False):
        image_id = str(getattr(row, ID_COL))
        image_path = Path(getattr(row, "image_path"))
        original_width, original_height = _get_image_size(row, image_path)
        width_after_resize = resized_width(
            original_width,
            original_height,
            target_height=config.target_height,
            max_width=config.max_width,
        )
        text = normalize_text(getattr(row, target_col)) if hasattr(row, target_col) else ""
        target_length = len(text)
        repeats = adjacent_repeat_count(text)
        required_steps = target_length + repeats
        encoder_time_steps = max(1, width_after_resize // config.time_downsample_factor)
        unclipped_width = resized_width(
            original_width,
            original_height,
            target_height=config.target_height,
            max_width=None,
        )
        rows.append(
            {
                ID_COL: image_id,
                "image_path": str(image_path),
                "original_width": original_width,
                "original_height": original_height,
                "resized_width": width_after_resize,
                "unclipped_resized_width": unclipped_width,
                "was_width_clipped": width_after_resize < unclipped_width,
                "target_length": target_length,
                "adjacent_repeat_count": repeats,
                "required_ctc_steps": required_steps,
                "encoder_time_steps": encoder_time_steps,
                "ctc_margin": encoder_time_steps - required_steps,
                "is_ctc_length_valid": encoder_time_steps >= required_steps,
                "pixels_per_char": width_after_resize / max(target_length, 1),
                target_col: text,
            }
        )
    return pd.DataFrame(rows)


def summarize_ctc_alignment(table: pd.DataFrame) -> dict[str, object]:
    """Summarize CTC alignment risks."""

    return {
        "rows": int(len(table)),
        "invalid_rows": int((~table["is_ctc_length_valid"]).sum()),
        "width_clipped_rows": int(table["was_width_clipped"].sum()),
        "min_ctc_margin": int(table["ctc_margin"].min()) if len(table) else None,
        "median_ctc_margin": float(table["ctc_margin"].median()) if len(table) else None,
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


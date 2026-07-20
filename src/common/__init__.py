"""Shared utilities used across OCR/HTR workflows."""

from src.common.image_preprocessing import (
    ResizeKeepAspectConfig,
    ResizeKeepAspectTransform,
    ResizePadConfig,
    ResizePadTransform,
    autocontrast,
    pad_to_width,
    resize_keep_aspect,
)
from src.common.text_normalization import (
    PRESERVE_TEXT_NORMALIZER,
    TextNormalizer,
    normalize_text,
    normalize_texts,
)
from src.common.torch_utils import (
    maybe_wrap_data_parallel,
    should_use_data_parallel,
    unwrap_model,
)

__all__ = [
    "PRESERVE_TEXT_NORMALIZER",
    "ResizeKeepAspectConfig",
    "ResizeKeepAspectTransform",
    "ResizePadConfig",
    "ResizePadTransform",
    "TextNormalizer",
    "autocontrast",
    "maybe_wrap_data_parallel",
    "normalize_text",
    "normalize_texts",
    "pad_to_width",
    "resize_keep_aspect",
    "should_use_data_parallel",
    "unwrap_model",
]


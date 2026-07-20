"""Datasets and collators for TrOCR fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.constants import ID_COL, TARGET_COL
from src.common.image_preprocessing import ResizePadConfig, ResizePadTransform
from src.common.text_normalization import normalize_text


TrOCRPreprocessMode = Literal["default", "aspect"]


@dataclass(frozen=True)
class TrOCRSample:
    """One image-text sample for TrOCR."""

    image_id: str
    image: Image.Image
    text: str | None


@dataclass(frozen=True)
class TrOCRBatch:
    """Batch consumed by VisionEncoderDecoderModel."""

    pixel_values: Tensor
    labels: Tensor | None
    image_ids: list[str]
    texts: list[str | None]


class TrOCRLineDataset(Dataset[TrOCRSample]):
    """PyTorch dataset for line-level TrOCR experiments."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        target_col: str = TARGET_COL,
        image_transform: Callable[[Image.Image], Image.Image] | None = None,
        normalize_targets: bool = True,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        self.target_col = target_col
        self.image_transform = image_transform
        self.normalize_targets = normalize_targets

        required = {ID_COL, "image_path"}
        missing = required - set(self.manifest.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> TrOCRSample:
        row = self.manifest.iloc[idx]
        with Image.open(Path(row["image_path"])) as image:
            image = image.convert("RGB")
            if self.image_transform is not None:
                image = self.image_transform(image)

        text = None
        if self.target_col in row and pd.notna(row[self.target_col]):
            text = (
                normalize_text(row[self.target_col])
                if self.normalize_targets
                else str(row[self.target_col])
            )

        return TrOCRSample(
            image_id=str(row[ID_COL]),
            image=image,
            text=text,
        )


@dataclass(frozen=True)
class TrOCRCollate:
    """Collate samples with a Hugging Face TrOCR processor."""

    processor: object
    preprocess_mode: TrOCRPreprocessMode = "default"
    max_label_length: int = 160

    def __post_init__(self) -> None:
        if self.preprocess_mode not in {"default", "aspect"}:
            raise ValueError("preprocess_mode must be one of: default, aspect.")
        if self.max_label_length <= 0:
            raise ValueError("max_label_length must be positive.")

    def __call__(self, samples: list[TrOCRSample]) -> TrOCRBatch:
        if not samples:
            raise ValueError("Cannot collate an empty batch.")

        processor_kwargs = {"return_tensors": "pt"}
        if self.preprocess_mode == "aspect":
            processor_kwargs["do_resize"] = False

        encoded_images = self.processor(
            images=[sample.image for sample in samples],
            **processor_kwargs,
        )
        texts = [sample.text for sample in samples]
        labels = None
        if all(text is not None for text in texts):
            labels = self._encode_labels([str(text) for text in texts])

        return TrOCRBatch(
            pixel_values=encoded_images["pixel_values"],
            labels=labels,
            image_ids=[sample.image_id for sample in samples],
            texts=texts,
        )

    def _encode_labels(self, texts: list[str]) -> Tensor:
        tokenizer = self.processor.tokenizer
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_label_length,
            return_tensors="pt",
        )
        labels = encoded.input_ids
        labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)
        return labels


def make_trocr_image_transform(
    *,
    preprocess_mode: TrOCRPreprocessMode,
    target_height: int = 384,
    canvas_width: int = 1536,
    pad_value: int = 255,
) -> ResizePadTransform | None:
    """Create an optional image transform for TrOCR preprocessing."""

    if preprocess_mode == "default":
        return None
    if preprocess_mode == "aspect":
        return ResizePadTransform(
            ResizePadConfig(
                target_height=target_height,
                max_width=canvas_width,
                pad_value=pad_value,
                align="left",
            )
        )
    raise ValueError("preprocess_mode must be one of: default, aspect.")

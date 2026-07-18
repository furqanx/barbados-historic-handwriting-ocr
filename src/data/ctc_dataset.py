"""PyTorch datasets and collate functions for CRNN-CTC training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.constants import ID_COL, TARGET_COL
from src.data.char_tokenizer import CharacterTokenizer
from src.features.image_preprocessing import (
    ResizeKeepAspectConfig,
    ResizeKeepAspectTransform,
)
from src.features.text_normalization import normalize_text


TensorTransform = Callable[[Image.Image], Tensor]


@dataclass(frozen=True)
class CTCSample:
    """One sample prepared for CTC training or inference."""

    image_id: str
    image: Tensor
    image_width: int
    target_ids: Tensor
    target_length: int
    text: str | None


@dataclass(frozen=True)
class CTCBatch:
    """Batch returned by the CTC collate function."""

    images: Tensor
    image_widths: Tensor
    input_lengths: Tensor
    targets: Tensor
    target_lengths: Tensor
    image_ids: list[str]
    texts: list[str | None]


def pil_to_tensor(image: Image.Image) -> Tensor:
    """Convert an RGB PIL image into a float tensor in CHW format."""

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image array, got shape {array.shape}")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class CTCLineDataset(Dataset[CTCSample]):
    """PyTorch dataset for handwritten line recognition with CTC targets."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        tokenizer: CharacterTokenizer,
        *,
        target_col: str = TARGET_COL,
        image_transform: Callable[[Image.Image], Image.Image] | None = None,
        tensor_transform: TensorTransform = pil_to_tensor,
        normalize_targets: bool = True,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.target_col = target_col
        self.image_transform = image_transform
        self.tensor_transform = tensor_transform
        self.normalize_targets = normalize_targets

        required = {ID_COL, "image_path"}
        missing = required - set(self.manifest.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> CTCSample:
        row = self.manifest.iloc[idx]
        image_id = str(row[ID_COL])
        image_path = Path(row["image_path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.image_transform is not None:
                image = self.image_transform(image)
            image_tensor = self.tensor_transform(image)

        if image_tensor.ndim != 3:
            raise ValueError(f"Expected CHW image tensor, got shape {image_tensor.shape}")

        text = None
        target_ids: list[int] = []
        if self.target_col in row and pd.notna(row[self.target_col]):
            text = (
                normalize_text(row[self.target_col])
                if self.normalize_targets
                else str(row[self.target_col])
            )
            target_ids = self.tokenizer.encode(text, normalize=False)

        target_tensor = torch.tensor(target_ids, dtype=torch.long)
        return CTCSample(
            image_id=image_id,
            image=image_tensor.float(),
            image_width=int(image_tensor.shape[-1]),
            target_ids=target_tensor,
            target_length=int(target_tensor.numel()),
            text=text,
        )


@dataclass(frozen=True)
class CTCCollate:
    """Collate CTC samples with dynamic width padding."""

    pad_value: float = 1.0
    width_multiple: int = 1
    time_downsample_factor: int = 1

    def __post_init__(self) -> None:
        if self.width_multiple <= 0:
            raise ValueError("width_multiple must be positive.")
        if self.time_downsample_factor <= 0:
            raise ValueError("time_downsample_factor must be positive.")

    def __call__(self, samples: list[CTCSample]) -> CTCBatch:
        if not samples:
            raise ValueError("Cannot collate an empty batch.")

        channels = samples[0].image.shape[0]
        height = samples[0].image.shape[1]
        for sample in samples:
            if sample.image.ndim != 3:
                raise ValueError(f"Expected CHW image tensor, got {sample.image.shape}")
            if sample.image.shape[0] != channels or sample.image.shape[1] != height:
                raise ValueError("All images in a CTC batch must share channels and height.")

        image_widths = torch.tensor([sample.image_width for sample in samples], dtype=torch.long)
        max_width = int(image_widths.max().item())
        padded_width = _round_up_to_multiple(max_width, self.width_multiple)

        batch_images = samples[0].image.new_full(
            (len(samples), channels, height, padded_width),
            fill_value=self.pad_value,
        )
        for idx, sample in enumerate(samples):
            width = sample.image.shape[-1]
            batch_images[idx, :, :, :width] = sample.image

        target_lengths = torch.tensor(
            [sample.target_length for sample in samples],
            dtype=torch.long,
        )
        if int(target_lengths.sum().item()) > 0:
            targets = torch.cat([sample.target_ids for sample in samples]).long()
        else:
            targets = torch.empty(0, dtype=torch.long)

        input_lengths = torch.clamp(
            image_widths // self.time_downsample_factor,
            min=1,
        )

        return CTCBatch(
            images=batch_images,
            image_widths=image_widths,
            input_lengths=input_lengths,
            targets=targets,
            target_lengths=target_lengths,
            image_ids=[sample.image_id for sample in samples],
            texts=[sample.text for sample in samples],
        )


def make_ctc_image_transform(
    *,
    target_height: int = 96,
    max_width: int | None = 2048,
    autocontrast_cutoff: int | None = None,
) -> ResizeKeepAspectTransform:
    """Create the default aspect-ratio-aware transform for CRNN-CTC."""

    return ResizeKeepAspectTransform(
        ResizeKeepAspectConfig(
            target_height=target_height,
            max_width=max_width,
            autocontrast_cutoff=autocontrast_cutoff,
        )
    )


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round value up to the nearest multiple."""

    return ((value + multiple - 1) // multiple) * multiple

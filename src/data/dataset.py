"""Dataset helpers for OCR/HTR experiments.

This module intentionally avoids framework-specific base classes. Training code
can wrap `HandwritingLineDataset` with PyTorch, TensorFlow, or Hugging Face
later without changing manifest handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd
from PIL import Image

from src.constants import ID_COL, TARGET_COL


ImageTransform = Callable[[Image.Image], object]


@dataclass(frozen=True)
class HandwritingSample:
    """One handwriting line sample."""

    image_id: str
    image_path: Path
    target: str | None
    image: Image.Image | object


class HandwritingLineDataset:
    """Lightweight dataset backed by a manifest dataframe."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        target_col: str = TARGET_COL,
        transform: ImageTransform | None = None,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        self.target_col = target_col
        self.transform = transform

        required = {ID_COL, "image_path"}
        missing = required - set(self.manifest.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __iter__(self) -> Iterator[HandwritingSample]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int) -> HandwritingSample:
        row = self.manifest.iloc[idx]
        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = None
        if self.target_col in row and pd.notna(row[self.target_col]):
            target = str(row[self.target_col])

        return HandwritingSample(
            image_id=str(row[ID_COL]),
            image_path=image_path,
            target=target,
            image=image,
        )


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a saved manifest CSV."""

    return pd.read_csv(path)


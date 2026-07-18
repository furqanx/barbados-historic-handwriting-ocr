"""Prediction and submission helpers for TrOCR checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.constants import ID_COL, TARGET_COL
from src.data.trocr_dataset import (
    TrOCRCollate,
    TrOCRLineDataset,
    TrOCRPreprocessMode,
    make_trocr_image_transform,
)


def load_trocr_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: torch.device | None = None,
) -> tuple[VisionEncoderDecoderModel, TrOCRProcessor, dict]:
    """Load a TrOCR checkpoint directory saved with save_pretrained."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    processor = TrOCRProcessor.from_pretrained(checkpoint_dir, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir)
    metadata_path = checkpoint_dir / "training_metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model.to(device)
    model.eval()
    return model, processor, metadata


@torch.no_grad()
def predict_manifest(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    manifest: pd.DataFrame,
    *,
    preprocess_mode: TrOCRPreprocessMode = "default",
    batch_size: int = 4,
    num_workers: int = 2,
    target_height: int = 384,
    canvas_width: int = 1536,
    max_label_length: int = 192,
    max_generation_length: int = 192,
    num_beams: int = 1,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Predict transcriptions for a manifest dataframe."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_transform = make_trocr_image_transform(
        preprocess_mode=preprocess_mode,
        target_height=target_height,
        canvas_width=canvas_width,
    )
    collate = TrOCRCollate(
        processor=processor,
        preprocess_mode=preprocess_mode,
        max_label_length=max_label_length,
    )
    loader = DataLoader(
        TrOCRLineDataset(manifest, image_transform=image_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )

    rows: list[dict[str, str]] = []
    model.eval()
    for batch in tqdm(loader, desc="predict", leave=False):
        pixel_values = batch.pixel_values.to(device, non_blocking=True)
        generated_ids = model.generate(
            pixel_values,
            max_length=max_generation_length,
            num_beams=num_beams,
            interpolate_pos_encoding=preprocess_mode == "aspect",
        )
        predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        rows.extend(
            {ID_COL: image_id, TARGET_COL: prediction}
            for image_id, prediction in zip(batch.image_ids, predictions)
        )

    return pd.DataFrame(rows)


def make_submission(
    predictions: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """Align predictions to sample submission ID order."""

    required = {ID_COL, TARGET_COL}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")

    prediction_map = predictions.set_index(ID_COL)[TARGET_COL].to_dict()
    submission = sample_submission.copy()
    missing_ids = sorted(set(submission[ID_COL]) - set(prediction_map))
    if missing_ids:
        preview = ", ".join(map(str, missing_ids[:5]))
        raise ValueError(f"Missing predictions for {len(missing_ids)} IDs: {preview}")

    submission[TARGET_COL] = submission[ID_COL].map(prediction_map).fillna("")
    return submission


def get_prediction_config(metadata: dict) -> dict:
    """Extract prediction defaults from training metadata."""

    training_config = metadata.get("training_config", {})
    return {
        "preprocess_mode": training_config.get("preprocess_mode", "default"),
        "batch_size": int(training_config.get("batch_size", 4)),
        "num_workers": int(training_config.get("num_workers", 2)),
        "target_height": int(training_config.get("target_height", 384)),
        "canvas_width": int(training_config.get("canvas_width", 1536)),
        "max_label_length": int(training_config.get("max_label_length", 192)),
        "max_generation_length": int(
            training_config.get("max_generation_length", 192)
        ),
        "num_beams": int(training_config.get("num_beams", 1)),
    }

"""Prediction and submission helpers for CRNN-CTC checkpoints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.constants import ID_COL, TARGET_COL
from src.data.char_tokenizer import CharacterTokenizer
from src.data.ctc_dataset import CTCCollate, CTCLineDataset, make_ctc_image_transform
from src.inference.ctc_decoder import greedy_decode_batch
from src.models.ctc_parallel import ctc_logits_to_time_first, maybe_wrap_ctc_data_parallel
from src.models.convnext_ctc import ConvNeXtCTCConfig, ConvNeXtCTCModel
from src.models.crnn_ctc import CRNNCTCConfig, CRNNCTCModel
from src.models.resnet_ctc import ResNetCTCConfig, ResNetCTCModel
from src.utils.torch_utils import unwrap_model


CTCPredictModel = CRNNCTCModel | ResNetCTCModel | ConvNeXtCTCModel


def load_crnn_ctc_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | None = None,
) -> tuple[CTCPredictModel, CharacterTokenizer, dict]:
    """Load a CRNN/ResNet CTC checkpoint and tokenizer metadata."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = CharacterTokenizer.from_dict(checkpoint["tokenizer"])
    model_type = checkpoint.get("model_type", "crnn_ctc")
    if model_type == "crnn_ctc":
        model_config = CRNNCTCConfig(**checkpoint["model_config"])
        model = CRNNCTCModel(model_config)
    elif model_type == "resnet_ctc":
        model_config = ResNetCTCConfig(**checkpoint["model_config"])
        model = ResNetCTCModel(model_config)
    elif model_type == "convnext_ctc":
        config_payload = dict(checkpoint["model_config"])
        config_payload["pretrained"] = False
        model_config = ConvNeXtCTCConfig(**config_payload)
        model = ConvNeXtCTCModel(model_config)
    else:
        raise ValueError(f"Unsupported CTC checkpoint model_type: {model_type}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer, checkpoint


@torch.no_grad()
def predict_manifest(
    model: CTCPredictModel,
    tokenizer: CharacterTokenizer,
    manifest: pd.DataFrame,
    *,
    batch_size: int = 8,
    num_workers: int = 2,
    target_height: int = 96,
    max_width: int | None = 2048,
    autocontrast_cutoff: int | None = None,
    pad_value: float = 1.0,
    width_multiple: int = 4,
    use_data_parallel: bool = True,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Predict transcriptions for a manifest dataframe."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_downsample_factor = unwrap_model(model).time_downsample_factor
    model = maybe_wrap_ctc_data_parallel(
        model,
        device,
        enabled=use_data_parallel,
    )
    image_transform = make_ctc_image_transform(
        target_height=target_height,
        max_width=max_width,
        autocontrast_cutoff=autocontrast_cutoff,
    )
    collate = CTCCollate(
        pad_value=pad_value,
        width_multiple=width_multiple,
        time_downsample_factor=time_downsample_factor,
    )
    loader = DataLoader(
        CTCLineDataset(manifest, tokenizer, image_transform=image_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )

    rows: list[dict[str, str]] = []
    model.eval()
    for batch in tqdm(loader, desc="predict", leave=False):
        images = batch.images.to(device, non_blocking=True)
        input_lengths = batch.input_lengths.to(device, non_blocking=True)
        logits = model(images)
        logits = ctc_logits_to_time_first(logits, model)
        input_lengths = input_lengths.clamp(max=logits.shape[0])
        predictions = greedy_decode_batch(logits, input_lengths, tokenizer)
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

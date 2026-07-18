"""Training utilities for TrOCR fine-tuning."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.constants import FOLD_COL, ID_COL, TARGET_COL
from src.data.trocr_dataset import (
    TrOCRCollate,
    TrOCRLineDataset,
    TrOCRPreprocessMode,
    make_trocr_image_transform,
)
from src.evaluation.metrics import TranscriptionScore, score_transcriptions
from src.utils.torch_utils import maybe_wrap_data_parallel, unwrap_model


@dataclass(frozen=True)
class TrOCRTrainingConfig:
    """Configuration for one TrOCR fine-tuning run."""

    model_name: str = "microsoft/trocr-small-handwritten"
    preprocess_mode: TrOCRPreprocessMode = "default"
    fold: int = 0
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_workers: int = 2
    target_height: int = 384
    canvas_width: int = 1536
    max_label_length: int = 192
    max_generation_length: int = 192
    num_beams: int = 1
    gradient_clip_norm: float = 1.0
    use_amp: bool = True
    use_data_parallel: bool = True
    seed: int = 42
    freeze_encoder: bool = False
    freeze_decoder: bool = False
    freeze_encoder_layers: int = 0
    freeze_decoder_layers: int = 0
    max_train_samples: int | None = None
    max_valid_samples: int | None = None


@dataclass(frozen=True)
class TrOCREpochMetrics:
    """Metrics reported after a TrOCR epoch."""

    loss: float
    wer: float | None = None
    cer: float | None = None
    score: float | None = None


@dataclass(frozen=True)
class TrOCRValidationResult:
    """Validation metrics plus row-level predictions."""

    metrics: TrOCREpochMetrics
    predictions: pd.DataFrame


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible fine-tuning."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_trocr_model_and_processor(
    model_name: str,
    *,
    device: torch.device,
    freeze_encoder: bool = False,
    freeze_decoder: bool = False,
    freeze_encoder_layers: int = 0,
    freeze_decoder_layers: int = 0,
) -> tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
    """Load TrOCR model/processor and apply optional freezing."""

    processor = TrOCRProcessor.from_pretrained(model_name, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    _configure_generation_tokens(model, processor)
    _apply_freezing(
        model,
        freeze_encoder=freeze_encoder,
        freeze_decoder=freeze_decoder,
        freeze_encoder_layers=freeze_encoder_layers,
        freeze_decoder_layers=freeze_decoder_layers,
    )
    model.to(device)
    return model, processor


def build_dataloaders(
    train_manifest: pd.DataFrame,
    processor: TrOCRProcessor,
    config: TrOCRTrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation dataloaders for one fold."""

    train_df = train_manifest[train_manifest[FOLD_COL] != config.fold].reset_index(drop=True)
    valid_df = train_manifest[train_manifest[FOLD_COL] == config.fold].reset_index(drop=True)
    if config.max_train_samples is not None:
        train_df = train_df.head(config.max_train_samples).reset_index(drop=True)
    if config.max_valid_samples is not None:
        valid_df = valid_df.head(config.max_valid_samples).reset_index(drop=True)
    if train_df.empty or valid_df.empty:
        raise ValueError(f"Fold {config.fold} produced empty train or validation data.")

    image_transform = make_trocr_image_transform(
        preprocess_mode=config.preprocess_mode,
        target_height=config.target_height,
        canvas_width=config.canvas_width,
    )
    collate = TrOCRCollate(
        processor=processor,
        preprocess_mode=config.preprocess_mode,
        max_label_length=config.max_label_length,
    )
    train_loader = DataLoader(
        TrOCRLineDataset(train_df, image_transform=image_transform),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        TrOCRLineDataset(valid_df, image_transform=image_transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    return train_loader, valid_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    config: TrOCRTrainingConfig,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> TrOCREpochMetrics:
    """Run one TrOCR training epoch."""

    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in tqdm(loader, desc="train", leave=False):
        pixel_values = batch.pixel_values.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
                interpolate_pos_encoding=config.preprocess_mode == "aspect",
                return_dict=False,
            )
            loss = outputs[0]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = pixel_values.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size

    return TrOCREpochMetrics(loss=total_loss / max(total_items, 1))


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    processor: TrOCRProcessor,
    loader: DataLoader,
    *,
    config: TrOCRTrainingConfig,
    device: torch.device,
) -> TrOCRValidationResult:
    """Run validation, generation, and WER/CER scoring."""

    model.eval()
    total_loss = 0.0
    total_items = 0
    references: list[str] = []
    predictions: list[str] = []
    image_ids: list[str] = []

    for batch in tqdm(loader, desc="valid", leave=False):
        pixel_values = batch.pixel_values.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)
        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
            interpolate_pos_encoding=config.preprocess_mode == "aspect",
            return_dict=False,
        )
        generator = unwrap_model(model)
        generated_ids = generator.generate(
            pixel_values,
            max_length=config.max_generation_length,
            num_beams=config.num_beams,
            interpolate_pos_encoding=config.preprocess_mode == "aspect",
        )
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

        predictions.extend(decoded)
        references.extend(["" if text is None else text for text in batch.texts])
        image_ids.extend(batch.image_ids)

        batch_size = pixel_values.shape[0]
        total_loss += float(outputs[0].detach().cpu()) * batch_size
        total_items += batch_size

    score = score_transcriptions(references, predictions)
    return TrOCRValidationResult(
        metrics=TrOCREpochMetrics(
            loss=total_loss / max(total_items, 1),
            wer=score.wer,
            cer=score.cer,
            score=score.score,
        ),
        predictions=pd.DataFrame(
            {
                ID_COL: image_ids,
                "reference": references,
                TARGET_COL: predictions,
            }
        ),
    )


def train_trocr(
    train_manifest: pd.DataFrame,
    *,
    config: TrOCRTrainingConfig,
    checkpoint_dir: str | Path,
    valid_predictions_path: str | Path | None = None,
    device: torch.device | None = None,
) -> tuple[VisionEncoderDecoderModel, TranscriptionScore | None]:
    """Fine-tune TrOCR on one fold and save the best checkpoint."""

    set_seed(config.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = prepare_trocr_model_and_processor(
        config.model_name,
        device=device,
        freeze_encoder=config.freeze_encoder,
        freeze_decoder=config.freeze_decoder,
        freeze_encoder_layers=config.freeze_encoder_layers,
        freeze_decoder_layers=config.freeze_decoder_layers,
    )
    model = maybe_wrap_data_parallel(
        model,
        device,
        enabled=config.use_data_parallel,
    )
    train_loader, valid_loader = build_dataloaders(train_manifest, processor, config)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=config.use_amp and device.type == "cuda",
    )

    best_score: float | None = None
    best_transcription_score: TranscriptionScore | None = None
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            config=config,
            device=device,
            scaler=scaler,
        )
        valid_result = validate_one_epoch(
            model,
            processor,
            valid_loader,
            config=config,
            device=device,
        )
        valid_metrics = valid_result.metrics

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics.loss:.5f} "
            f"valid_loss={valid_metrics.loss:.5f} "
            f"wer={valid_metrics.wer:.5f} "
            f"cer={valid_metrics.cer:.5f} "
            f"score={valid_metrics.score:.5f}"
        )

        if best_score is None or valid_metrics.score < best_score:
            best_score = valid_metrics.score
            best_transcription_score = TranscriptionScore(
                wer=float(valid_metrics.wer),
                cer=float(valid_metrics.cer),
            )
            save_trocr_checkpoint(
                checkpoint_dir,
                model=model,
                processor=processor,
                config=config,
                epoch=epoch,
                valid_metrics=valid_metrics,
            )
            print(f"saved_best_checkpoint={checkpoint_dir}")
            if valid_predictions_path is not None:
                saved_predictions = save_predictions(
                    valid_result.predictions,
                    valid_predictions_path,
                )
                print(f"saved_best_valid_predictions={saved_predictions}")

    return model, best_transcription_score


def save_trocr_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    config: TrOCRTrainingConfig,
    epoch: int,
    valid_metrics: TrOCREpochMetrics,
) -> Path:
    """Save model, processor, and run metadata in Hugging Face format."""

    output = Path(checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = unwrap_model(model)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    metadata = {
        "epoch": epoch,
        "training_config": asdict(config),
        "valid_metrics": asdict(valid_metrics),
    }
    (output / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def save_predictions(predictions: pd.DataFrame, path: str | Path) -> Path:
    """Save row-level predictions to CSV."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    return output


def _configure_generation_tokens(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
) -> None:
    tokenizer = processor.tokenizer
    decoder_start_token_id = (
        tokenizer.cls_token_id
        if tokenizer.cls_token_id is not None
        else tokenizer.bos_token_id
    )
    eos_token_id = (
        tokenizer.sep_token_id
        if tokenizer.sep_token_id is not None
        else tokenizer.eos_token_id
    )
    if decoder_start_token_id is None:
        raise ValueError("Tokenizer has no CLS/BOS token for decoder_start_token_id.")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no PAD token.")

    model.config.decoder_start_token_id = decoder_start_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size


def _apply_freezing(
    model: VisionEncoderDecoderModel,
    *,
    freeze_encoder: bool,
    freeze_decoder: bool,
    freeze_encoder_layers: int,
    freeze_decoder_layers: int,
) -> None:
    if freeze_encoder:
        _freeze_module(model.encoder)
    if freeze_decoder:
        _freeze_module(model.decoder)
    _freeze_first_layers(_get_encoder_layers(model), freeze_encoder_layers)
    _freeze_first_layers(_get_decoder_layers(model), freeze_decoder_layers)


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _freeze_first_layers(layers: list[nn.Module], count: int) -> None:
    if count <= 0:
        return
    for layer in layers[:count]:
        _freeze_module(layer)


def _get_encoder_layers(model: VisionEncoderDecoderModel) -> list[nn.Module]:
    encoder = model.encoder
    layers = getattr(getattr(encoder, "encoder", None), "layer", None)
    return list(layers) if layers is not None else []


def _get_decoder_layers(model: VisionEncoderDecoderModel) -> list[nn.Module]:
    decoder = model.decoder
    candidate_paths = [
        ("model", "decoder", "layers"),
        ("decoder", "layers"),
        ("transformer", "h"),
    ]
    for path in candidate_paths:
        current = decoder
        for attr in path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None:
            return list(current)
    return []

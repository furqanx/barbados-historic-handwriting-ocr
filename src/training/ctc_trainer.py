"""Training utilities for the CRNN-CTC baseline."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.constants import FOLD_COL, ID_COL, TARGET_COL
from src.data.char_tokenizer import CharacterTokenizer
from src.data.ctc_dataset import CTCCollate, CTCLineDataset, make_ctc_image_transform
from src.evaluation.metrics import TranscriptionScore, score_transcriptions
from src.inference.ctc_decoder import greedy_decode_batch
from src.models.crnn_ctc import CRNNCTCConfig, CRNNCTCModel
from src.models.resnet_ctc import ResNetCTCConfig, ResNetCTCModel


CTCModel = CRNNCTCModel | ResNetCTCModel
CTCModelConfig = CRNNCTCConfig | ResNetCTCConfig


@dataclass(frozen=True)
class CTCTrainingConfig:
    """Configuration for one CRNN-CTC training run."""

    fold: int = 0
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    target_height: int = 96
    max_width: int | None = 2048
    autocontrast_cutoff: int | None = None
    pad_value: float = 1.0
    width_multiple: int = 4
    gradient_clip_norm: float = 5.0
    use_amp: bool = True
    seed: int = 42


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics reported after an epoch."""

    loss: float
    wer: float | None = None
    cer: float | None = None
    score: float | None = None


@dataclass(frozen=True)
class ValidationResult:
    """Validation metrics plus row-level predictions."""

    metrics: EpochMetrics
    predictions: pd.DataFrame


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible baseline runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    train_manifest: pd.DataFrame,
    tokenizer: CharacterTokenizer,
    config: CTCTrainingConfig,
    *,
    time_downsample_factor: int = CRNNCTCModel.time_downsample_factor,
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation dataloaders for one fold."""

    train_df = train_manifest[train_manifest[FOLD_COL] != config.fold].reset_index(drop=True)
    valid_df = train_manifest[train_manifest[FOLD_COL] == config.fold].reset_index(drop=True)
    if train_df.empty or valid_df.empty:
        raise ValueError(f"Fold {config.fold} produced empty train or validation data.")

    image_transform = make_ctc_image_transform(
        target_height=config.target_height,
        max_width=config.max_width,
        autocontrast_cutoff=config.autocontrast_cutoff,
    )
    collate = CTCCollate(
        pad_value=config.pad_value,
        width_multiple=config.width_multiple,
        time_downsample_factor=time_downsample_factor,
    )

    train_loader = DataLoader(
        CTCLineDataset(train_df, tokenizer, image_transform=image_transform),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        CTCLineDataset(valid_df, tokenizer, image_transform=image_transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    return train_loader, valid_loader


def train_one_epoch(
    model: CTCModel,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    gradient_clip_norm: float,
) -> EpochMetrics:
    """Run one training epoch."""

    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch.images.to(device, non_blocking=True)
        targets = batch.targets.to(device, non_blocking=True)
        target_lengths = batch.target_lengths.to(device, non_blocking=True)
        input_lengths = batch.input_lengths.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(images)
            log_probs = logits.log_softmax(dim=-1)
            input_lengths = input_lengths.clamp(max=logits.shape[0])
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size

    return EpochMetrics(loss=total_loss / max(total_items, 1))


@torch.no_grad()
def validate_one_epoch(
    model: CTCModel,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    tokenizer: CharacterTokenizer,
    *,
    device: torch.device,
) -> ValidationResult:
    """Run validation and compute CTC loss plus WER/CER."""

    model.eval()
    total_loss = 0.0
    total_items = 0
    references: list[str] = []
    predictions: list[str] = []
    image_ids: list[str] = []

    for batch in tqdm(loader, desc="valid", leave=False):
        images = batch.images.to(device, non_blocking=True)
        targets = batch.targets.to(device, non_blocking=True)
        target_lengths = batch.target_lengths.to(device, non_blocking=True)
        input_lengths = batch.input_lengths.to(device, non_blocking=True)

        logits = model(images)
        log_probs = logits.log_softmax(dim=-1)
        input_lengths = input_lengths.clamp(max=logits.shape[0])
        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        decoded = greedy_decode_batch(logits, input_lengths, tokenizer)
        predictions.extend(decoded)
        references.extend(["" if text is None else text for text in batch.texts])
        image_ids.extend(batch.image_ids)

        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size

    score = score_transcriptions(references, predictions)
    return ValidationResult(
        metrics=EpochMetrics(
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


def save_checkpoint(
    path: str | Path,
    *,
    model: CTCModel,
    tokenizer: CharacterTokenizer,
    model_config: CTCModelConfig,
    model_type: str,
    training_config: CTCTrainingConfig,
    epoch: int,
    valid_metrics: EpochMetrics,
) -> Path:
    """Save a training checkpoint."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "model_config": model_config.to_dict(),
            "training_config": asdict(training_config),
            "tokenizer": tokenizer.to_dict(),
            "valid_metrics": asdict(valid_metrics),
        },
        output,
    )
    return output


def save_predictions(predictions: pd.DataFrame, path: str | Path) -> Path:
    """Save row-level predictions to CSV."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    return output


def train_ctc_model(
    train_manifest: pd.DataFrame,
    tokenizer: CharacterTokenizer,
    *,
    model: CTCModel,
    model_config: CTCModelConfig,
    model_type: str,
    config: CTCTrainingConfig,
    checkpoint_path: str | Path,
    valid_predictions_path: str | Path | None = None,
    device: torch.device | None = None,
) -> tuple[CTCModel, TranscriptionScore | None]:
    """Train a CTC OCR model on one fold and save the best checkpoint."""

    set_seed(config.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader, valid_loader = build_dataloaders(
        train_manifest,
        tokenizer,
        config,
        time_downsample_factor=model.time_downsample_factor,
    )

    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=config.use_amp and device.type == "cuda"
    )

    best_score: float | None = None
    best_transcription_score: TranscriptionScore | None = None
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            scaler=scaler,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        valid_result = validate_one_epoch(
            model,
            valid_loader,
            criterion,
            tokenizer,
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
            save_checkpoint(
                checkpoint_path,
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
                model_type=model_type,
                training_config=config,
                epoch=epoch,
                valid_metrics=valid_metrics,
            )
            print(f"saved_best_checkpoint={checkpoint_path}")
            if valid_predictions_path is not None:
                saved_predictions = save_predictions(
                    valid_result.predictions,
                    valid_predictions_path,
                )
                print(f"saved_best_valid_predictions={saved_predictions}")

    return model, best_transcription_score


def train_crnn_ctc(
    train_manifest: pd.DataFrame,
    tokenizer: CharacterTokenizer,
    *,
    config: CTCTrainingConfig,
    checkpoint_path: str | Path,
    valid_predictions_path: str | Path | None = None,
    device: torch.device | None = None,
) -> tuple[CRNNCTCModel, TranscriptionScore | None]:
    """Train CRNN-CTC on one fold and save the best checkpoint."""

    model_config = CRNNCTCConfig(vocab_size=tokenizer.vocab_size)
    model = CRNNCTCModel(model_config)
    return train_ctc_model(
        train_manifest,
        tokenizer,
        model=model,
        model_config=model_config,
        model_type="crnn_ctc",
        config=config,
        checkpoint_path=checkpoint_path,
        valid_predictions_path=valid_predictions_path,
        device=device,
    )


def train_resnet_ctc(
    train_manifest: pd.DataFrame,
    tokenizer: CharacterTokenizer,
    *,
    config: CTCTrainingConfig,
    checkpoint_path: str | Path,
    valid_predictions_path: str | Path | None = None,
    device: torch.device | None = None,
    base_channels: int = 64,
    rnn_hidden_size: int = 256,
    rnn_layers: int = 2,
    dropout: float = 0.1,
) -> tuple[ResNetCTCModel, TranscriptionScore | None]:
    """Train ResNet-CTC on one fold and save the best checkpoint."""

    model_config = ResNetCTCConfig(
        vocab_size=tokenizer.vocab_size,
        base_channels=base_channels,
        rnn_hidden_size=rnn_hidden_size,
        rnn_layers=rnn_layers,
        dropout=dropout,
    )
    model = ResNetCTCModel(model_config)
    return train_ctc_model(
        train_manifest,
        tokenizer,
        model=model,
        model_config=model_config,
        model_type="resnet_ctc",
        config=config,
        checkpoint_path=checkpoint_path,
        valid_predictions_path=valid_predictions_path,
        device=device,
    )

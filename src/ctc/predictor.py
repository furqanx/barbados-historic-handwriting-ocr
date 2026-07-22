"""Prediction and submission helpers for CRNN-CTC checkpoints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.constants import ID_COL, TARGET_COL
from src.ctc.tokenizer import CharacterTokenizer
from src.ctc.dataset import CTCCollate, CTCLineDataset, make_ctc_image_transform
from src.ctc.decoder import greedy_decode_batch
from src.ctc.decoding import (
    BeamSearchConfig,
    CTCDecoderConfig,
    CharNGramLanguageModel,
    RerankConfig,
    ctc_prefix_beam_search_batch,
    rerank_candidates,
)
from src.ctc.parallel import ctc_logits_to_time_first, maybe_wrap_ctc_data_parallel
from src.ctc.models.convnext import ConvNeXtCTCConfig, ConvNeXtCTCModel
from src.ctc.models.crnn import CRNNCTCConfig, CRNNCTCModel
from src.ctc.models.resnet import ResNetCTCConfig, ResNetCTCModel
from src.common.torch_utils import unwrap_model


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
    decoder_config: CTCDecoderConfig | None = None,
    language_model: CharNGramLanguageModel | None = None,
    include_references: bool = True,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Predict transcriptions for a manifest dataframe."""

    decoder_config = decoder_config or CTCDecoderConfig()
    if decoder_config.decoder in {"beam_lm", "beam_lm_rerank"} and language_model is None:
        raise ValueError(f"Decoder {decoder_config.decoder!r} requires a language model.")

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
        predictions, candidate_payloads = decode_ctc_batch(
            logits,
            input_lengths,
            tokenizer,
            decoder_config=decoder_config,
            language_model=language_model,
        )

        for idx, (image_id, prediction) in enumerate(
            zip(batch.image_ids, predictions, strict=True)
        ):
            row = {ID_COL: image_id, TARGET_COL: prediction}
            if include_references and batch.texts[idx] is not None:
                row["reference"] = batch.texts[idx]
            if candidate_payloads is not None:
                row["candidates"] = candidate_payloads[idx]
            rows.append(row)

    return pd.DataFrame(rows)


def decode_ctc_batch(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    tokenizer: CharacterTokenizer,
    *,
    decoder_config: CTCDecoderConfig,
    language_model: CharNGramLanguageModel | None = None,
) -> tuple[list[str], list[str] | None]:
    """Decode one CTC batch according to the requested strategy."""

    if decoder_config.decoder == "greedy":
        return greedy_decode_batch(logits, input_lengths, tokenizer), None

    beam_config = BeamSearchConfig(
        beam_size=decoder_config.beam_size,
        top_tokens_per_step=decoder_config.top_tokens_per_step,
        lm_weight=decoder_config.lm_weight,
        length_bonus=decoder_config.length_bonus,
        candidates_top_k=decoder_config.candidates_top_k,
    )
    candidate_batches = ctc_prefix_beam_search_batch(
        logits,
        input_lengths,
        tokenizer,
        config=beam_config,
        language_model=language_model,
    )
    if decoder_config.decoder == "beam_lm_rerank":
        rerank_config = RerankConfig(
            short_text_penalty=decoder_config.rerank_short_text_penalty,
            min_chars=decoder_config.rerank_min_chars,
            repeated_whitespace_penalty=decoder_config.rerank_repeated_whitespace_penalty,
            repeated_punctuation_penalty=decoder_config.rerank_repeated_punctuation_penalty,
            edge_space_penalty=decoder_config.rerank_edge_space_penalty,
        )
        candidate_batches = [
            rerank_candidates(candidates, rerank_config) for candidates in candidate_batches
        ]

    predictions = [candidates[0].text if candidates else "" for candidates in candidate_batches]
    candidate_payloads = [
        " ||| ".join(
            f"{candidate.text}\t{candidate.score:.6f}\t"
            f"{candidate.acoustic_score:.6f}\t{candidate.lm_score:.6f}"
            for candidate in candidates
        )
        for candidates in candidate_batches
    ]
    return predictions, candidate_payloads


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

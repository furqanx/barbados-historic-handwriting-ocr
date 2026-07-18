"""ConvNeXt-tiny visual encoder + BiLSTM + CTC model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ConvNeXtCTCConfig:
    """Configuration for the ConvNeXt-CTC model."""

    vocab_size: int
    backbone_name: str = "convnext_tiny"
    pretrained: bool = True
    out_index: int = 0
    encoder_channels: int = 96
    rnn_hidden_size: int = 256
    rnn_layers: int = 2
    dropout: float = 0.1
    normalize_images: bool = True

    def to_dict(self) -> dict[str, bool | int | float | str]:
        """Serialize config for checkpoints."""

        return asdict(self)


class ConvNeXtCTCModel(nn.Module):
    """Timm ConvNeXt feature extractor + BiLSTM sequence model + CTC head."""

    time_downsample_factor = 4

    def __init__(self, config: ConvNeXtCTCConfig) -> None:
        super().__init__()
        self.config = config
        timm = _import_timm()
        self.encoder = timm.create_model(
            config.backbone_name,
            pretrained=config.pretrained,
            features_only=True,
            out_indices=(config.out_index,),
        )
        feature_info = self.encoder.feature_info[config.out_index]
        encoder_channels = int(feature_info["num_chs"])
        reduction = int(feature_info["reduction"])
        self.time_downsample_factor = reduction

        self.sequence_model = nn.LSTM(
            input_size=encoder_channels,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_layers,
            dropout=config.dropout if config.rnn_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Linear(config.rnn_hidden_size * 2, config.vocab_size)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return logits shaped as [time, batch, vocab]."""

        if self.config.normalize_images:
            images = (images - self.image_mean) / self.image_std
        features = self.encoder(images)[0]
        features = features.mean(dim=2)
        sequence = features.permute(2, 0, 1).contiguous()
        sequence, _ = self.sequence_model(sequence)
        return self.classifier(sequence)

    def output_lengths(self, image_widths: Tensor) -> Tensor:
        """Compute CTC time-step lengths from original padded image widths."""

        return torch.clamp(image_widths // self.time_downsample_factor, min=1)


def _import_timm():
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "ConvNeXtCTCModel requires timm. Install it with `pip install timm`."
        ) from exc
    return timm

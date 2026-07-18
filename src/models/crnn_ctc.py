"""CRNN-BiLSTM-CTC baseline model for handwritten line recognition."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CRNNCTCConfig:
    """Configuration for the CRNN-CTC baseline."""

    vocab_size: int
    input_channels: int = 3
    rnn_hidden_size: int = 256
    rnn_layers: int = 2
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        """Serialize config for checkpoints."""

        return asdict(self)


class ConvBlock(nn.Module):
    """Conv-BN-ReLU block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.block(images)


class CRNNCTCModel(nn.Module):
    """CNN encoder + BiLSTM sequence model + CTC classifier."""

    time_downsample_factor = 4

    def __init__(self, config: CRNNCTCConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            ConvBlock(config.input_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ConvBlock(256, 512),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ConvBlock(512, 512),
        )
        self.sequence_model = nn.LSTM(
            input_size=512,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_layers,
            dropout=config.dropout if config.rnn_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Linear(config.rnn_hidden_size * 2, config.vocab_size)

    def forward(self, images: Tensor) -> Tensor:
        """Return logits shaped as [time, batch, vocab]."""

        features = self.encoder(images)
        features = features.mean(dim=2)
        sequence = features.permute(2, 0, 1).contiguous()
        sequence, _ = self.sequence_model(sequence)
        return self.classifier(sequence)

    def output_lengths(self, image_widths: Tensor) -> Tensor:
        """Compute CTC time-step lengths from original padded image widths."""

        return torch.clamp(image_widths // self.time_downsample_factor, min=1)

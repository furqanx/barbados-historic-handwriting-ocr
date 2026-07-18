"""ResNet-style CNN + BiLSTM + CTC model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ResNetCTCConfig:
    """Configuration for the ResNet-CTC model."""

    vocab_size: int
    input_channels: int = 3
    base_channels: int = 64
    rnn_hidden_size: int = 256
    rnn_layers: int = 2
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        """Serialize config for checkpoints."""

        return asdict(self)


class ResidualBlock(nn.Module):
    """Small residual block for OCR line images."""

    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, images: Tensor) -> Tensor:
        residual = self.shortcut(images)
        features = self.relu(self.bn1(self.conv1(images)))
        features = self.bn2(self.conv2(features))
        return self.relu(features + residual)


class ResNetCTCModel(nn.Module):
    """ResNet-style visual encoder + BiLSTM sequence model + CTC classifier."""

    time_downsample_factor = 4

    def __init__(self, config: ResNetCTCConfig) -> None:
        super().__init__()
        self.config = config
        c = config.base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(config.input_channels, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.Sequential(
            ResidualBlock(c, c, stride=(1, 1)),
            ResidualBlock(c, c * 2, stride=(2, 2)),
            ResidualBlock(c * 2, c * 2, stride=(1, 1)),
            ResidualBlock(c * 2, c * 4, stride=(2, 2)),
            ResidualBlock(c * 4, c * 4, stride=(1, 1)),
            ResidualBlock(c * 4, c * 8, stride=(2, 1)),
            ResidualBlock(c * 8, c * 8, stride=(2, 1)),
        )
        self.sequence_model = nn.LSTM(
            input_size=c * 8,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_layers,
            dropout=config.dropout if config.rnn_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Linear(config.rnn_hidden_size * 2, config.vocab_size)

    def forward(self, images: Tensor) -> Tensor:
        """Return logits shaped as [time, batch, vocab]."""

        features = self.encoder(self.stem(images))
        features = features.mean(dim=2)
        sequence = features.permute(2, 0, 1).contiguous()
        sequence, _ = self.sequence_model(sequence)
        return self.classifier(sequence)

    def output_lengths(self, image_widths: Tensor) -> Tensor:
        """Compute CTC time-step lengths from original padded image widths."""

        return torch.clamp(image_widths // self.time_downsample_factor, min=1)

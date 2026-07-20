"""Model definitions for CTC-based recognizers."""

from src.ctc.models.convnext import ConvNeXtCTCConfig, ConvNeXtCTCModel
from src.ctc.models.crnn import CRNNCTCConfig, CRNNCTCModel
from src.ctc.models.resnet import ResNetCTCConfig, ResNetCTCModel

__all__ = [
    "CRNNCTCConfig",
    "CRNNCTCModel",
    "ConvNeXtCTCConfig",
    "ConvNeXtCTCModel",
    "ResNetCTCConfig",
    "ResNetCTCModel",
]


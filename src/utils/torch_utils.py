"""Small PyTorch runtime helpers."""

from __future__ import annotations

import torch
from torch import nn


def should_use_data_parallel(
    device: torch.device,
    *,
    enabled: bool = True,
) -> bool:
    """Return whether DataParallel should be used for this runtime."""

    return enabled and device.type == "cuda" and torch.cuda.device_count() > 1


def maybe_wrap_data_parallel(
    model: nn.Module,
    device: torch.device,
    *,
    enabled: bool = True,
) -> nn.Module:
    """Wrap a model with DataParallel when multiple CUDA devices are available."""

    if should_use_data_parallel(device, enabled=enabled):
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs.")
        return nn.DataParallel(model)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when a model is wrapped by DataParallel."""

    if isinstance(model, nn.DataParallel):
        return unwrap_model(model.module)
    wrapped_model = getattr(model, "wrapped_model", None)
    if wrapped_model is not None:
        return unwrap_model(wrapped_model)
    return model

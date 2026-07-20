"""DataParallel helpers for CTC models."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.common.torch_utils import should_use_data_parallel


class CTCBatchFirstWrapper(nn.Module):
    """Return CTC logits as [batch, time, vocab] for DataParallel gathering."""

    def __init__(self, wrapped_model: nn.Module) -> None:
        super().__init__()
        self.wrapped_model = wrapped_model

    def forward(self, images: Tensor) -> Tensor:
        logits = self.wrapped_model(images)
        return logits.transpose(0, 1).contiguous()


def maybe_wrap_ctc_data_parallel(
    model: nn.Module,
    device: torch.device,
    *,
    enabled: bool = True,
) -> nn.Module:
    """Wrap a CTC model so DataParallel gathers over batch, not time."""

    if should_use_data_parallel(device, enabled=enabled):
        print(f"Using CTC DataParallel on {torch.cuda.device_count()} GPUs.")
        return nn.DataParallel(CTCBatchFirstWrapper(model))
    return model


def ctc_logits_to_time_first(logits: Tensor, model: nn.Module) -> Tensor:
    """Convert DataParallel batch-first CTC logits back to [time, batch, vocab]."""

    inner = model.module if isinstance(model, nn.DataParallel) else model
    if isinstance(inner, CTCBatchFirstWrapper):
        return logits.transpose(0, 1).contiguous()
    return logits

"""PyTorch 2.1-compatible AMP helpers."""

from contextlib import nullcontext

import torch


def make_grad_scaler(device, enabled):
    """Return a CUDA scaler, disabled for CPU or when the protocol disables AMP."""
    use_amp = bool(enabled) and torch.device(device).type == "cuda"
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(device, enabled):
    """Use the legacy CUDA AMP API supported by PyTorch 2.1."""
    if torch.device(device).type == "cuda":
        return torch.cuda.amp.autocast(enabled=bool(enabled))
    return nullcontext()

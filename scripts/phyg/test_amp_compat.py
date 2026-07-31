#!/usr/bin/env python3
"""Minimal disabled-AMP training step; never reads Development/Test/Eval."""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phyg.amp_compat import autocast_context, make_grad_scaler


def main():
    device = torch.device("cpu")
    model = torch.nn.Linear(3, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1, eta_min=1e-7
    )
    scaler = make_grad_scaler(device, enabled=False)
    if scaler.is_enabled():
        raise AssertionError("AMP scaler must be disabled")

    inputs = torch.tensor([[0.1, 0.2, 0.3]], device=device)
    targets = torch.tensor([[0.4, 0.5]], device=device)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, enabled=False):
        loss = torch.nn.functional.l1_loss(model(inputs), targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    if not torch.isfinite(loss):
        raise AssertionError("minimal training loss is not finite")
    if scheduler.last_epoch != 1:
        raise AssertionError("scheduler did not complete one step")
    print(json.dumps({
        "torch_version": torch.__version__,
        "pytorch_2_1_compatible_amp_api": "PASS",
        "scaler_enabled": scaler.is_enabled(),
        "forward_backward_optimizer_scheduler": "PASS",
        "formal_training_started": "NO",
        "official_test_participation": "NONE",
    }, indent=2))


if __name__ == "__main__":
    main()

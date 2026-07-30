#!/usr/bin/env python3
"""CPU tiny-model exact-resume integration test; no project data is accessed."""

import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phyg.checkpoint import REQUIRED_FIELDS, build_checkpoint, restore_checkpoint
from phyg.gamma_replay import GammaReplay
from phyg.provenance import config_sha256


def make_context(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-5, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=7, eta_min=1e-7
    )
    config = {"batch_size": 2, "seed": seed, "test": "tiny_cpu_only"}
    return {
        "model": model, "optimizer": optimizer, "scheduler": scheduler,
        "amp_scaler": None, "loader_generator":
            torch.Generator().manual_seed(seed),
        "replay": GammaReplay(seed), "config": config,
        "config_sha256": config_sha256(config),
        "split_raw_sha256": "tiny-no-data-raw",
        "split_canonical_sha256": "tiny-no-data-canonical",
        "initialization_checkpoint_sha256": "tiny-model-init",
        "git_commit": "tiny-test", "epoch": 1, "batch_in_epoch": 0,
        "global_step": 0, "epoch_order": [0, 1, 2, 3],
        "best_metric": None, "best_epoch": None, "best_step": None,
    }


def step(context):
    x = torch.rand(2, 4)
    y = torch.rand(2, 4)
    x, gamma = context["replay"](x)
    context["optimizer"].zero_grad()
    loss = torch.nn.functional.l1_loss(context["model"](x), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(context["model"].parameters(), 0.01)
    context["optimizer"].step()
    context["scheduler"].step()
    context["batch_in_epoch"] += 1
    context["global_step"] += 1
    return (float(loss.detach()), gamma, x.clone(), y.clone())


def run(context, count):
    return [step(context) for _ in range(count)]


def nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict) and left.keys() == right.keys()
            and all(nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left)) and len(left) == len(right)
            and all(nested_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def main():
    continuous = make_context()
    continuous_trace = run(continuous, 7)
    interrupted = make_context()
    prefix = run(interrupted, 4)
    state = build_checkpoint(interrupted)
    missing = REQUIRED_FIELDS - state.keys()
    if missing:
        raise AssertionError(f"checkpoint fields missing: {missing}")
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "trusted_complete.pth"
        torch.save(state, path)
        resumed = make_context(seed=999)
        for key in (
            "config", "config_sha256", "split_raw_sha256",
            "split_canonical_sha256", "initialization_checkpoint_sha256",
        ):
            resumed[key] = interrupted[key]
        restore_checkpoint(path, resumed, "cpu")
        suffix = run(resumed, 3)
    combined = prefix + suffix
    for expected, actual in zip(continuous_trace, combined):
        if expected[:2] != actual[:2]:
            raise AssertionError("loss/Gamma trace differs after resume")
        if not torch.equal(expected[2], actual[2]) or not torch.equal(
            expected[3], actual[3]
        ):
            raise AssertionError("input/target trace differs after resume")
    for expected, actual in zip(
        continuous["model"].state_dict().values(),
        resumed["model"].state_dict().values(),
    ):
        if not torch.equal(expected, actual):
            raise AssertionError("model state differs after resume")
    if not nested_equal(
        continuous["optimizer"].state_dict(), resumed["optimizer"].state_dict()
    ):
        raise AssertionError("optimizer state differs after resume")
    if not nested_equal(
        continuous["scheduler"].state_dict(), resumed["scheduler"].state_dict()
    ):
        raise AssertionError("scheduler state differs after resume")
    print("PASS: continuous 7 steps == 4 steps + trusted restore + 3 steps")
    print("PASS: required checkpoint fields and weights_only=False load path")
    print("official_test_participation=NONE")


if __name__ == "__main__":
    main()

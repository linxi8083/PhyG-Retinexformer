#!/usr/bin/env python3
"""CUDA exact-resume test using Retinexformer and Development Train only."""

import argparse
import json
import random
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from basicsr.models.archs import define_network
from basicsr.models.losses.losses import L1Loss
from phyg.checkpoint import restore_checkpoint, save_checkpoint
from phyg.initialization import load_strict_model_only
from phyg.provenance import git_commit, load_config, sha256_file
from scripts.phyg.train import (
    OrderedSampler,
    apply_replay,
    apply_shared_mixup,
    build_development,
    isolated_loader_iterator,
    make_replay,
    seed_everything,
)


def configure_determinism(device):
    if device.type != "cuda":
        torch.use_deterministic_algorithms(True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("real-stack resume test requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def make_context(config, split_raw, split_canonical, device):
    seed_everything(config["seed"])
    model = define_network(dict(config["network"])).to(device)
    initialization = load_strict_model_only(
        model, config["initialization"], config["initialization_sha256"], device
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["optimizer"]["lr"],
        betas=tuple(config["optimizer"]["betas"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["scheduler"]["T_max"],
        eta_min=config["scheduler"]["eta_min"],
    )
    context = {
        "model": model, "optimizer": optimizer, "scheduler": scheduler,
        "amp_scaler": None,
        "loader_generator": torch.Generator().manual_seed(config["seed"]),
        "iteration_generator":
            torch.Generator().manual_seed(config["seed"] + 1),
        "replay": make_replay(config, device), "config": config,
        "config_sha256": config["_config_sha256"],
        "split_raw_sha256": split_raw,
        "split_canonical_sha256": split_canonical,
        "initialization_checkpoint_sha256": initialization["sha256"],
        "git_commit": git_commit(ROOT), "epoch": 1, "batch_in_epoch": 0,
        "global_step": 0, "epoch_order": None, "best_metric": None,
        "best_epoch": None, "best_step": None,
    }
    return context


def next_iterator(context, dataset):
    config = context["config"]
    if context["epoch_order"] is None:
        context["epoch_order"] = torch.randperm(
            len(dataset), generator=context["loader_generator"]
        ).tolist()
        context["batch_in_epoch"] = 0
    usable = config["steps_per_epoch"] * config["batch_size"]
    order = context["epoch_order"][:usable]
    start = context["batch_in_epoch"] * config["batch_size"]
    return isolated_loader_iterator(
        dataset, batch_size=config["batch_size"],
        sampler=OrderedSampler(order[start:]), drop_last=True,
        iteration_generator=context["iteration_generator"],
    )


def train_steps(context, dataset, count, device):
    criterion = L1Loss(loss_weight=1.0, reduction="mean").to(device)
    trace = []
    iterator = next_iterator(context, dataset)
    for _ in range(count):
        low, gt, names = next(iterator)
        low, gt = low.to(device), gt.to(device)
        low, gt = apply_shared_mixup(low, gt, context["config"], device)
        low, degradation = apply_replay(
            context["config"]["replay"]["type"], context["replay"], low
        )
        context["optimizer"].zero_grad(set_to_none=True)
        output = context["model"](low)
        loss = criterion(output, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(context["model"].parameters(), 0.01)
        context["optimizer"].step()
        context["scheduler"].step()
        context["batch_in_epoch"] += 1
        context["global_step"] += 1
        trace.append({
            "names": list(names), "loss": float(loss.detach().cpu()),
            "degradation": deepcopy(degradation),
        })
    if device.type == "cuda":
        torch.cuda.synchronize()
    return trace


@torch.no_grad()
def preview_next(context, dataset, device):
    iterator = next_iterator(context, dataset)
    low, gt, names = next(iterator)
    low, gt = low.to(device), gt.to(device)
    low, gt = apply_shared_mixup(low, gt, context["config"], device)
    low, degradation = apply_replay(
        context["config"]["replay"]["type"], context["replay"], low
    )
    return {
        "names": list(names), "low": low.detach().cpu(),
        "gt": gt.detach().cpu(), "degradation": deepcopy(degradation),
    }


def nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(
            left.detach().cpu(), right.detach().cpu()
        )
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


def check_mode(config_path, total_steps, interrupt_step, device):
    config = load_config(config_path)
    dataset, _, split_raw, split_canonical = build_development(config)
    continuous = make_context(config, split_raw, split_canonical, device)
    continuous_trace = train_steps(
        continuous, dataset, total_steps, device
    )
    continuous_next = preview_next(continuous, dataset, device)
    interrupted = make_context(config, split_raw, split_canonical, device)
    prefix = train_steps(interrupted, dataset, interrupt_step, device)
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "mid_epoch.pth"
        save_checkpoint(checkpoint, interrupted)
        resumed = make_context(config, split_raw, split_canonical, device)
        restore_checkpoint(checkpoint, resumed, device)
        suffix = train_steps(
            resumed, dataset, total_steps - interrupt_step, device
        )
    resumed_next = preview_next(resumed, dataset, device)
    if continuous_trace != prefix + suffix:
        raise AssertionError("step trace differs after real-stack resume")
    for field in ("model", "optimizer", "scheduler"):
        left = (continuous[field].state_dict()
                if hasattr(continuous[field], "state_dict") else continuous[field])
        right = (resumed[field].state_dict()
                 if hasattr(resumed[field], "state_dict") else resumed[field])
        if not nested_equal(left, right):
            raise AssertionError(f"{field} differs after real-stack resume")
    if continuous["global_step"] != resumed["global_step"]:
        raise AssertionError("global_step differs after resume")
    if not nested_equal(continuous_next, resumed_next):
        raise AssertionError("next batch/degradation differs after resume")
    return {
        "config": str(config_path),
        "replay": config["replay"]["type"],
        "steps": total_steps,
        "interrupt_step": interrupt_step,
        "global_step": resumed["global_step"],
        "next_degradation": resumed_next["degradation"],
        "pass": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--interrupt-step", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 < args.interrupt_step < args.steps:
        raise ValueError("interrupt-step must be inside the test trajectory")
    device = torch.device(args.device)
    configure_determinism(device)
    results = [
        check_mode(
            "configs/phyg/gamma_replay_seed1234.yml",
            args.steps, args.interrupt_step, device,
        ),
        check_mode(
            "configs/phyg/physical_full_seed1234.yml",
            args.steps, args.interrupt_step, device,
        ),
    ]
    print(json.dumps({
        "results": results,
        "network": "RetinexFormer",
        "dataset": "LOLv2/Synthetic/Train Development 810 only",
        "crop_flip": True, "mixup": True, "cuda": device.type == "cuda",
        "official_test_participation": "NONE",
    }, indent=2))


if __name__ == "__main__":
    main()

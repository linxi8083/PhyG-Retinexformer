#!/usr/bin/env python3
"""Authorized 100-step CUDA benchmark on Development Train only."""

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from basicsr.models.archs import define_network
from basicsr.models.losses.losses import L1Loss
from phyg.initialization import load_strict_model_only
from phyg.provenance import load_config
from scripts.phyg.train import (
    OrderedSampler,
    apply_replay,
    apply_shared_mixup,
    build_development,
    isolated_loader_iterator,
    make_replay,
    seed_everything,
)


CONFIGS = {
    "identity": "configs/phyg/identity_seed1234.yml",
    "gamma_replay": "configs/phyg/gamma_replay_seed1234.yml",
    "physical_full": "configs/phyg/physical_full_seed1234.yml",
}


def benchmark_arm(name, config_path, steps=100):
    config = load_config(config_path)
    seed_everything(config["seed"])
    device = torch.device("cuda")
    dataset, _, _, _ = build_development(config)
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
    criterion = L1Loss(loss_weight=1.0, reduction="mean").to(device)
    replay = make_replay(config, device)
    order_generator = torch.Generator().manual_seed(config["seed"])
    iteration_generator = torch.Generator().manual_seed(config["seed"] + 1)
    order = torch.randperm(len(dataset), generator=order_generator).tolist()
    usable = config["steps_per_epoch"] * config["batch_size"]
    iterator = isolated_loader_iterator(
        dataset, batch_size=config["batch_size"],
        sampler=OrderedSampler(order[:usable]), drop_last=True,
        iteration_generator=iteration_generator,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    losses, elapsed = [], []
    nan = False
    oom = False
    model.train()
    try:
        for _ in range(steps):
            torch.cuda.synchronize()
            started = time.perf_counter()
            low, gt, _ = next(iterator)
            low, gt = low.to(device), gt.to(device)
            low, gt = apply_shared_mixup(low, gt, config, device)
            low, _ = apply_replay(config["replay"]["type"], replay, low)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(low), gt)
            if not torch.isfinite(loss):
                nan = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - started)
            losses.append(float(loss.detach().cpu()))
    except torch.cuda.OutOfMemoryError:
        oom = True
        torch.cuda.empty_cache()
    average = sum(elapsed) / len(elapsed) if elapsed else None
    return {
        "arm": name,
        "config": config_path,
        "steps_requested": steps,
        "steps_completed": len(elapsed),
        "average_step_seconds": average,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "final_loss": losses[-1] if losses else None,
        "nan": nan,
        "oom": oom,
        "initialization_sha256": initialization["sha256"],
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("100-step benchmark requires CUDA")
    arms = [
        benchmark_arm(name, path) for name, path in CONFIGS.items()
    ]
    if any(arm["oom"] or arm["nan"] or arm["steps_completed"] != 100
           for arm in arms):
        estimate = None
    else:
        seconds = {arm["arm"]: arm["average_step_seconds"] for arm in arms}
        estimate = {
            "identity_60epoch_seconds": seconds["identity"] * 6060,
            "gamma_60epoch_seconds": seconds["gamma_replay"] * 6060,
            "each_physical_60epoch_seconds": seconds["physical_full"] * 6060,
            "six_runs_total_seconds": (
                seconds["identity"] + seconds["gamma_replay"]
                + 4 * seconds["physical_full"]
            ) * 6060,
            "six_runs_total_hours": (
                seconds["identity"] + seconds["gamma_replay"]
                + 4 * seconds["physical_full"]
            ) * 6060 / 3600,
            "scope": "training steps only; six Development validations/run excluded",
        }
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "arms": arms,
        "estimate": estimate,
        "formal_training": False,
        "dataset": "LOLv2/Synthetic/Train Development 810 only",
        "official_test_participation": "NONE",
    }
    output = ROOT / "outputs/phyg_benchmark_seed1234.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

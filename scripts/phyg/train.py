#!/usr/bin/env python3
"""Retinexformer PhyG training entry. Never accepts an official Test/Eval path."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicsr.models.archs import define_network
from basicsr.models.losses.losses import L1Loss
from phyg.checkpoint import restore_checkpoint, save_checkpoint
from phyg.development_dataset import (
    DevelopmentPairDataset,
    audit_pairs,
    load_frozen_manifest,
)
from phyg.gamma_replay import GammaReplay
from phyg.initialization import load_strict_model_only
from phyg.metrics import AuthorValidationMetrics
from phyg.physical_degradation import PhysicalBatchAugmenter
from phyg.protocol import validate_protocol
from phyg.provenance import (
    assert_development_root,
    git_commit,
    load_config,
    sha256_file,
)


class OrderedSampler(Sampler):
    def __init__(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def isolated_loader_iterator(dataset, *, batch_size, sampler, drop_last,
                             iteration_generator):
    """Create an iterator without advancing global or persistent Torch RNG."""
    state = iteration_generator.get_state()
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=0, sampler=sampler,
        drop_last=drop_last, generator=iteration_generator,
    )
    iterator = iter(loader)
    iteration_generator.set_state(state)
    return iterator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--resume")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_development(config):
    root = assert_development_root(config["data_root"])
    sections, canonical_hash = load_frozen_manifest(config["split_manifest"])
    raw_hash = sha256_file(config["split_manifest"])
    if raw_hash != config["split_raw_sha256"]:
        raise ValueError("split raw SHA-256 mismatch")
    audit_pairs(root, sections, config["split_canonical_sha256"], canonical_hash)
    train = DevelopmentPairDataset(root, sections["train"], config["crop_size"])
    validation = DevelopmentPairDataset(root, sections["validation"], None)
    return train, validation, raw_hash, canonical_hash


def make_replay(config, device):
    replay_type = config["replay"]["type"]
    seed = config["seed"] + config["replay"]["rng_offset"]
    if replay_type == "identity":
        return None
    if replay_type == "gamma":
        gamma = config["replay"]["gamma"]
        return GammaReplay(seed, gamma["start_percent"], gamma["end_percent"])
    physical = config["replay"]["physical"]
    return PhysicalBatchAugmenter(
        seed, physical["mode"], physical["identity_probability"], device.type
    )


def apply_replay(replay_type, replay, low):
    if replay_type == "identity":
        return low, None
    return replay(low)


def apply_shared_mixup(low, gt, config, device):
    mixup = config["mixup"]
    if mixup["use_identity"] and random.randint(0, 1) == 1:
        return low, gt
    beta = torch.distributions.beta.Beta(
        torch.tensor([mixup["beta"]]), torch.tensor([mixup["beta"]])
    )
    lam = beta.rsample((1, 1)).item()
    permutation = torch.randperm(gt.size(0), device=device)
    return (
        lam * low + (1 - lam) * low[permutation],
        lam * gt + (1 - lam) * gt[permutation],
    )


def better(candidate, incumbent):
    if incumbent is None:
        return True
    return (
        candidate["psnr"], candidate["ssim"], -candidate["lpips"]
    ) > (
        incumbent["psnr"], incumbent["ssim"], -incumbent["lpips"]
    )


@torch.no_grad()
def validate_development(model, dataset, device, metrics):
    model.eval()
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    iteration_generator = torch.Generator().manual_seed(0)
    loader = isolated_loader_iterator(
        dataset, batch_size=1,
        sampler=OrderedSampler(range(len(dataset))), drop_last=False,
        iteration_generator=iteration_generator,
    )
    for low, gt, _ in loader:
        output = model(low.to(device)).clamp(0, 1)
        values = metrics.evaluate_tensor_pair(output, gt.to(device))
        for key in totals:
            totals[key] += values[key]
    return {key: value / len(dataset) for key, value in totals.items()}


def main():
    args = parse_args()
    config = load_config(args.config)
    validate_protocol(config)
    seed_everything(config["seed"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_set, val_set, split_raw, split_canonical = build_development(config)
    if len(train_set) != 810 or len(val_set) != 90:
        raise ValueError("Development split is not 810/90")
    model = define_network(dict(config["network"])).to(device)
    init_hash = sha256_file(config["initialization"])
    if not args.resume:
        initialization = load_strict_model_only(
            model, config["initialization"],
            config["initialization_sha256"], device,
        )
        if initialization["sha256"] != init_hash:
            raise RuntimeError("initialization changed while loading")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["optimizer"]["lr"],
        betas=tuple(config["optimizer"]["betas"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["scheduler"]["T_max"],
        eta_min=config["scheduler"]["eta_min"],
    )
    criterion = L1Loss(loss_weight=1.0, reduction="mean").to(device)
    amp_scaler = torch.amp.GradScaler(
        "cuda", enabled=config["amp"] and device.type == "cuda"
    )
    loader_generator = torch.Generator().manual_seed(config["seed"])
    iteration_generator = torch.Generator().manual_seed(config["seed"] + 1)
    replay = make_replay(config, device)
    context = {
        "model": model, "optimizer": optimizer, "scheduler": scheduler,
        "amp_scaler": amp_scaler, "loader_generator": loader_generator,
        "iteration_generator": iteration_generator,
        "replay": replay, "config": config,
        "config_sha256": config["_config_sha256"],
        "split_raw_sha256": split_raw,
        "split_canonical_sha256": split_canonical,
        "initialization_checkpoint_sha256": init_hash,
        "git_commit": git_commit(ROOT), "epoch": 1, "batch_in_epoch": 0,
        "global_step": 0, "epoch_order": None, "best_metric": None,
        "best_epoch": None, "best_step": None,
    }
    if args.resume:
        restore_checkpoint(args.resume, context, device)
    output_dir = Path(args.output_root) / config["experiment"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(json.dumps({
        "config": config,
        "config_sha256": context["config_sha256"],
        "split_raw_sha256": split_raw,
        "split_canonical_sha256": split_canonical,
        "initialization_checkpoint_sha256": init_hash,
        "git_commit": context["git_commit"],
        "official_test_participation": "NONE",
    }, indent=2), encoding="utf-8")
    metrics = AuthorValidationMetrics(device)
    while context["global_step"] < config["total_steps"]:
        if context["epoch_order"] is None:
            context["epoch_order"] = torch.randperm(
                len(train_set), generator=loader_generator
            ).tolist()
            context["batch_in_epoch"] = 0
        usable = config["steps_per_epoch"] * config["batch_size"]
        order = context["epoch_order"][:usable]
        start = context["batch_in_epoch"] * config["batch_size"]
        loader = isolated_loader_iterator(
            train_set, batch_size=config["batch_size"],
            sampler=OrderedSampler(order[start:]), drop_last=True,
            iteration_generator=iteration_generator,
        )
        model.train()
        for low, gt, _ in loader:
            low, gt = low.to(device), gt.to(device)
            low, gt = apply_shared_mixup(low, gt, config, device)
            low, _ = apply_replay(config["replay"]["type"], replay, low)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=config["amp"] and device.type == "cuda",
            ):
                loss = criterion(model(low), gt)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            scheduler.step()
            context["batch_in_epoch"] += 1
            context["global_step"] += 1
            if context["global_step"] % 100 == 0:
                save_checkpoint(output_dir / "latest.pth", context)
        completed_epoch = context["epoch"]
        context["epoch"] += 1
        context["batch_in_epoch"] = 0
        context["epoch_order"] = None
        is_best = False
        if completed_epoch % config["validation"]["interval_epochs"] == 0:
            values = validate_development(model, val_set, device, metrics)
            if better(values, context["best_metric"]):
                context["best_metric"] = values
                context["best_epoch"] = completed_epoch
                context["best_step"] = context["global_step"]
                is_best = True
        save_checkpoint(output_dir / "latest.pth", context)
        if completed_epoch % 10 == 0:
            save_checkpoint(output_dir / f"epoch_{completed_epoch}.pth", context)
        if is_best:
            save_checkpoint(output_dir / "best_validation.pth", context)


if __name__ == "__main__":
    main()

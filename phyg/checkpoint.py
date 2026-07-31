"""Complete, atomic and PyTorch-2.6-compatible experiment checkpoints."""

import os
import random
from pathlib import Path

import numpy as np
import torch


REQUIRED_FIELDS = {
    "model", "optimizer", "scheduler", "amp_scaler", "epoch",
    "batch_in_epoch", "global_step", "epoch_order", "sampler_position",
    "python_rng", "numpy_rng", "torch_cpu_rng", "torch_cuda_rng",
    "dataloader_generator", "dataloader_iteration_generator",
    "augmentation_rng", "replay_rng",
    "physical_noise_generator", "config", "config_sha256",
    "split_raw_sha256", "split_canonical_sha256",
    "initialization_checkpoint_sha256", "git_commit", "best_metric",
    "best_epoch", "best_step", "official_test_participation",
    "epoch_loss_sum", "epoch_loss_count", "epoch_elapsed_seconds",
}


def cpu_byte_rng_state(state):
    """Normalize an RNG state loaded on any device to CPU ByteTensor."""
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state)
    return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def capture_global_rng():
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_global_rng(state):
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(cpu_byte_rng_state(state["torch_cpu_rng"]))
    if state["torch_cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([
            cpu_byte_rng_state(cuda_state)
            for cuda_state in state["torch_cuda_rng"]
        ])


def build_checkpoint(context):
    rng = capture_global_rng()
    replay = context.get("replay")
    replay_state = replay.state_dict() if replay is not None else None
    physical_noise = None
    if replay is not None and hasattr(replay, "noise_generator"):
        physical_noise = replay.noise_generator.get_state()
    state = {
        "format_version": 1,
        "model": context["model"].state_dict(),
        "optimizer": context["optimizer"].state_dict(),
        "scheduler": context["scheduler"].state_dict(),
        "amp_scaler": (
            context["amp_scaler"].state_dict()
            if context.get("amp_scaler") is not None else None
        ),
        "epoch": context["epoch"],
        "batch_in_epoch": context["batch_in_epoch"],
        "global_step": context["global_step"],
        "epoch_order": context["epoch_order"],
        "sampler_position": context["batch_in_epoch"] * context["config"]["batch_size"],
        **rng,
        "dataloader_generator": context["loader_generator"].get_state(),
        "dataloader_iteration_generator":
            context["iteration_generator"].get_state(),
        "augmentation_rng": rng["python_rng"],
        "replay_rng": replay_state,
        "physical_noise_generator": physical_noise,
        "config": context["config"],
        "config_sha256": context["config_sha256"],
        "split_raw_sha256": context["split_raw_sha256"],
        "split_canonical_sha256": context["split_canonical_sha256"],
        "initialization_checkpoint_sha256": context[
            "initialization_checkpoint_sha256"
        ],
        "git_commit": context["git_commit"],
        "best_metric": context.get("best_metric"),
        "best_epoch": context.get("best_epoch"),
        "best_step": context.get("best_step"),
        "epoch_loss_sum": context.get("epoch_loss_sum", 0.0),
        "epoch_loss_count": context.get("epoch_loss_count", 0),
        "epoch_elapsed_seconds": context.get("epoch_elapsed_seconds", 0.0),
        "official_test_participation": "NONE",
    }
    missing = REQUIRED_FIELDS - state.keys()
    if missing:
        raise ValueError(f"checkpoint missing fields: {sorted(missing)}")
    return state


def save_checkpoint(path, context):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(build_checkpoint(context), temporary)
    os.replace(temporary, path)


def trusted_torch_load(path, map_location="cpu"):
    """Only for this project’s SHA-verified, trusted complete checkpoints."""
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_checkpoint(path, context, map_location="cpu"):
    state = trusted_torch_load(path, map_location)
    missing = REQUIRED_FIELDS - state.keys()
    if missing:
        raise ValueError(f"checkpoint missing fields: {sorted(missing)}")
    for key in (
        "config_sha256", "split_raw_sha256", "split_canonical_sha256",
        "initialization_checkpoint_sha256",
    ):
        if state[key] != context[key]:
            raise ValueError(f"resume {key} mismatch")
    if state["official_test_participation"] != "NONE":
        raise ValueError("checkpoint participated in official Test/Eval")
    context["model"].load_state_dict(state["model"], strict=True)
    context["optimizer"].load_state_dict(state["optimizer"])
    context["scheduler"].load_state_dict(state["scheduler"])
    if context.get("amp_scaler") is not None and state["amp_scaler"] is not None:
        context["amp_scaler"].load_state_dict(state["amp_scaler"])
    restore_global_rng(state)
    context["loader_generator"].set_state(
        cpu_byte_rng_state(state["dataloader_generator"])
    )
    context["iteration_generator"].set_state(
        cpu_byte_rng_state(state["dataloader_iteration_generator"])
    )
    if context.get("replay") is not None:
        context["replay"].load_state_dict(state["replay_rng"])
    for key in (
        "epoch", "batch_in_epoch", "global_step", "epoch_order",
        "best_metric", "best_epoch", "best_step",
        "epoch_loss_sum", "epoch_loss_count", "epoch_elapsed_seconds",
    ):
        context[key] = state[key]
    return state

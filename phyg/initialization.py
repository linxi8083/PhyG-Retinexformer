"""Strict, non-repairing validation of the official Retinexformer weights."""

from pathlib import Path

import torch

from phyg.provenance import sha256_file


def load_strict_model_only(model, path, expected_sha256, map_location="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"official initialization missing: {path}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(
            f"official initialization SHA-256 mismatch: {digest}"
        )
    checkpoint = torch.load(
        path, map_location=map_location, weights_only=True
    )
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"params"}:
        raise ValueError(
            "official initialization must be an exact {'params': state_dict}"
        )
    state = checkpoint["params"]
    if not isinstance(state, dict):
        raise TypeError("official params is not a state_dict")
    expected = model.state_dict()
    if state.keys() != expected.keys():
        missing = sorted(expected.keys() - state.keys())
        unexpected = sorted(state.keys() - expected.keys())
        raise RuntimeError(
            f"strict key mismatch; missing={missing}, unexpected={unexpected}"
        )
    for key in expected:
        if state[key].shape != expected[key].shape:
            raise RuntimeError(
                f"strict shape mismatch for {key}: "
                f"{tuple(state[key].shape)} != {tuple(expected[key].shape)}"
            )
        if state[key].dtype != expected[key].dtype:
            raise RuntimeError(
                f"strict dtype mismatch for {key}: "
                f"{state[key].dtype} != {expected[key].dtype}"
            )
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load returned incompatibility: {incompatible}")
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "parameter_tensors": len(state),
        "parameter_values": sum(tensor.numel() for tensor in state.values()),
        "strict": True,
        "renamed": False,
        "dropped": False,
        "repaired": False,
    }

"""Shared validation for frozen Retinexformer robustness evaluation."""

import hashlib
import json
from pathlib import Path

import torch

from basicsr.models.archs import define_network
from phyg.checkpoint import trusted_torch_load
from phyg.development_dataset import audit_pairs, load_frozen_manifest
from phyg.provenance import sha256_file
from phyg.robustness_protocol import (
    PROTOCOLS,
    apply_frozen_degradation,
    canonical_protocol_json,
    preset_seed,
)


MODEL_NAMES = (
    "identity", "gamma_replay", "physical_exposure",
    "physical_exposure_noise", "physical_exposure_color", "physical_full",
)


def reject_forbidden_path(path):
    resolved = Path(path).resolve()
    forbidden = [part for part in resolved.parts
                 if "test" in part.lower() or "eval" in part.lower()]
    if forbidden:
        raise ValueError(f"Test/Eval path is forbidden: {resolved}")
    return resolved


def load_robustness_config(path):
    path = reject_forbidden_path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("official_test_participation") != "NONE":
        raise ValueError("official_test_participation must be NONE")
    if tuple(config["checkpoints"]) != MODEL_NAMES:
        raise ValueError("checkpoint arms or ordering differ from frozen matrix")
    config["_path"] = str(path)
    config["_sha256"] = sha256_file(path)
    return config


def load_validation(config):
    root = reject_forbidden_path(config["data_root"])
    manifest = reject_forbidden_path(config["split_manifest"])
    sections, canonical = load_frozen_manifest(manifest)
    raw = sha256_file(manifest)
    if raw != config["split_raw_sha256"]:
        raise ValueError("split raw SHA-256 mismatch")
    audit_pairs(root, sections, config["split_canonical_sha256"], canonical)
    return root, sections["validation"], raw, canonical


def checkpoint_audit(config, strict_models=True):
    records = {}
    for name in MODEL_NAMES:
        spec = config["checkpoints"][name]
        path = reject_forbidden_path(spec["path"])
        actual = sha256_file(path)
        if actual != spec["sha256"]:
            raise ValueError(f"{name} checkpoint SHA-256 mismatch: {actual}")
        state = trusted_torch_load(path, "cpu")
        if state.get("official_test_participation") != "NONE":
            raise ValueError(f"{name} checkpoint used official Test/Eval")
        for key in ("split_raw_sha256", "split_canonical_sha256"):
            if state.get(key) != config[key]:
                raise ValueError(f"{name} {key} mismatch")
            if state.get("config", {}).get(key) != config[key]:
                raise ValueError(f"{name} embedded config {key} mismatch")
        if state.get("config", {}).get("split_manifest") != config["split_manifest"]:
            raise ValueError(f"{name} split manifest path mismatch")
        if strict_models:
            model = define_network(dict(config["network"]))
            model.load_state_dict(state["model"], strict=True)
            del model
        records[name] = {"path": str(path), "sha256": actual}
    return records


def protocol_sha256():
    return hashlib.sha256(canonical_protocol_json().encode("utf-8")).hexdigest()


def assert_shared_degradations(image, seed, image_index=0):
    """Regenerate the same instance for all six arms and require bit identity."""
    signatures = {}
    for protocol, presets in PROTOCOLS.items():
        signatures[protocol] = {}
        for preset_name, parameters in presets.items():
            reference = None
            for _model_name in MODEL_NAMES:
                generator = torch.Generator(device=image.device)
                generator.manual_seed(
                    preset_seed(seed, image_index, protocol, preset_name)
                )
                degraded = apply_frozen_degradation(
                    image, generator=generator, **parameters
                )
                if reference is None:
                    reference = degraded
                elif not torch.equal(reference, degraded):
                    raise AssertionError("degradation differs across model arms")
            signatures[protocol][preset_name] = hashlib.sha256(
                reference.cpu().numpy().tobytes()
            ).hexdigest()
    return signatures

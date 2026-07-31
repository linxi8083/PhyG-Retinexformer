#!/usr/bin/env python3
"""CPU-only preflight for frozen robustness evaluation; reads Development only."""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phyg.robustness_evaluation import (
    assert_shared_degradations,
    checkpoint_audit,
    load_robustness_config,
    load_validation,
    protocol_sha256,
    reject_forbidden_path,
)
from phyg.robustness_protocol import PROTOCOLS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/phyg/robustness_seed1234.json"
    )
    args = parser.parse_args()
    config = load_robustness_config(args.config)
    root, filenames, raw, canonical = load_validation(config)
    checkpoints = checkpoint_audit(config, strict_models=True)
    image = TF.to_tensor(
        Image.open(root / "Low" / filenames[0]).convert("RGB")
    ).unsqueeze(0)
    signatures = assert_shared_degradations(image, config["seed"])
    if len(filenames) != 90:
        raise ValueError("evaluation must use exactly 90 Development Validation images")
    expected_names = {
        "seen": ("identity", "bright_clean", "dark_noisy",
                 "warm_camera", "cool_camera"),
        "unseen": ("identity", "bright_cool_cross", "dark_warm_cross",
                   "neutral_new_ccm", "neutral_high_iso"),
    }
    if {key: tuple(value) for key, value in PROTOCOLS.items()} != expected_names:
        raise ValueError("Seen/Unseen preset names or ordering changed")
    for forbidden in ("datasets/LOLv2/Synthetic/Test",
                      "datasets/LOLv2/Synthetic/Eval"):
        try:
            reject_forbidden_path(forbidden)
        except ValueError:
            pass
        else:
            raise AssertionError(f"path guard accepted {forbidden}")
    print(json.dumps({
        "status": "PASS", "device": "cpu", "validation_images": 90,
        "split_raw_sha256": raw, "split_canonical_sha256": canonical,
        "config_sha256": config["_sha256"],
        "checkpoint_sha256": checkpoints,
        "protocol_sha256": protocol_sha256(),
        "preset_names": {key: list(value) for key, value in PROTOCOLS.items()},
        "shared_instance_sha256_first_image": signatures,
        "all_six_arms_same_split": True,
        "all_six_arms_same_degradation_instances": True,
        "test_eval_path_guard": "PASS",
        "official_test_participation": "NONE",
    }, indent=2))


if __name__ == "__main__":
    main()

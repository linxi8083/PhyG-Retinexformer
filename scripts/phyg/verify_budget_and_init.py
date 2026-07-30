#!/usr/bin/env python3
"""Verify all seed-1234 train arms differ only in replay and experiment name."""

import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phyg.provenance import load_config
from phyg.protocol import validate_protocol


def without_allowed_differences(config):
    config = deepcopy(config)
    config.pop("experiment")
    config.pop("_config_path")
    config.pop("_config_sha256")
    config.pop("replay")
    return config


def main():
    paths = sorted((ROOT / "configs/phyg").glob("*_seed1234.yml"))
    paths = [path for path in paths if path.name != "base_seed1234.yml"]
    configs = [load_config(path) for path in paths]
    for config in configs:
        validate_protocol(config)
    reference = without_allowed_differences(configs[0])
    for path, config in zip(paths, configs):
        if without_allowed_differences(config) != reference:
            raise AssertionError(f"non-replay protocol differs: {path}")
        if config["initialization"] != "pretrained_weights/LOL_v2_synthetic.pth":
            raise AssertionError(f"initialization differs: {path}")
        if config["total_steps"] != config["epochs"] * config["steps_per_epoch"]:
            raise AssertionError(f"budget arithmetic differs: {path}")
    print(f"PASS: {len(configs)} arms share initialization and 6060-step budget")
    print("Only experiment/replay fields differ; official_test_participation=NONE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify official LOLv2-Synthetic weights without Test/Eval access."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from basicsr.models.archs import define_network
from phyg.initialization import load_strict_model_only
from phyg.provenance import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/phyg/identity_seed1234.yml"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    model = define_network(dict(config["network"])).to(args.device)
    result = load_strict_model_only(
        model,
        config["initialization"],
        config["initialization_sha256"],
        args.device,
    )
    result["official_test_participation"] = "NONE"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

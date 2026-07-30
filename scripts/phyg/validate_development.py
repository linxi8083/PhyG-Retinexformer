#!/usr/bin/env python3
"""Development-only validation; rejects every Test/Eval path."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phyg.checkpoint import trusted_torch_load
from phyg.provenance import assert_development_root, load_config
from scripts.phyg.train import (
    build_development,
    load_official_initialization,
    validate_development,
)
from basicsr.models.archs import define_network
from phyg.metrics import AuthorValidationMetrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--official-initialization", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    assert_development_root(config["data_root"])
    _, validation, _, _ = build_development(config)
    device = torch.device(args.device)
    model = define_network(dict(config["network"])).to(device)
    if args.official_initialization:
        load_official_initialization(model, config["initialization"], device)
    else:
        state = trusted_torch_load(args.checkpoint, device)
        if state["official_test_participation"] != "NONE":
            raise ValueError("checkpoint is not Development-only")
        model.load_state_dict(state["model"], strict=True)
    print(validate_development(
        model, validation, device, AuthorValidationMetrics(device)
    ))


if __name__ == "__main__":
    main()

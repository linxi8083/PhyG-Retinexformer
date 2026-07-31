#!/usr/bin/env python3
"""Prove DataLoader iterator reconstruction consumes no persistent Torch RNG."""

import sys
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.phyg.train import OrderedSampler, isolated_loader_iterator


def main():
    dataset = TensorDataset(torch.arange(16))
    global_before = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(1235)
    generator_before = generator.get_state().clone()
    iterator = isolated_loader_iterator(
        dataset, batch_size=4, sampler=OrderedSampler(range(16)),
        drop_last=True, iteration_generator=generator,
    )
    if not torch.equal(global_before, torch.get_rng_state()):
        raise AssertionError("DataLoader reconstruction consumed global Torch RNG")
    if not torch.equal(generator_before, generator.get_state()):
        raise AssertionError("DataLoader reconstruction advanced persistent generator")
    if not torch.equal(next(iterator)[0], torch.arange(4)):
        raise AssertionError("isolated DataLoader changed sample order")
    print("PASS: DataLoader iterator reconstruction is Torch-RNG neutral")
    print("official_test_participation=NONE")


if __name__ == "__main__":
    main()

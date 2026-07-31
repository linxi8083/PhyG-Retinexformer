#!/usr/bin/env python3
"""Static/CPU-only preflight. Does not enumerate any dataset directory."""

import argparse
import random
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phyg.gamma_replay import GammaReplay
from phyg.physical_degradation import PhysicalBatchAugmenter
from phyg.protocol import validate_protocol
from phyg.provenance import (
    assert_development_root,
    load_config,
    sha256_file,
    verify_sha256_manifest,
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def author_integrity():
    require(subprocess.check_output(
        ["git", "rev-list", "-n", "1", "retinexformer-official-frozen"],
        cwd=ROOT, text=True,
    ).strip() == "e8086f9d6e4badf0eebf48825264a785d3aa541e", "tag moved")
    require(subprocess.check_output(
        ["git", "rev-parse", "author-baseline"], cwd=ROOT, text=True
    ).strip() == "e8086f9d6e4badf0eebf48825264a785d3aa541e",
            "author-baseline moved")
    forbidden = subprocess.check_output([
        "git", "diff", "--name-only", "retinexformer-official-frozen", "--",
        "basicsr/models/archs/RetinexFormer_arch.py",
        "Enhancement/test_from_dataset.py", "basicsr/test.py",
    ], cwd=ROOT, text=True).strip()
    require(not forbidden, f"forbidden official files changed: {forbidden}")


def repository_completeness():
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "basicsr/data"], cwd=ROOT, text=True
    ).splitlines())
    local = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "basicsr/data").rglob("*") if path.is_file()
    }
    require(local == tracked, "basicsr/data tracked/local file sets differ")
    ignored = subprocess.run(
        ["git", "check-ignore", "basicsr/data/paired_image_dataset.py"],
        cwd=ROOT, capture_output=True,
    )
    require(ignored.returncode == 1, "basicsr/data is still ignored")


def source_equivalence():
    records = verify_sha256_manifest(
        ROOT, ROOT / "configs/phyg/protocols/source_sha256.txt"
    )
    require(len(records) == 4, "protocol SHA-256 manifest must contain four files")


def split_and_guard(config):
    lines = Path(config["split_manifest"]).read_text(encoding="utf-8").splitlines()
    validation_index = lines.index("[validation]")
    sections = {
        "train": lines[1:validation_index],
        "validation": lines[validation_index + 1:],
    }
    import hashlib
    canonical_text = "\n".join(
        ["[train]", *sections["train"], "[validation]", *sections["validation"]]
    )
    canonical = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    require(len(sections["train"]) == 810, "train split is not 810")
    require(len(sections["validation"]) == 90, "validation split is not 90")
    require(not set(sections["train"]) & set(sections["validation"]),
            "split overlap")
    require(canonical == config["split_canonical_sha256"], "canonical hash")
    require(sha256_file(config["split_manifest"]) == config["split_raw_sha256"],
            "raw hash")
    assert_development_root("sandbox/LOLv2/Synthetic/Train")
    for forbidden in ("sandbox/LOLv2/Synthetic/Test",
                      "sandbox/LOLv2/Real/Eval"):
        try:
            assert_development_root(forbidden)
        except ValueError:
            pass
        else:
            raise AssertionError(f"path guard accepted {forbidden}")


def replay_equivalence(config):
    seed = config["seed"]
    gamma = GammaReplay(seed, 60, 120)
    reference_rng = random.Random(seed)
    batch = torch.linspace(0.01, 1.0, 96).reshape(2, 3, 4, 4)
    output, sampled = gamma(batch)
    expected_gamma = reference_rng.randint(60, 120) / 100.0
    require(sampled == expected_gamma, "Gamma draw differs")
    require(torch.equal(output, batch ** expected_gamma), "Gamma numeric differs")
    a = PhysicalBatchAugmenter(seed, "full", 0.5, "cpu")
    b = PhysicalBatchAugmenter(seed, "full", 0.5, "cpu")
    out_a, params_a = a(batch)
    out_b, params_b = b(batch)
    require(params_a == params_b and torch.equal(out_a, out_b),
            "Physical deterministic replay differs")
    state = a.state_dict()
    expected, expected_params = a(batch)
    resumed = PhysicalBatchAugmenter(999, "full", 0.5, "cpu")
    resumed.load_state_dict(state)
    actual, actual_params = resumed(batch)
    require(expected_params == actual_params and torch.equal(expected, actual),
            "Physical restored trajectory differs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/phyg/physical_full_seed1234.yml"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    author_integrity()
    repository_completeness()
    source_equivalence()
    validate_protocol(config)
    split_and_guard(config)
    replay_equivalence(config)
    print("PASS: author integrity")
    print("PASS: repository completeness")
    print("PASS: repository protocol SHA-256 and Physical/Gamma CPU equivalence")
    print("PASS: split count/nesting/hashes and Test/Eval path guard")
    print("official_test_participation=NONE")


if __name__ == "__main__":
    main()

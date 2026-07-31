#!/usr/bin/env python3
"""Unit tests for experiment logging; performs no model training."""

import json
import os
import random
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phyg.experiment_logging import (
    EPOCH_FIELDS,
    SUMMARY_FIELDS,
    ExperimentLogger,
    make_summary,
    replay_statistics,
)
from phyg.gamma_replay import GammaReplay
from phyg.physical_degradation import PhysicalBatchAugmenter


def rng_snapshot(gamma, physical):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "gamma": gamma.state_dict(),
        "physical": physical.state_dict(),
    }


def assert_rng_equal(left, right):
    assert left["python"] == right["python"]
    assert left["numpy"][0] == right["numpy"][0]
    assert np.array_equal(left["numpy"][1], right["numpy"][1])
    assert left["numpy"][2:] == right["numpy"][2:]
    assert torch.equal(left["torch"], right["torch"])
    assert left["gamma"] == right["gamma"]
    assert left["physical"]["parameter_rng"] == right["physical"]["parameter_rng"]
    assert torch.equal(
        left["physical"]["noise_generator"],
        right["physical"]["noise_generator"],
    )
    assert left["physical"]["counts"] == right["physical"]["counts"]


def main():
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    gamma = GammaReplay(4)
    physical = PhysicalBatchAugmenter(5, mode="full", device="cpu")
    gamma(torch.full((1, 3, 2, 2), 0.5))
    physical(torch.full((1, 3, 2, 2), 0.5))
    before = rng_snapshot(gamma, physical)

    with tempfile.TemporaryDirectory() as temporary:
        logger = ExperimentLogger(temporary)
        base = {
            "global_step": 101,
            "mean_train_loss": 0.25,
            "learning_rate": 9.9e-6,
            "elapsed_seconds": 1.5,
        }
        assert logger.append_epoch({"epoch": 1, **base})
        assert logger.append_epoch({"epoch": 2, **base})
        assert not ExperimentLogger(temporary).append_epoch(
            {"epoch": 2, **base}
        )
        lines = Path(temporary, "metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        assert [json.loads(line)["epoch"] for line in lines] == [1, 2]
        assert EPOCH_FIELDS <= json.loads(lines[0]).keys()

        context = {
            "config_sha256": "config", "split_raw_sha256": "raw",
            "split_canonical_sha256": "canonical",
            "initialization_checkpoint_sha256": "initialization",
            "git_commit": "commit", "best_epoch": 10, "best_step": 1010,
            "best_metric": {"psnr": 20.0, "ssim": 0.8, "lpips": 0.2},
            "epoch": 11, "global_step": 1010,
        }
        summary = make_summary(context, False)
        with patch(
            "phyg.experiment_logging.os.replace", wraps=os.replace
        ) as atomic_replace:
            logger.update_summary(summary)
        atomic_replace.assert_called_once_with(
            logger.summary_path.with_suffix(".json.tmp"),
            logger.summary_path,
        )
        assert SUMMARY_FIELDS <= json.loads(
            logger.summary_path.read_text(encoding="utf-8")
        ).keys()
        context["epoch"] = 12
        logger.update_summary(make_summary(context, True))
        assert not logger.summary_path.with_suffix(".json.tmp").exists()
        assert json.loads(
            logger.summary_path.read_text(encoding="utf-8")
        )["completed"]

        assert replay_statistics("gamma", gamma, 1) == {
            "gamma_call_count": 1, "last_gamma": gamma.last_gamma,
        }
        counts = replay_statistics("physical", physical, 1)[
            "physical_sample_counts"
        ]
        assert set(counts) == {
            "identity", "bright_clean", "dark_noisy",
            "warm_camera", "cool_camera",
        }

    after = rng_snapshot(gamma, physical)
    assert_rng_equal(before, after)
    print(json.dumps({
        "append": "PASS",
        "resume_no_duplicate_epoch": "PASS",
        "atomic_summary": "PASS",
        "json_fields": "PASS",
        "rng_unchanged": "PASS",
        "official_test_participation": "NONE",
    }, indent=2))


if __name__ == "__main__":
    main()

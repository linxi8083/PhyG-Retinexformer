"""Frozen cross-backbone protocol assertions without training-stack imports."""


def validate_protocol(config):
    expected = {
        "seed": 1234, "batch_size": 8, "crop_size": 128, "epochs": 60,
        "steps_per_epoch": 101, "total_steps": 6060, "drop_last": True,
        "num_workers": 0, "official_test_participation": "NONE",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"frozen protocol mismatch: {key}")
    if config["optimizer"] != {
        "type": "Adam", "lr": 1e-5, "betas": [0.9, 0.999]
    }:
        raise ValueError("optimizer protocol mismatch")
    if config["scheduler"] != {
        "type": "CosineAnnealingLR", "T_max": 6060, "eta_min": 1e-7
    }:
        raise ValueError("scheduler protocol mismatch")
    if config["loss"] != {
        "type": "L1Loss", "loss_weight": 1.0, "reduction": "mean"
    }:
        raise ValueError("loss protocol mismatch")
    if config["gradient_clip"] != {"enabled": True, "max_norm": 0.01}:
        raise ValueError("gradient clip mismatch")
    if not config["mixup"]["enabled"] or config["validation"]["gt_mean_correction"]:
        raise ValueError("mixup/GT-mean protocol mismatch")
    if config["replay"]["type"] not in {"identity", "gamma", "physical"}:
        raise ValueError("unknown replay type")

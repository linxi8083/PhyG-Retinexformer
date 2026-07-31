"""Frozen Seen/Unseen robustness protocol migrated from HVI-HAFB-physical."""

import json

import torch


SOURCE_PHYSICAL_SHA256 = (
    "e2fef7c26352e29c8c30fea7d86a24f953a2f0cc3a33178502206a8101c181da"
)
SOURCE_EVALUATOR_SHA256 = (
    "8056591328eb36c44c2a865b3b4c496ed022346312688e333e8333b84f3261da"
)

SEEN_PRESETS = {
    "identity": {"exposure_ev": 0.0, "shot_noise_level": 0.0,
                 "read_noise_std": 0.0},
    "bright_clean": {"exposure_ev": 0.75,
                     "shot_noise_level": 1.0 / 2048.0,
                     "read_noise_std": 0.001},
    "dark_noisy": {"exposure_ev": -0.5,
                   "shot_noise_level": 1.0 / 1024.0,
                   "read_noise_std": 0.0025},
    "warm_camera": {
        "exposure_ev": 0.25, "white_balance": (1.08, 1.00, 0.92),
        "color_matrix": ((1.03, -0.02, -0.01),
                         (-0.01, 1.02, -0.01),
                         (-0.01, -0.03, 1.04)),
        "shot_noise_level": 1.0 / 1024.0, "read_noise_std": 0.002,
    },
    "cool_camera": {
        "exposure_ev": -0.25, "white_balance": (0.93, 1.00, 1.08),
        "color_matrix": ((1.02, -0.01, -0.01),
                         (-0.02, 1.03, -0.01),
                         (-0.01, -0.02, 1.03)),
        "shot_noise_level": 1.0 / 768.0, "read_noise_std": 0.003,
    },
}

UNSEEN_PRESETS = {
    "identity": SEEN_PRESETS["identity"],
    "bright_cool_cross": {
        "exposure_ev": 0.45, "white_balance": (0.96, 1.00, 1.05),
        "color_matrix": ((1.015, -0.005, -0.010),
                         (-0.010, 1.020, -0.010),
                         (-0.005, -0.015, 1.020)),
        "shot_noise_level": 1.0 / 1600.0, "read_noise_std": 0.0015,
    },
    "dark_warm_cross": {
        "exposure_ev": -0.35, "white_balance": (1.05, 1.00, 0.95),
        "color_matrix": ((1.020, -0.015, -0.005),
                         (-0.005, 1.015, -0.010),
                         (-0.010, -0.015, 1.025)),
        "shot_noise_level": 1.0 / 900.0, "read_noise_std": 0.0022,
    },
    "neutral_new_ccm": {
        "exposure_ev": 0.0, "white_balance": (1.02, 0.99, 1.01),
        "color_matrix": ((1.025, -0.020, -0.005),
                         (-0.015, 1.025, -0.010),
                         (-0.005, -0.025, 1.030)),
        "shot_noise_level": 1.0 / 1100.0, "read_noise_std": 0.0020,
    },
    "neutral_high_iso": {"exposure_ev": -0.15,
                         "shot_noise_level": 1.0 / 640.0,
                         "read_noise_std": 0.0035},
}

PROTOCOLS = {"seen": SEEN_PRESETS, "unseen": UNSEEN_PRESETS}


def srgb_to_linear(image):
    image = image.clamp(0, 1)
    return torch.where(image <= 0.04045, image / 12.92,
                       ((image + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(image):
    image = image.clamp_min(0)
    return torch.where(image <= 0.0031308, image * 12.92,
                       1.055 * image.pow(1.0 / 2.4) - 0.055).clamp(0, 1)


def apply_frozen_degradation(image, generator=None, exposure_ev=0.0,
                             white_balance=(1.0, 1.0, 1.0),
                             color_matrix=None, shot_noise_level=0.0,
                             read_noise_std=0.0, quantization_bits=8):
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected an NCHW RGB tensor.")
    linear = srgb_to_linear(image) * (2.0 ** float(exposure_ev))
    if shot_noise_level > 0:
        photon_count = 1.0 / float(shot_noise_level)
        linear = torch.poisson(linear * photon_count,
                               generator=generator) / photon_count
    if read_noise_std > 0:
        noise = torch.randn(linear.shape, dtype=linear.dtype,
                            device=linear.device, generator=generator)
        linear = linear + noise * float(read_noise_std)
    linear = linear.clamp_min(0)
    linear = linear * image.new_tensor(white_balance).view(1, 3, 1, 1)
    if color_matrix is not None:
        linear = torch.einsum(
            "ij,bjhw->bihw", image.new_tensor(color_matrix).view(3, 3), linear
        )
    output = linear_to_srgb(linear.clamp(0, 1))
    if quantization_bits > 0:
        levels = float((1 << quantization_bits) - 1)
        output = torch.round(output * levels) / levels
    return output.clamp(0, 1)


def preset_seed(seed, image_index, protocol, preset_name):
    presets = PROTOCOLS[protocol]
    return int(seed) + image_index * len(presets) + list(presets).index(preset_name)


def canonical_protocol_json():
    return json.dumps(PROTOCOLS, sort_keys=True, separators=(",", ":"))

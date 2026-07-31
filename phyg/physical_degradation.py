"""Exact external replay of the frozen Physical Stage-1 protocol-v1."""

from dataclasses import asdict, dataclass
from typing import Optional
import random

import torch

IDENTITY_MATRIX = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
FAMILIES = ("bright_clean", "dark_noisy", "warm_camera", "cool_camera")
WARM_MATRIX = ((1.03, -0.02, -0.01), (-0.01, 1.02, -0.01), (-0.01, -0.03, 1.04))
COOL_MATRIX = ((1.02, -0.01, -0.01), (-0.02, 1.03, -0.01), (-0.01, -0.02, 1.03))


def srgb_to_linear(image):
    image = image.clamp(0, 1)
    return torch.where(image <= 0.04045, image / 12.92,
                       ((image + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(image):
    image = image.clamp_min(0)
    return torch.where(image <= 0.0031308, image * 12.92,
                       1.055 * image.pow(1.0 / 2.4) - 0.055).clamp(0, 1)


def interpolate_matrix(target, strength):
    return tuple(tuple(
        IDENTITY_MATRIX[row][column]
        + strength * (target[row][column] - IDENTITY_MATRIX[row][column])
        for column in range(3)
    ) for row in range(3))


@dataclass(frozen=True)
class SampledParameters:
    family: str
    strength: float
    exposure_ev: float = 0.0
    shot_noise_level: float = 0.0
    read_noise_std: float = 0.0
    white_balance: tuple = (1.0, 1.0, 1.0)
    color_matrix: tuple = IDENTITY_MATRIX
    quantization_bits: int = 8

    def to_dict(self):
        return asdict(self)


def sample_frozen_parameters(parameter_rng, mode="full", identity_probability=0.5):
    if mode not in {"identity", "exposure", "exposure_noise", "exposure_color", "full"}:
        raise ValueError(f"unsupported Physical mode: {mode}")
    if mode == "identity" or parameter_rng.random() < identity_probability:
        return SampledParameters("identity", 0.0, quantization_bits=0)
    family = parameter_rng.choice(FAMILIES)
    strength = parameter_rng.uniform(0.5, 1.0)
    if family == "bright_clean":
        values = dict(exposure_ev=0.75 * strength,
                      shot_noise_level=strength / 2048.0,
                      read_noise_std=0.001 * strength)
    elif family == "dark_noisy":
        values = dict(exposure_ev=-0.5 * strength,
                      shot_noise_level=strength / 1024.0,
                      read_noise_std=0.0025 * strength)
    elif family == "warm_camera":
        values = dict(exposure_ev=0.25 * strength,
                      white_balance=(1.0 + 0.08 * strength, 1.0, 1.0 - 0.08 * strength),
                      color_matrix=interpolate_matrix(WARM_MATRIX, strength),
                      shot_noise_level=strength / 1024.0,
                      read_noise_std=0.002 * strength)
    else:
        values = dict(exposure_ev=-0.25 * strength,
                      white_balance=(1.0 - 0.07 * strength, 1.0, 1.0 + 0.08 * strength),
                      color_matrix=interpolate_matrix(COOL_MATRIX, strength),
                      shot_noise_level=strength / 768.0,
                      read_noise_std=0.003 * strength)
    if mode == "exposure":
        values = {"exposure_ev": values["exposure_ev"]}
    elif mode == "exposure_noise":
        values = {key: values[key] for key in
                  ("exposure_ev", "shot_noise_level", "read_noise_std")}
    elif mode == "exposure_color":
        values = {key: value for key, value in values.items()
                  if key not in ("shot_noise_level", "read_noise_std")}
    return SampledParameters(family, strength, **values)


def apply_physical_degradation(image, parameters, noise_generator):
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("expected NCHW RGB tensor")
    if parameters.family == "identity":
        return image.clone()
    linear = srgb_to_linear(image) * (2.0 ** parameters.exposure_ev)
    if parameters.shot_noise_level > 0:
        count = 1.0 / parameters.shot_noise_level
        linear = torch.poisson(linear.clamp_min(0) * count,
                               generator=noise_generator) / count
    if parameters.read_noise_std > 0:
        linear = linear + torch.randn(
            linear.shape, dtype=linear.dtype, device=linear.device,
            generator=noise_generator
        ) * parameters.read_noise_std
    linear = linear.clamp_min(0)
    linear = linear * image.new_tensor(parameters.white_balance).view(1, 3, 1, 1)
    linear = torch.einsum("ij,bjhw->bihw",
                          image.new_tensor(parameters.color_matrix), linear)
    output = linear_to_srgb(linear.clamp(0, 1))
    if parameters.quantization_bits > 0:
        levels = float((1 << parameters.quantization_bits) - 1)
        output = torch.round(output * levels) / levels
    return output.clamp(0, 1)


class PhysicalBatchAugmenter:
    """Owns all Physical RNG and advances once independently per batch sample."""

    def __init__(self, seed, mode="full", identity_probability=0.5, device="cpu"):
        self.mode = mode
        self.identity_probability = float(identity_probability)
        self.parameter_rng = random.Random(int(seed))
        self.noise_generator = torch.Generator(device=torch.device(device).type)
        self.noise_generator.manual_seed(int(seed) + 1)
        self.counts = {name: 0 for name in ("identity", *FAMILIES)}
        self.last_parameters = []

    def __call__(self, batch):
        outputs, parameters = [], []
        for sample in batch:
            sampled = sample_frozen_parameters(
                self.parameter_rng, self.mode, self.identity_probability
            )
            outputs.append(apply_physical_degradation(
                sample.unsqueeze(0), sampled, self.noise_generator
            ))
            parameters.append(sampled.to_dict())
            self.counts[sampled.family] += 1
        self.last_parameters = parameters
        return torch.cat(outputs, dim=0), parameters

    def state_dict(self):
        return {"parameter_rng": self.parameter_rng.getstate(),
                "noise_generator": self.noise_generator.get_state(),
                "counts": dict(self.counts)}

    def load_state_dict(self, state):
        self.parameter_rng.setstate(state["parameter_rng"])
        noise_state = state["noise_generator"]
        if not isinstance(noise_state, torch.Tensor):
            noise_state = torch.as_tensor(noise_state)
        noise_state = noise_state.detach().to(
            device="cpu", dtype=torch.uint8
        ).contiguous()
        self.noise_generator.set_state(noise_state)
        self.counts = dict(state["counts"])

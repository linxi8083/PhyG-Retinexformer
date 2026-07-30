"""Checkpointable replay of the author's frozen random-gamma augmentation."""

import random


class GammaReplay:
    """One integer gamma draw per batch, matching author train.py lines 52-55."""

    def __init__(self, seed, start_percent=60, end_percent=120):
        self.start_percent = int(start_percent)
        self.end_percent = int(end_percent)
        if self.start_percent > self.end_percent:
            raise ValueError("gamma start_percent exceeds end_percent")
        self.parameter_rng = random.Random(int(seed))
        self.last_gamma = None

    def __call__(self, batch):
        gamma = self.parameter_rng.randint(
            self.start_percent, self.end_percent
        ) / 100.0
        self.last_gamma = gamma
        return batch ** gamma, gamma

    def state_dict(self):
        return {
            "parameter_rng": self.parameter_rng.getstate(),
            "last_gamma": self.last_gamma,
        }

    def load_state_dict(self, state):
        self.parameter_rng.setstate(state["parameter_rng"])
        self.last_gamma = state["last_gamma"]

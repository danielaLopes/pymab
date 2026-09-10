"""Uniform random bandit policy."""

from __future__ import annotations

import numpy as np

from pymab.policies.policy import ActionValuePolicy


class RandomPolicy(ActionValuePolicy):
    """Select each action uniformly at random."""

    def __init__(self, *, n_arms: int) -> None:
        super().__init__(n_arms=n_arms, initial_value=0.0)

    def select_action(self, *, rng: np.random.Generator) -> int:
        return int(rng.integers(self.n_arms))

    def __repr__(self) -> str:
        return "RandomPolicy()"


__all__ = ["RandomPolicy"]

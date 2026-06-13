"""Uniform random bandit policy."""

from __future__ import annotations

import numpy as np

from pymab.policies.policy import ActionValuePolicy


class RandomPolicy(ActionValuePolicy):
    """Select each action uniformly at random."""

    def __init__(
        self,
        *,
        n_arms: int | None = None,
        n_bandits: int | None = None,
        **_: object,
    ) -> None:
        arms = n_arms if n_arms is not None else n_bandits
        if arms is None:
            raise TypeError("n_arms is required")
        super().__init__(n_arms=int(arms), initial_value=0.0)

    def select_action(self, *, rng: np.random.Generator) -> int:
        return int(rng.integers(self.n_arms))

    def __repr__(self) -> str:
        return "RandomPolicy()"

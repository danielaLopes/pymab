"""Epsilon-greedy bandit policy."""

from __future__ import annotations

import numpy as np

from pymab.policies.greedy import GreedyPolicy
from pymab.policies.policy import choose_argmax, validate_probability


class EpsilonGreedyPolicy(GreedyPolicy):
    """Explore uniformly with probability epsilon, otherwise exploit."""

    def __init__(
        self,
        *,
        n_arms: int | None = None,
        n_bandits: int | None = None,
        initial_value: float | None = None,
        optimistic_initialization: float = 0.0,
        epsilon: float = 0.1,
        **kwargs: object,
    ) -> None:
        validate_probability(epsilon, name="epsilon")
        super().__init__(
            n_arms=n_arms,
            n_bandits=n_bandits,
            initial_value=initial_value,
            optimistic_initialization=optimistic_initialization,
            **kwargs,
        )
        self.epsilon = float(epsilon)

    def select_action(self, *, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_arms))
        return choose_argmax(self.estimates, rng)

    def __repr__(self) -> str:
        return (
            "EpsilonGreedyPolicy("
            f"epsilon={self.epsilon}, initial_value={self.initial_value})"
        )

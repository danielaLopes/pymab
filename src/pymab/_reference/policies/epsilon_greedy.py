"""Epsilon-greedy bandit policy."""

from __future__ import annotations

import numpy as np

from pymab._reference.policies.greedy import GreedyPolicy
from pymab.policies.policy import (
    choose_argmax,
    validate_positive,
    validate_probability,
)


class EpsilonGreedyPolicy(GreedyPolicy):
    """Explore uniformly with probability epsilon, otherwise exploit."""

    def __init__(
        self,
        *,
        n_arms: int,
        initial_value: float = 0.0,
        epsilon: float = 0.1,
    ) -> None:
        validate_probability(epsilon, name="epsilon")
        super().__init__(
            n_arms=n_arms,
            initial_value=initial_value,
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


class DecayingEpsilonGreedyPolicy(GreedyPolicy):
    """Epsilon-greedy policy with a monotone exploration schedule.

    The schedule is hyperbolic by default:
    ``epsilon_t = min_epsilon + (initial_epsilon - min_epsilon) / (1 + decay_rate * t)``.
    This gives users an intuitive policy that explores heavily early and settles
    into exploitation without fully disabling exploration unless ``min_epsilon``
    is set to zero.
    """

    def __init__(
        self,
        *,
        n_arms: int,
        initial_value: float = 0.0,
        initial_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        decay_rate: float = 0.01,
    ) -> None:
        validate_probability(initial_epsilon, name="initial_epsilon")
        validate_probability(min_epsilon, name="min_epsilon")
        if min_epsilon > initial_epsilon:
            raise ValueError("min_epsilon must be <= initial_epsilon")
        validate_positive(decay_rate, name="decay_rate", allow_zero=True)
        super().__init__(
            n_arms=n_arms,
            initial_value=initial_value,
        )
        self.initial_epsilon = float(initial_epsilon)
        self.min_epsilon = float(min_epsilon)
        self.decay_rate = float(decay_rate)

    @property
    def epsilon(self) -> float:
        """Current exploration probability."""

        decayed = self.min_epsilon + (
            (self.initial_epsilon - self.min_epsilon)
            / (1.0 + self.decay_rate * self.step)
        )
        return float(max(self.min_epsilon, decayed))

    def select_action(self, *, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_arms))
        return choose_argmax(self.estimates, rng)

    def __repr__(self) -> str:
        return (
            "DecayingEpsilonGreedyPolicy("
            f"initial_epsilon={self.initial_epsilon}, "
            f"min_epsilon={self.min_epsilon}, decay_rate={self.decay_rate})"
        )


__all__ = ["DecayingEpsilonGreedyPolicy", "EpsilonGreedyPolicy"]

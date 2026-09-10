"""Gradient bandit policy."""

from __future__ import annotations

import numpy as np

from pymab.policies.policy import Policy, softmax, validate_positive
from pymab.types import FloatArray


class GradientBanditPolicy(Policy):
    """Gradient bandit algorithm using action preferences."""

    def __init__(
        self,
        *,
        n_arms: int,
        learning_rate: float = 0.1,
        use_baseline: bool = True,
    ) -> None:
        validate_positive(learning_rate, name="learning_rate")
        self.learning_rate = float(learning_rate)
        self.use_baseline = bool(use_baseline)
        self.preferences: FloatArray
        self.probabilities: FloatArray
        self.step = 0
        self.average_reward = 0.0
        super().__init__(n_arms=n_arms)
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.average_reward = 0.0
        self.preferences = np.zeros(self.n_arms, dtype=float)
        self.probabilities = np.full(self.n_arms, 1.0 / self.n_arms, dtype=float)

    def select_action(self, *, rng: np.random.Generator) -> int:
        self.probabilities = softmax(self.preferences)
        return int(rng.choice(self.n_arms, p=self.probabilities))

    def update(self, *, action: int, reward: float) -> None:
        self._validate_action(action)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        self.step += 1
        baseline = self.average_reward if self.use_baseline else 0.0
        one_hot = np.zeros(self.n_arms, dtype=float)
        one_hot[action] = 1.0
        self.preferences += (
            self.learning_rate * (reward - baseline) * (one_hot - self.probabilities)
        )
        self.average_reward += (reward - self.average_reward) / self.step

    def recommend_action(self) -> int:
        return int(np.argmax(self.preferences))

    def __repr__(self) -> str:
        return (
            "GradientBanditPolicy("
            f"learning_rate={self.learning_rate}, use_baseline={self.use_baseline})"
        )


__all__ = ["GradientBanditPolicy"]
